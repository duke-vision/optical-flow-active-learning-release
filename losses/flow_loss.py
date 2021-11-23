import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / (occu_mask1.mean() + 1e-6)

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.edge_aware_alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target, occ_aware=True):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_ph_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            if not occ_aware:
                occu_mask1 = torch.ones_like(occu_mask1)
                
            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.
                
            if i == 0:  # for analysis
                self.l_ph_0 = loss_warp
                self.l_ph_L1_map_0 = (im1_scaled - im1_recons).abs().mean(dim=1)

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_ph_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        total_loss = warp_loss + smooth_loss

        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()

    

class FlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(FlowLoss, self).__init__()
        self.cfg = cfg  
        
    def charbonnier_one_scale(self, flow, flow_gt_scaled, mask=None):
        if not hasattr(self.cfg, 'sup_type') or self.cfg.sup_type == 'L2':
            loss_map = ((flow - flow_gt_scaled) ** 2 + self.cfg.charbonnier_epsilon) ** self.cfg.charbonnier_q
        elif self.cfg.sup_type == 'L1':
            loss_map = ((flow - flow_gt_scaled).abs() + self.cfg.charbonnier_epsilon) ** self.cfg.charbonnier_q
        else:
            raise NotImplementedError(self.cfg.sup_type)
            
        if mask is None:
            return loss_map.mean()
        else:
            loss_map = loss_map.mean(dim=1)
            valid = mask > 0.5
            return loss_map.view(-1)[valid.view(-1)].sum() / (valid.sum() + 1e-6)
        
        
    def forward(self, output, img_pair, flow_gt, mask=None):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param img_pair: image pairs B x 6 x H x W
        :param flow_gt: ground-truth flows [B x 2 x h x w]
        :return:
        """
        
        pyramid_flows = [flow[:, :2, :, :] for flow in output] # we only use the forward flow to compute sup loss
        im1_origin = img_pair[:, :3]
        im2_origin = img_pair[:, 3:]
        
        pyramid_sup_losses = []
        
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_sup_scales[i] == 0:
                pyramid_sup_losses.append(0)
                continue
                
            h, w = flow.shape[-2:]
            H, W = flow_gt.shape[-2:]

            # resize flows to match the size of layer
            if mask is None:
                flow_gt_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear')
                mask_scaled = None
            else:
                flow_gt_scaled = F.interpolate(flow_gt, (h, w), mode='nearest')
                mask_scaled = F.interpolate(mask, (h, w), mode='nearest')
            
            flow_gt_scaled[:, 0] = flow_gt_scaled[:, 0] / W * w
            flow_gt_scaled[:, 1] = flow_gt_scaled[:, 1] / H * h
            
            loss = self.charbonnier_one_scale(flow, flow_gt_scaled, mask_scaled)
            pyramid_sup_losses.append(loss)

        pyramid_sup_losses = [l * w for l, w in
                               zip(pyramid_sup_losses, self.cfg.w_sup_scales)]
        total_loss = sum(pyramid_sup_losses)
        
        return total_loss, pyramid_flows[0].abs().mean()
    

    
class semiFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(semiFlowLoss, self).__init__()
        self.cfg = cfg
        self.flow_loss = FlowLoss(cfg)
        self.unflow_loss = unFlowLoss(cfg)
        
    def forward(self, output, img_pair, flow_gt, labeled_flag, mask=None):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param img_pair: image pairs B x 6 x H x W
        :param flow_gt: ground-truth flows [B x 2 x h x w]
        :param labeled_flag: a flag telling whether the sample is labeled or not
        :return:
        """
        
        b, = labeled_flag.shape
        n_lab = labeled_flag.sum().item()
        
        if n_lab > 0:   # the batch has labeled samples
            output_lab = [flow[labeled_flag == 1] for flow in output]
            img_pair_lab = img_pair[labeled_flag == 1]
            flow_gt_lab = flow_gt[labeled_flag == 1]
            mask_lab = mask[labeled_flag == 1] if mask is not None else None
            
            sup_loss, sup_flow_mean = self.flow_loss(output_lab, img_pair_lab, flow_gt_lab, mask_lab)
        else:
            sup_loss, sup_flow_mean = 0.0, 0.0
                 
        if (b - n_lab) > 0:   # the batch has unlabeled samples
            output_unlab = [flow[labeled_flag == 0] for flow in output]
            img_pair_unlab = img_pair[labeled_flag == 0]
            
            unsup_loss, warp_loss, smooth_loss, unsup_flow_mean = self.unflow_loss(output_unlab, img_pair_unlab)
            self.pyramid_occu_mask1 = self.unflow_loss.pyramid_occu_mask1    # for later ar
        else:
            unsup_loss, warp_loss, smooth_loss, unsup_flow_mean = 0.0, 0.0, 0.0, 0.0
            
        final_loss = (sup_loss * n_lab * self.cfg.w_alpha + unsup_loss * (b - n_lab)) / b
        flow_mean = (sup_flow_mean * n_lab + unsup_flow_mean * (b - n_lab)) / b
        
        return final_loss, sup_loss, unsup_loss, warp_loss, smooth_loss, flow_mean
        