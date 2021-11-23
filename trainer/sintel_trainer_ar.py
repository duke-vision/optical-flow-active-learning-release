import time
import torch
import numpy as np
from copy import deepcopy
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow, flow_to_image
from utils.misc_utils import AverageMeter
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config, resume=False):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config, resume=resume)

        self.sp_transform = RandomAffineFlow(
            self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise).to(self.device)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_sup', 'l_unsup', 'l_ph', 'l_sm', 'flow_mean', 'l_atst', 'l_ot']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        if 'stage1' in self.cfg:
            if self.i_epoch >= self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

        name_dataset = self.train_loader.dataset.name
        for i_step, data in enumerate(self.train_loader):
            if i_step >= self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
            img_pair = torch.cat([img1, img2], 1)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # run 1st pass
            res_dict = self.model(img_pair, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            
            gt_flows = data['target']['flow'].to(self.device)
            labeled_flag = data['labeled']
            
            loss, l_sup, l_unsup, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair, gt_flows, labeled_flag)
            
            if (1 - labeled_flag).sum() > 0:    # the batch has unlabeled data
                flow_ori = res_dict['flows_fw'][0].detach()
                flow_ori = flow_ori[labeled_flag == 0]
            
                if self.cfg.run_atst:
                    img1, img2 = data['img1_ph'].to(self.device), data['img2_ph'].to(self.device)
                    img1, img2 = img1[labeled_flag == 0], img2[labeled_flag == 0]
                    
                    # construct augment sample
                    noc_ori = self.loss_func.pyramid_occu_mask1[0]  # non-occluded region
                    s = {'imgs': [img1, img2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
                    st_res = self.sp_transform(deepcopy(s)) if self.cfg.run_st else deepcopy(s)
                    flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

                    # run 2nd pass
                    img_pair = torch.cat(st_res['imgs'], 1)
                    
                    # Need to use single GPU here if the number of unlabeled sample < the number of GPUs,
                    # in which case some GPUs may have no sample and thus raise errors
                    if (1 - labeled_flag).sum() < len(self.device_ids):
                        flow_t_pred = self.model.module(img_pair, with_bk=False)['flows_fw'][0]
                    else:
                        flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]

                    if not self.cfg.mask_st:
                        noc_t = torch.ones_like(noc_t)
                    l_atst = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                    l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

                    loss += self.cfg.w_ar * l_atst
                else:
                    l_atst = 0.0

                if self.cfg.run_ot:
                    img1, img2 = data['img1_ph'].to(self.device), data['img2_ph'].to(self.device)
                    img1, img2 = img1[labeled_flag == 0], img2[labeled_flag == 0]
                    
                    # run 3rd pass
                    img_pair = torch.cat([img1, img2], 1)

                    # random crop images
                    img_pair, flow_t, occ_t = random_crop(img_pair, flow_ori, 1 - noc_ori,
                                                          self.cfg.ot_size)

                    # slic 200, random select 8~16
                    if self.cfg.ot_slic:
                        img2 = img_pair[:, 3:]
                        seg_mask = run_slic_pt(img2, n_seg=200,
                                               compact=self.cfg.ot_compact, rd_select=[8, 16],
                                               fast=self.cfg.ot_fast).type_as(img2)  # Nx1xHxW
                        noise = torch.rand(img2.size()).type_as(img2)
                        img2 = img2 * (1 - seg_mask) + noise * seg_mask
                        img_pair[:, 3:] = img2

                    flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]
                    noc_t = 1 - occ_t
                    l_ot = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                    l_ot = (l_ot * noc_t).mean() / (noc_t.mean() + 1e-7)

                    loss += self.cfg.w_ar * l_ot
                else:
                    l_ot = 0.0
            
            else:
                l_atst, l_ot = 0.0, 0.0

            # update meters
            def get_value(v):
                return v.item() if type(v) == torch.Tensor else v
                
            key_meters.update(
                [get_value(loss), get_value(l_sup), get_value(l_unsup), get_value(l_ph), get_value(l_sm), 
                 get_value(flow_mean), get_value(l_atst), get_value(l_ot)],
                labeled_flag.size(0))

            # compute gradient and do optimization step
            self.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            if grad_norm >= self.cfg.max_grad_norm:
                self._log.info('Cliping gradients norm from {} to {}.'.format(grad_norm, self.cfg.max_grad_norm))
            
            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if (self.i_iter + 1) % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('train:{}/'.format(name_dataset) + name, v, self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/time_batch'.format(name_dataset), am_batch_time.val, self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/time_data'.format(name_dataset), am_data_time.val, self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/learning_rate'.format(name_dataset), self.optimizer.param_groups[0]['lr'], self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/max_gpu_mem_allocated_G'.format(name_dataset), torch.cuda.max_memory_allocated() / 1024 ** 3, self.i_iter + 1)

            if (self.i_iter + 1) % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step + 1, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        if type(self.valid_loader) is not list:
            self.valid_loader = [self.valid_loader]

        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            name_dataset = loader.dataset.name
            
            error_names = ['EPE_all']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)
                gt_flows = data['target']['flow'].numpy().transpose([0, 2, 3, 1])

                # compute output
                flows = self.model(img_pair)['flows_fw']
                pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])

                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img_pair.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

                if self.i_epoch % 10 == 0 and i_step == 0:  # always plot the first sample in tensorboard
                    if self.i_epoch == 10:
                        img_viz = (img1[0] * 255).to(dtype=torch.uint8)
                        gt_flow_viz = flow_to_image(gt_flows[0]).transpose(2, 0, 1)
                        self.summary_writer.add_image('valid_{}:{}/data_input'.format(i_set, name_dataset), img_viz, self.i_iter)
                        self.summary_writer.add_image('valid_{}:{}/gt_flow_fw'.format(i_set, name_dataset), gt_flow_viz, self.i_iter)
                    flow_fw_viz = flow_to_image(pred_flows[0]).transpose(2, 0, 1)
                    self.summary_writer.add_image('valid_{}:{}/flow_fw'.format(i_set, name_dataset), flow_fw_viz, self.i_iter)

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar('valid_{}:{}/'.format(i_set, name_dataset) + name, value, self.i_epoch)
            self.summary_writer.add_scalar('valid_{}:{}/time_batch_avg'.format(i_set, name_dataset), batch_time.avg, self.i_epoch)            
            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0] + all_error_avgs[1], name='model')

        if self.i_epoch % 50 == 0:
            self.save_model(all_error_avgs[0], name='model_ep{}'.format(self.i_epoch), is_best=False)

        return all_error_avgs, all_error_names
