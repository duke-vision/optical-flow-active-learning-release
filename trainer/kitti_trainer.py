import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from utils.flow_utils import load_flow, evaluate_flow, flow_to_image
from utils.misc_utils import AverageMeter


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config, resume=False):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config, resume=resume)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
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
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            res_dict = self.model(img_pair, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair)

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
                              img_pair.size(0))

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
            
            error_names = ['EPE_all', 'EPE_noc', 'EPE_occ', 'Fl_all', 'Fl_noc']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)

                res = list(map(load_flow, data['flow_occ']))
                gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
                res = list(map(load_flow, data['flow_noc']))
                _, noc_masks = [r[0] for r in res], [r[1] for r in res]

                gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                            flow, occ_mask, noc_mask in
                            zip(gt_flows, occ_masks, noc_masks)]

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

                if self.i_epoch % 10 == 1 and i_step == 0:  # always plot the first sample in tensorboard
                    if self.i_epoch == 1:
                        img_viz = (img1[0] * 255).to(dtype=torch.uint8)
                        gt_flow_viz = flow_to_image(gt_flows[0]).transpose(2, 0, 1)
                        self.summary_writer.add_image('valid_{}:{}/data_input'.format(i_set, name_dataset), img_viz, self.i_iter)
                        self.summary_writer.add_image('valid_{}:{}/gt_flow_fw'.format(i_set, name_dataset), gt_flow_viz, self.i_iter)
                    flow_fw_viz = flow_to_image(flows[0][0, ...].detach().cpu().numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
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
            self.save_model(all_error_avgs[0], name='model')
            
        if self.i_epoch % 100 == 0:
            self.save_model(all_error_avgs[0], name='model_ep{}'.format(self.i_epoch), is_best=False)

        return all_error_avgs, all_error_names
