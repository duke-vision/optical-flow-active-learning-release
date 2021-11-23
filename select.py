import pprint
import datetime
import argparse

import os
from path import Path
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchvision import transforms

import numpy as np
from transforms import sep_transforms

from datasets.flow_datasets import SintelRaw, Sintel
from datasets.flow_datasets import KITTIRawFile, KITTIFlow, KITTIFlowMV
from losses.flow_loss import unFlowLoss
from losses.loss_blocks import gradient
from models.pwclite import PWCLite
from utils.flow_utils import resize_flow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--candidate-set', default='')
    parser.add_argument('--score_method', choices=['photo_loss', 'flow_grad_norm', 'flow_norm', 'occ_ratio',
                                                   'photo_loss_no_occ_mask', 'grad_norm', 'max_softmax_corr'])
    parser.add_argument('--DEBUG', action='store_true')
    args = parser.parse_args()
    
    # get the model
    model_args = EasyDict({"n_frames": 2, "reduce_dense": True, "type": "pwclite", "upsample": True})
    model = PWCLite(model_args)
    model.cuda()
    
    ckpt_dict = torch.load(args.model)
    model.load_state_dict(ckpt_dict['state_dict'])
    
    # get the data set
    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])
    
    if args.candidate_set == 'kitti_train':
        root_kitti15 = "/PATH/TO/YOUR/KITTI-2015/training/"
        root_kitti12 = "/PATH/TO/YOUR/KITTI-2012/training/"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(384, 1216))
        candidate_set_1 = KITTIFlow(root_kitti15, name='KITTI-2015_train', subsplit='train', label_ratio=0., transform=input_transform)
        candidate_set_2 = KITTIFlow(root_kitti12, name='KITTI-2012_train', subsplit='train', label_ratio=0., transform=input_transform)
        candidate_set = ConcatDataset([candidate_set_1, candidate_set_2])
        
        candidate_loader = [torch.utils.data.DataLoader(s, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) for s in candidate_set.datasets]
    
    elif args.candidate_set == 'kitti_trainval':
        root_kitti15 = "/PATH/TO/YOUR/KITTI-2015/training/"
        root_kitti12 = "/PATH/TO/YOUR/KITTI-2012/training/"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(384, 1216))
        candidate_set_1 = KITTIFlow(root_kitti15, name='KITTI-2015_trainval', subsplit='trainval', label_ratio=0., transform=input_transform)
        candidate_set_2 = KITTIFlow(root_kitti12, name='KITTI-2012_trainval', subsplit='trainval', label_ratio=0., transform=input_transform)
        candidate_set = ConcatDataset([candidate_set_1, candidate_set_2])
        
        candidate_loader = [torch.utils.data.DataLoader(s, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) for s in candidate_set.datasets]

    elif args.candidate_set == 'kitti_mv_train':
        root_kitti15 = "/PATH/TO/YOUR/KITTI-2015/training/"
        root_kitti12 = "/PATH/TO/YOUR/KITTI-2012/training/"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(384, 1216))
        candidate_set_1 = KITTIFlowMV(root_kitti15, name='KITTI-2015MV_train', left_view_only=True, subsplit='train', label_ratio=0., transform=input_transform)
        candidate_set_2 = KITTIFlowMV(root_kitti12, name='KITTI-2012MV_train', left_view_only=True, subsplit='train', label_ratio=0., transform=input_transform)
        candidate_set = ConcatDataset([candidate_set_1, candidate_set_2])
        
        candidate_loader = [torch.utils.data.DataLoader(s, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) for s in candidate_set.datasets]
        
    elif args.candidate_set == 'sintel_train':
        root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(448, 1024))
        
        # Sintel clean and final frames correspond to each other, so we only need to request labels among final; the label for the corresponding clean set 
        # is then given at the same time.
        candidate_set = Sintel(root_sintel, type='final', name='Sintel-final_train',
                             split='training', subsplit='train', label_ratio=0., transform=input_transform )
        
        candidate_loader = [torch.utils.data.DataLoader(candidate_set, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)]
        
    elif args.candidate_set == 'sintel_val':  # for cross validation
        root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(448, 1024))
        
        # Sintel clean and final frames correspond to each other, so we only need to request labels among final; the label for the corresponding clean set 
        # is then given at the same time.
        candidate_set = Sintel(root_sintel, type='final', name='Sintel-final_val',
                             split='training', subsplit='val', label_ratio=0., transform=input_transform )
        
        candidate_loader = [torch.utils.data.DataLoader(candidate_set, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)]
       
    elif args.candidate_set == 'sintel_trainval':  # for benchmark testing
        root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(448, 1024))
        
        # Sintel clean and final frames correspond to each other, so we only need to request labels among final; the label for the corresponding clean set 
        # is then given at the same time.
        candidate_set = Sintel(root_sintel, type='final', name='Sintel-final_trainval',
                             split='training', subsplit='trainval', label_ratio=0., transform=input_transform )
        
        candidate_loader = [torch.utils.data.DataLoader(candidate_set, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)]

    elif args.candidate_set == 'sintel_train_cf':  # try mix clean and final
        root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(448, 1024))
        
        # Sintel clean and final frames correspond to each other, so we only need to request labels among final; the label for the corresponding clean set 
        # is then given at the same time.
        candidate_set_1 = Sintel(root_sintel, type='clean', name='Sintel-final_trainval',
                             split='training', subsplit='train', label_ratio=0., transform=input_transform )
        candidate_set_2 = Sintel(root_sintel, type='final', name='Sintel-final_trainval',
                             split='training', subsplit='train', label_ratio=0., transform=input_transform )
        candidate_set = ConcatDataset([candidate_set_1, candidate_set_2])
        
        candidate_loader = [torch.utils.data.DataLoader(s, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) for s in candidate_set.datasets]

    elif args.candidate_set == 'sintel_trainval_cf':  # for testing
        root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        
        input_transform.transforms.insert(0, sep_transforms.Zoom(448, 1024))
        
        # Sintel clean and final frames correspond to each other, so we only need to request labels among final; the label for the corresponding clean set 
        # is then given at the same time.
        candidate_set_1 = Sintel(root_sintel, type='clean', name='Sintel-final_trainval',
                             split='training', subsplit='trainval', label_ratio=0., transform=input_transform )
        candidate_set_2 = Sintel(root_sintel, type='final', name='Sintel-final_trainval',
                             split='training', subsplit='trainval', label_ratio=0., transform=input_transform )
        candidate_set = ConcatDataset([candidate_set_1, candidate_set_2])
        
        candidate_loader = [torch.utils.data.DataLoader(s, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) for s in candidate_set.datasets]


    # start evaluation
    model.eval()
    
    # used as a helper function to compute l_ph and occlusion
    loss_help = unFlowLoss(EasyDict({"edge_aware_alpha": 10,
          "occ_from_back": False,
          "w_l1": 0.0,
          "w_ph_scales": [1.0, 1.0, 1.0, 1.0, 0.0],
          "w_sm_scales": [1.0, 0.0, 0.0, 0.0, 0.0],
          "w_smooth": 75.0,
          "w_ssim": 0.0,
          "w_ternary": 1.0,
          "warp_pad": "border",
          "with_bk": False,
          "smooth_2nd": True}))
        
    scores = []
    for i_set, loader in enumerate(candidate_loader):
        name_dataset = loader.dataset.name
        print('Start {}'.format(name_dataset))

        for i_step, data in tqdm(enumerate(loader)):
            
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).cuda()
            
            res_dict = model(img_pair, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            
            # l_photo_mutli_scale, l_photo_last_scale
            if args.score_method == 'photo_loss':
                loss, l_ph, _, _ = loss_help(flows, img_pair)
                scores.append(l_ph.item())
            
            # flow_grad_norm
            elif args.score_method == 'flow_grad_norm':
                dx, dy = gradient(flows_12[0])
                flow_grad_norm = dx.norm(dim=1).mean() + dy.norm(dim=1).mean()
                scores.append(flow_grad_norm.item())
            
            # flow_norm
            elif args.score_method == 'flow_norm':
                flow_norm = flows_12[0].norm(dim=1).mean()
                scores.append(flow_norm.item())
            
            # occ_ratio (The `pyramid_occu_mask1` here is actually a visibility mask.)
            elif args.score_method == 'occ_ratio':
                _, _, _, _ = loss_help(flows, img_pair)
                scores.append(1 - loss_help.pyramid_occu_mask1[0].mean().item())
            
            # img_grad_norm
            elif args.score_method == 'img_grad_norm':
                dx, dy = gradient(img1)
                img_grad_norm = dx.norm(dim=1).mean() + dy.norm(dim=1).mean()
                scores.append(- img_grad_norm.item())
            
            elif args.score_method == 'photo_loss_no_occ_mask':
                _, l_ph_no_occ_mask, _, _ = loss_help(flows, img_pair, occ_aware=False)
                scores.append(l_ph_no_occ_mask.item()) 
                
            elif args.score_method == 'grad_norm':
                model.train()
                loss, _, _, _ = loss_help(flows, img_pair)
                model.zero_grad()
                loss.backward()

                grad_flat = [param_group.grad.reshape(-1) for param_group in model.parameters()]
                all_grad = torch.cat(grad_flat)
                grad_norm = all_grad.norm()

                model.eval()
                scores.append(grad_norm.item())
                
            elif args.score_method == 'max_softmax_corr':
                softmax_corr = model.corr_volumes[-1].softmax(dim=1)
                max_softmax_corr, _ = softmax_corr.max(dim=1)
                scores.append(max_softmax_corr.mean().item())           
                
            else:
                raise NotImplementedError(args.score_method)

    with open(args.model[:-8] + '_{}_{}.txt'.format(args.candidate_set, args.score_method), 'w') as fout:
        top_list = np.argsort(scores)[::-1]
        for idx in top_list:
            fout.write('{}\t{}\n'.format(idx, scores[idx]))
    
    exit()
            
        

