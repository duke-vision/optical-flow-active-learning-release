import copy
from torchvision import transforms

import numpy as np
from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from datasets.flow_datasets import FlyingChairs, FlyingThings3D
from datasets.flow_datasets import SintelRaw, Sintel
from datasets.flow_datasets import KITTIRawFile, KITTIFlow, KITTIFlowMV


def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])
    
    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)

    if cfg.type == 'CHAIRS':   
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        
        train_set = FlyingChairs(cfg.root, sp_file=cfg.train_val_split_file, n_frames=2, name='FlyingChairs_train', 
                                 subsplit='train', label_ratio=cfg.label_ratio, 
                                 ap_transform=ap_transform,
                                 transform=input_transform, 
                                 target_transform={'flow': sep_transforms.ArrayToTensor()},
                                 co_transform=co_transform)

        valid_set = FlyingChairs(cfg.root, sp_file=cfg.train_val_split_file, n_frames=2, name='FlyingChairs_val',
                                 subsplit='val', 
                                 transform=input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()})
        
    elif cfg.type == 'THINGS3D':   
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        
        train_set = FlyingThings3D(cfg.root, n_frames=2, name='FlyingThings3D_train', split='train', label_ratio=cfg.label_ratio, 
                                 ap_transform=ap_transform,
                                 transform=input_transform, 
                                 target_transform={'flow': sep_transforms.ArrayToTensor()},
                                 co_transform=co_transform)

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set = FlyingThings3D(cfg.root, n_frames=2, name='FlyingThings3D_val', split='val', 
                                 transform=valid_input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()})
            
    elif cfg.type == 'Sintel':
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

        train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean', name='Sintel-clean_' + cfg.train_subsplit,
                             split='training', subsplit=cfg.train_subsplit, label_ratio=cfg.label_ratio, 
                             ap_transform=ap_transform,
                             transform=input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()},
                             co_transform=co_transform
                             )
        train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final', name='Sintel-final_' + cfg.train_subsplit,
                             split='training', subsplit=cfg.train_subsplit, label_ratio=cfg.label_ratio, 
                             ap_transform=ap_transform,
                             transform=input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()},
                             co_transform=co_transform
                             )
        if not hasattr(cfg, 'score_base') or cfg.score_base != 'clean+final':
            train_set_2.set_labels(labeled_idx=train_set_1.labeled_idx)    # make sure that the clean and final set use labels for the same corresponding frames
        
        if hasattr(cfg, 'sample_scores_sorted') and cfg.sample_scores_sorted is not None:  # request samples based on the input sample scores
            scores = np.genfromtxt(cfg.sample_scores_sorted, delimiter='\t')
            num_labeled = int(cfg.label_ratio * len(train_set_1))
            
            if hasattr(cfg, 'double_list') and cfg.double_list:
                labeled_idx_sample = scores[:(num_labeled * 2), 0].astype(int)  # only pick the top ones
                labeled_idx = np.random.choice(labeled_idx_sample, size=num_labeled, replace=False)
                
            else:

                if not hasattr(cfg, 'sample_scores_portion'):
                    cfg.sample_scores_portion = 1

                num_sample_scores = int(num_labeled * cfg.sample_scores_portion)
                num_rand = num_labeled - num_sample_scores

                labeled_idx_sample = scores[:num_sample_scores, 0].astype(int)  # only pick the top ones
                labeled_idx_rand = np.random.choice(scores[num_sample_scores:, 0], size=num_rand, replace=False)
                labeled_idx = np.hstack((labeled_idx_sample, labeled_idx_rand))

            if hasattr(cfg, 'score_base') and cfg.score_base == 'clean+final':  # score based on clean and final separately
                labeled_idx_1 = labeled_idx[labeled_idx < len(train_set_1)]
                labeled_idx_2 = labeled_idx[labeled_idx >= len(train_set_1)] - len(train_set_1)
                train_set_1.set_labels(labeled_idx=labeled_idx_1)
                train_set_2.set_labels(labeled_idx=labeled_idx_2) 

            else:
                train_set_1.set_labels(labeled_idx=labeled_idx)    # make sure that the clean and final set use labels for the same corresponding frames
                train_set_2.set_labels(labeled_idx=labeled_idx)
            
        if hasattr(cfg, 'train_with_labeled_only') and cfg.train_with_labeled_only:  # for ablation study
            train_set_1.remove_unlabeled_samples()
            train_set_2.remove_unlabeled_samples()
                
        train_set = ConcatDataset([train_set_1, train_set_2])
        train_set.name = 'Sintel-clean+final_' + cfg.train_subsplit
        
        if hasattr(cfg, 'train_with_raw_also') and cfg.train_with_raw_also:  # for ablation study
            train_set_3 = SintelRaw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames, name='Sintel-raw',
                                    transform=input_transform, co_transform=co_transform, ap_transform=ap_transform,
                                    target_transform={'flow': sep_transforms.ArrayToTensor()})
            train_set.datasets.append(train_set_3)
            train_set.cumulative_sizes = train_set.cumsum(train_set.datasets)  # maintain the ConcatDataset class (https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset)
            
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean', name='Sintel-clean_' + cfg.val_subsplit,
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final', name='Sintel-final_' + cfg.val_subsplit,
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()}
                             )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])

    elif cfg.type == 'Sintel_Raw':
        train_set = SintelRaw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames, name='Sintel-raw',
                              transform=input_transform, co_transform=co_transform)
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
        valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean', name='Sintel-clean_val',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()}
                             )
        valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final', name='Sintel-final_val',
                             split='training', subsplit=cfg.val_subsplit,
                             transform=valid_input_transform,
                             target_transform={'flow': sep_transforms.ArrayToTensor()}
                             )
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
        
    elif cfg.type == 'KITTI':
        train_input_transform = copy.deepcopy(input_transform)

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.train_n_frames, name='KITTI-2015_' + cfg.train_subsplit, subsplit=cfg.train_subsplit, label_ratio=cfg.label_ratio, is_val=False,
                                transform=train_input_transform, ap_transform=ap_transform, co_transform=co_transform, 
                                target_transform={'flow': sep_transforms.ArrayToTensor(), 'mask': sep_transforms.ArrayToTensor()} )
        train_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.train_n_frames, name='KITTI-2012_' + cfg.train_subsplit, subsplit=cfg.train_subsplit, label_ratio=cfg.label_ratio, is_val=False,
                                transform=train_input_transform, ap_transform=ap_transform, co_transform=co_transform, 
                                target_transform={'flow': sep_transforms.ArrayToTensor(), 'mask': sep_transforms.ArrayToTensor()} )
        
        if hasattr(cfg, 'sample_scores_sorted') and cfg.sample_scores_sorted is not None:  # request samples based on the input sample scores
            scores = np.genfromtxt(cfg.sample_scores_sorted, delimiter='\t')
            num_labeled = int(cfg.label_ratio * (len(train_set_1) + len(train_set_2)))
            labeled_idx = scores[:num_labeled, 0].astype(int)  # only pick the top ones
            
            labeled_idx_1 = labeled_idx[labeled_idx < len(train_set_1)]
            labeled_idx_2 = labeled_idx[labeled_idx >= len(train_set_1)] - len(train_set_1)
            
            train_set_1.set_labels(labeled_idx=labeled_idx_1)
            train_set_2.set_labels(labeled_idx=labeled_idx_2)
        
        
        train_set = ConcatDataset([train_set_1, train_set_2])
        train_set.name = 'KITTI-2015+2012_' + cfg.train_subsplit
        
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        # The __getitem__ function is overwritten for validation (to take care of EPE_noc), so no need to specify target_transform here 
        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames, name='KITTI-2015_' + cfg.val_subsplit, is_val=True,
                                subsplit=cfg.val_subsplit, transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames, name='KITTI-2012_' + cfg.val_subsplit, is_val=True,
                                subsplit=cfg.val_subsplit, transform=valid_input_transform)
        valid_set = ConcatDataset([valid_set_1, valid_set_2])   
    
    elif cfg.type == 'KITTI_PSEUDO':
        train_input_transform = copy.deepcopy(input_transform)

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set_1 = KITTIFlowMV(cfg.root_kitti15, cfg.train_n_frames, name='KITTI-2015MV_train',
                                  load_pseudo_label = True, label_ratio = cfg.label_ratio, subsplit='train', left_view_only=True,
                                  transform=train_input_transform, ap_transform=ap_transform, co_transform=co_transform,
                                  target_transform={'flow': sep_transforms.ArrayToTensor()} )
        train_set_2 = KITTIFlowMV(cfg.root_kitti12, cfg.train_n_frames, name='KITTI-2012MV_train',
                                  load_pseudo_label = True, label_ratio = cfg.label_ratio, subsplit='train', left_view_only=True,
                                  transform=train_input_transform, ap_transform=ap_transform, co_transform=co_transform,
                                  target_transform={'flow': sep_transforms.ArrayToTensor()} )
        
        if hasattr(cfg, 'sample_scores_sorted') and cfg.sample_scores_sorted is not None:  # request samples based on the input sample scores
            scores = np.genfromtxt(cfg.sample_scores_sorted, delimiter='\t')
            num_labeled = int(cfg.label_ratio * (len(train_set_1) + len(train_set_2)))
            labeled_idx = scores[:num_labeled, 0].astype(int)  # only pick the top ones
            
            labeled_idx_1 = labeled_idx[labeled_idx < len(train_set_1)]
            labeled_idx_2 = labeled_idx[labeled_idx >= len(train_set_1)] - len(train_set_1)
            
            train_set_1.set_labels(labeled_idx=labeled_idx_1)
            train_set_2.set_labels(labeled_idx=labeled_idx_2)
        
        
        train_set = ConcatDataset([train_set_1, train_set_2])
        train_set.name = 'KITTI-2015+2012MV_train'
        
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        # The __getitem__ function is overwritten for validation (to take care of EPE_noc), so no need to specify target_transform here 
        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames, name='KITTI-2015_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames, name='KITTI-2012_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set = ConcatDataset([valid_set_1, valid_set_2])          

    elif cfg.type == 'KITTI_Raw':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIRawFile(cfg.root, cfg.train_file, cfg.train_n_frames, name='KITTI-raw',
                                 transform=train_input_transform, ap_transform=ap_transform, co_transform=co_transform) # no target here

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames, name='KITTI-2015_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames, name='KITTI-2012_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
        
    elif cfg.type == 'KITTI_MV':
        train_input_transform = copy.deepcopy(input_transform)

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        if cfg.kitti15_only:
            train_set = KITTIFlowMV(
                cfg.root_kitti15,
                cfg.train_n_frames,
                name='KITTI-2015MV',
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
        
        else:
            train_set_1 = KITTIFlowMV(
                cfg.root_kitti15,
                cfg.train_n_frames,
                name='KITTI-2015MV',
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set_2 = KITTIFlowMV(
                cfg.root_kitti12,
                cfg.train_n_frames,
                name='KITTI-2012MV',
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set = ConcatDataset([train_set_1, train_set_2])
            train_set.name = 'KITTI-2015+2012MV'

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        # The __getitem__ function is overwritten for validation (to take care of EPE_noc), so no need to specify target_transform here 
        valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames, name='KITTI-2015_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames, name='KITTI-2012_val',
                                subsplit='val', transform=valid_input_transform)
        valid_set = ConcatDataset([valid_set_1, valid_set_2])
        
    else:
        raise NotImplementedError(cfg.type)
    return train_set, valid_set