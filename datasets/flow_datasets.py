import imageio
import numpy as np
import random

import os
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow

from glob import glob


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, name='', label_ratio=0.0, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.name = name
        self.label_ratio = label_ratio
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()
        self.set_labels(label_ratio=label_ratio)  # can also be overwritten later if we want to specify which samples to label

    @abstractmethod
    def collect_samples(self):
        pass

    def set_labels(self, label_ratio=0, labeled_idx=None):
        if labeled_idx is None:    # randomly assign labels
            n_samples = len(self.samples)
            n_labeled_samples = int(n_samples * label_ratio)
            labeled_idx = np.random.choice(n_samples, size=n_labeled_samples, replace=False)
        
        for idx in range(len(self.samples)):
            if idx in labeled_idx:
                self.samples[idx]['labeled'] = True
            else:
                self.samples[idx]['labeled'] = False
            
        self.labeled_idx = labeled_idx
        self.n_labeled_samples = len(labeled_idx)
        
    def remove_unlabeled_samples(self):
        # used to remove all unlabeled samples from the data set (for example, if we want to do sup instead of semi-sup for stage 2)
        new_samples = []
        for idx in range(len(self.samples)):
            if self.samples[idx]['labeled']:
                new_samples.append(self.samples[idx])
                
        self.samples = new_samples
        self.labeled_idx = np.array(range(len(self.samples)))
                
        
    def _load_sample(self, s):
        images = s['imgs']
        images = [imageio.imread(self.root / p).astype(np.float32) for p in images]

        target = {}
        if 'flow' in s and s['labeled']:
            if hasattr(self, "pseudo_label_folder") and self.pseudo_label_folder is not None:   # the pseudo labels are outside of this data folder
                target['flow'] = load_flow(self.pseudo_label_folder / s['flow'])
            else:   
                target['flow'] = load_flow(self.root / s['flow'])
        else:
            h, w, _= images[0].shape
            target['flow'] = (np.ones((h, w, 2)) * np.nan).astype(np.float32)
                
        return images, target, s['labeled']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target, labeled = self._load_sample(self.samples[idx])

        if self.co_transform is not None:
            images, target = self.co_transform(images, target)
            
        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
                
        data['img1_path'] = self.samples[idx]['imgs'][0]
        data['img2_path'] = self.samples[idx]['imgs'][1]
        data['target'] = target
        data['labeled'] = labeled
        return data

class FlyingChairs(ImgSeqDataset):  # 384 x 512
    def __init__(self, root, sp_file, n_frames=2, name='', label_ratio=1.0, subsplit='train', ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.sp_file = sp_file
        self.subsplit = subsplit
        super(FlyingChairs, self).__init__(root, n_frames, name, label_ratio, input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        
        split_no = [int(i) for i in open(Path(self.root) / self.sp_file, 'r').readlines()]
        if self.subsplit == 'train':
            frame_ids, = np.where(np.array(split_no) == 1)
            frame_ids = frame_ids + 1    # the ids start from 1
        else:
            frame_ids, = np.where(np.array(split_no) == 2)
            frame_ids = frame_ids + 1    # the ids start from 1            
        for i in list(frame_ids):
            img1_path = "data/{0:05d}_img1.ppm".format(i)
            img2_path = "data/{0:05d}_img2.ppm".format(i)
            flow_path = "data/{0:05d}_flow.flo".format(i)
            sample = {'imgs': [img1_path, img2_path],
                      'flow': flow_path}
            samples.append(sample)
            
        return samples
    
    
class FlyingThings3D(ImgSeqDataset):    # 540 x 960
    def __init__(self, root, n_frames=2, name='', label_ratio=1.0, split='train', ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.split = split
        
        # bad samples (with NaN values) to exclude
        if split == 'train':
            self.bad_samples = [119, 879, 1717, 3147, 3149, 4117, 4118, 4154, 4304, 4573, 
                                6336, 6337, 6922, 11530, 14658, 15148, 15748, 16948, 17578]
        else:
            self.bad_samples = [789]
            
        super(FlyingThings3D, self).__init__(root, n_frames, name, label_ratio, input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        
        flow_dir = Path(self.split) / 'flow/left/into_future'
        img_dir = Path(self.split) / 'image_clean/left'
        flow_list = sorted(glob(self.root / flow_dir / '*.flo'))
        
        ''' 
        # used to locate the bad samples (with NaN values) in the data set; 
        # results already included in `bad_train/val_samples`, so no need to run this again
        
        for i, flow in enumerate(flow_list):
            a = load_flow(flow)
            if np.isnan(a).sum() > 0:
                print(flow)
            if i % 100 == 0:
                print(i)
        '''

        for flow in flow_list:
            frame_id = int(flow[-11:-4])        
            if frame_id in self.bad_samples:
                continue
                
            img1_path = img_dir / "{0:07d}.png".format(frame_id)
            img2_path = img_dir / "{0:07d}.png".format(frame_id + 1)
            flow_path = flow_dir / "{0:07d}.flo".format(frame_id)
            sample = {'imgs': [img1_path, img2_path],
                      'flow': flow_path}
            samples.append(sample)
            
        return samples
    
    
class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, name='', ap_transform=None, transform=None, target_transform=None, co_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, name, label_ratio=0.0, ap_transform=ap_transform,
                                        input_transform=transform, target_transform=target_transform, co_transform=co_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        samples = []
        for scene in scene_list:
            img_list = scene.files('*.png')
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                samples.append(sample)
        return samples


class Sintel(ImgSeqDataset):
    def __init__(self, root, n_frames=2, name='', label_ratio=1.0, type='clean', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.dataset_type = type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(Sintel, self).__init__(root, n_frames, name, label_ratio, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'

        assert img_dir.isdir()
        
        samples = []
        img_list = sorted((self.root / img_dir).glob('*/*.png'))
        for img1, img2 in zip(img_list[:-1], img_list[1:]):
            info1 = img1.splitall()
            info2 = img2.splitall()
            if info1[-2] != info2[-2]:  # not the same scene
                continue
            
            scene, filename = info1[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue
            
            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}

            if self.with_flow:
                if self.n_frames == 3:
                    # for img1 img2 img3, only flow_23 will be evaluated
                    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                elif self.n_frames == 2:
                    # for img1 img2, flow_12 will be evaluated
                    s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                else:
                    raise NotImplementedError(
                        'n_frames {} with flow or mask'.format(self.n_frames))

                if self.with_flow:
                    assert s['flow'].isfile()
                
            samples.append(s)
                    
        '''
        samples = []
        for flow_map in sorted((self.root / flow_dir).glob('*/*.flo')):
            info = flow_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue

            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

                if self.with_flow:
                    if self.n_frames == 3:
                        # for img1 img2 img3, only flow_23 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                    elif self.n_frames == 2:
                        # for img1 img2, flow_12 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                    else:
                        raise NotImplementedError(
                            'n_frames {} with flow or mask'.format(self.n_frames))

                    if self.with_flow:
                        assert s['flow'].isfile()
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)
        '''

        return samples


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, root, sp_file, n_frames=2, name='', ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.sp_file = sp_file
        super(KITTIRawFile, self).__init__(root, n_frames, name, label_ratio=0.0,
                                           input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                s = {'imgs': [sp[i] for i in range(self.n_frames)]}
                samples.append(s)
            return samples


class KITTIFlowMV(ImgSeqDataset):

    def __init__(self, root, n_frames=2, name='', label_ratio=0.0, load_pseudo_label=False, left_view_only=False, subsplit='trainval',
                 transform=None, co_transform=None, ap_transform=None, target_transform=None):
        
        self.left_view_only = left_view_only
        self.subsplit = subsplit
        self.training_size = 150  # Unofficial train-val split        
        if load_pseudo_label:
            if '2015' in root:
                self.pseudo_label_folder = Path('/usr/xtmp/vision/estimations/KITTI-2015-2/flow_inference_by_RAFT')
            else:
                self.pseudo_label_folder = Path('/usr/xtmp/vision/estimations/KITTI-2012/flow_inference_by_RAFT')
        else:
            self.pseudo_label_folder = None
            
        super(KITTIFlowMV, self).__init__(root, n_frames, name, label_ratio=label_ratio,
                                          input_transform=transform,
                                          target_transform=target_transform,
                                          co_transform=co_transform,
                                          ap_transform=ap_transform)

    def collect_samples(self):
        
        flow_occ_dir = 'flow_' + 'occ'
        assert (self.root / flow_occ_dir).isdir()

        img_l_dir, img_r_dir = 'image_2', 'image_3'
        assert (self.root / img_l_dir).isdir() and (self.root / img_r_dir).isdir()
        img_dirs = [img_l_dir] if self.left_view_only else [img_l_dir, img_r_dir]

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]
            
            if self.subsplit != 'trainval':
                if self.subsplit == 'train' and int(root_filename) >= self.training_size:
                    continue
                if self.subsplit == 'val' and int(root_filename) < self.training_size:
                    continue

            for img_dir in img_dirs:
                img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
                img_list.sort()

                for st in range(0, len(img_list) - self.n_frames + 1):
                    seq = img_list[st:st + self.n_frames]
                    sample = {}
                    sample['imgs'] = []
                    for i, file in enumerate(seq):
                        frame_id = int(file[-6:-4])
                        if 12 >= frame_id >= 9:
                            break
                        sample['imgs'].append(self.root.relpathto(file))
                    
                    if self.pseudo_label_folder is not None: 
                        sample['flow'] = os.path.basename(seq[0])[:-4] + '.flo'
                        
                    if len(sample['imgs']) == self.n_frames:
                        samples.append(sample)
        return samples


class KITTIFlow(ImgSeqDataset):

    def __init__(self, root, n_frames=2, name='', label_ratio=1.0, subsplit='train', ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, is_val=False):
        self.subsplit = subsplit
        self.is_val = is_val
        self.training_size = 150  # Unofficial train-val split
        
        super(KITTIFlow, self).__init__(root, n_frames, name, label_ratio, 
                                     input_transform=transform, target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

        
    def __getitem__(self, idx):
        s = self.samples[idx]

        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        st = 1 if self.n_frames == 2 else 0
        ed = st + self.n_frames
        imgs = [s['img{}'.format(i)] for i in range(st, ed)]

        images = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]
        h, w = images[0].shape[:2]

        if self.subsplit == 'test':
            data = {
                'im_shape': (h, w),
                'img1_path': self.root / s['img1'],
            }

            images = [self.input_transform(i) for i in images]
            data.update({'img{}'.format(i + 1): images[i] for i in range(self.n_frames)})
            return data   
        
        if self.is_val:
            data = {
                'im_shape': (h, w),
                'img1_path': self.root / s['img1'],
                'flow_occ': self.root / s['flow_occ'],
                'flow_noc': self.root / s['flow_noc'],
            }

            images = [self.input_transform(i) for i in images]
            data.update({'img{}'.format(i + 1): images[i] for i in range(self.n_frames)})
            return data
        
        else:
            if s['labeled']:
                gt_flow, mask =  load_flow(self.root / s['flow_occ'])
                target = {'flow': gt_flow, 'mask': mask.astype(np.uint8)}
            else:
                target = {'flow': (np.ones((h, w, 2)) * np.nan).astype(np.float32),
                          'mask': (np.ones((h, w, 1)) * np.nan).astype(np.uint8)}
            
            if self.co_transform is not None:
                images, target = self.co_transform(images, target)
                
            if self.input_transform is not None:
                images = [self.input_transform(i) for i in images]
            data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}
                
            if self.ap_transform is not None:
                imgs_ph = self.ap_transform([data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
                for i in range(self.n_frames):
                    data['img{}_ph'.format(i + 1)] = imgs_ph[i]
            
            if self.target_transform is not None:
                for key in self.target_transform.keys():
                    target[key] = self.target_transform[key](target[key])
        
            data['img1_path'] = s['img1']
            data['img2_path'] = s['img2']
            data['target'] = target
            data['labeled'] = s['labeled']
            return data
           

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        if self.subsplit == 'test':
            img1s = sorted((self.root / 'image_2').glob('*_10.png'))
            img2s = sorted((self.root / 'image_2').glob('*_11.png'))
            
            samples = []
            for img1, img2 in zip(img1s, img2s):
                samples.append({'img1': img1, 'img2': img2})
            
            return samples
        
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        img_dir = 'image_2'
        assert (self.root / img_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]
            
            if self.subsplit != 'trainval':
                if self.subsplit == 'train' and int(root_filename) >= self.training_size:
                    continue
                if self.subsplit == 'val' and int(root_filename) < self.training_size:
                    continue

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            s = {'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map}

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s.update({'img1': img1, 'img2': img2})
            if self.n_frames == 3:
                img0 = img_dir + '/' + root_filename + '_09.png'
                assert (self.root / img0).isfile()
                s.update({'img0': img0})
            samples.append(s)
        return samples
