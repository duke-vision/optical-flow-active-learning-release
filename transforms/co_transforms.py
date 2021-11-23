import numbers
import random
import numpy as np
import scipy.ndimage as ndimage

import cv2


def get_co_transforms(aug_args):
    transforms = []
    
    # for version compatibility
    if not hasattr(aug_args, 'rescale'):
        aug_args.rescale = False
    
    if aug_args.rescale:
        if aug_args.crop:    # need to make sure when we scale down, the size is still larger than crop_size
            transforms.append(RandomRescale(**aug_args.para_rescale, crop_size=aug_args.para_crop))
        else:
            transforms.append(RandomRescale(aug_args.para_rescale, crop_size=None))
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
        if 'mask' in target:
            target['mask'] = target['mask'][y1: y1 + th, x1: x1 + tw]
        if 'flow' in target:
            target['flow'] = target['flow'][y1: y1 + th, x1: x1 + tw]
        return inputs, target


class RandomSwap(object):
    def __call__(self, inputs, target):
        n = len(inputs)
        if random.random() < 0.5:
            inputs = inputs[::-1]
            if 'mask' in target:
                target['mask'] = target['mask'][::-1]
            if 'flow' in target:
                raise NotImplementedError("swap cannot apply to flow")
        return inputs, target


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
            if 'flow' in target:
                target['flow'] = np.copy(np.fliplr(target['flow']))
                target['flow'][:, :, 0] *= -1
            if 'mask' in target:
                target['mask'] = np.copy(np.fliplr(target['mask']))
                
        return inputs, target
    
class Resize(object):
    def __init__(self, size):
        self.new_h, self.new_w = size

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        if h == self.new_h and w == self.new_w:
            return inputs, target
        
        inputs = [cv2.resize(image, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA) for image in inputs]
        if 'flow' in target:
            target['flow'] = cv2.resize(target['flow'], (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR )
            target['flow'][:, :, 0] *= self.new_w / w
            target['flow'][:, :, 1] *= self.new_h / h
        if 'mask' in target:
            target['mask'] = cv2.resize(target['mask'], (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR )
            if len(target['mask'].shape) == 2:    # cv2.resize may delete the last dimension since there is only 1 channel
                target['mask'] = target['mask'][..., None]
        
        return inputs, target

# Adapted from RAFT code augmentor: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py
class RandomRescale(object):
    def __init__(self, min_scale, max_scale, crop_size=None):
        self.min_scale, self.max_scale  = min_scale, max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.crop_size = crop_size
        
    def __call__(self, inputs, target):
        if np.random.rand() < 1 - self.spatial_aug_prob:
            return inputs, target
        
        h, w, _ = inputs[0].shape
        
        if self.crop_size is None: 
            min_scale = 0
        else:
            min_scale = np.maximum(self.crop_size[0] / float(h), self.crop_size[1] / float(w))
        
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        
        # rescale the images
        inputs = [cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for img in inputs]
        
        if 'flow' in target:
            target['flow'] = cv2.resize(target['flow'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            target['flow'] = target['flow'] * [scale_x, scale_y]
            
        if 'mask' in target:   # rescaling is not recommended for KITTI because the mask is discrete
            target['mask'] = cv2.resize(target['mask'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST )          
            if len(target['mask'].shape) == 2:    # cv2.resize may delete the last dimension since there is only 1 channel
                target['mask'] = target['mask'][..., None]
                
        return inputs, target
    
class Crop(object):
    """Crops the given PIL.Image at a specified location
    used to crop the masked region of KITTI
    """

    def __init__(self, idx):
        self.y1, self.y2, self.x1, self.x2 = idx

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        
        x1 = 0 if self.x1 is None else self.x1
        x2 = w if self.x2 is None else self.x2
        y1 = 0 if self.y1 is None else self.y1
        y2 = h if self.y2 is None else self.y2        
        
        if x1 == 0 and x2 == w and y1 == 0 and y2 == h:
            return inputs, target

        inputs = [img[y1:y2, x1:x2] for img in inputs]
        if 'mask' in target:
            target['mask'] = target['mask'][y1:y2, x1:x2]
        if 'flow' in target:
            target['flow'] = target['flow'][y1:y2, x1:x2]
        return inputs, target