# credits: adapted from https://github.com/anuragranj/cc/blob/master/test_flow.py

import torch
from PIL import Image
import numpy as np
import cv2
import imageio

from torchvision import transforms

import os
from path import Path
from easydict import EasyDict

import argparse
import json
from tqdm import tqdm

from models.get_model import get_model
from utils.torch_utils import restore_model
from utils.flow_utils import evaluate_flow, flow_to_image, load_flow, resize_flow, writeFlow, writeFlowKITTI
from datasets.flow_datasets import KITTIFlow, Sintel
from transforms import sep_transforms

parser = argparse.ArgumentParser(description='create_submission',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", type=str, choices=['sintel', 'kitti'], help="Dataset")
parser.add_argument("--model-folder", required=True, type=str, help='the model folder (that contains the configuration file)')
parser.add_argument("--output-dir", default=None, type=str, help="Output directory; default is test_flow under the folder of the model")
parser.add_argument("--trained-model", required=True, type=str, help="trained model path in the model folder")


@torch.no_grad()
def create_sintel_submission(model, args):
    """ Create submission for the Sintel leaderboard """
    
    input_transform = transforms.Compose([
        sep_transforms.Zoom(args.img_height, args.img_width),
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    # start inference
    model.eval()
    for dstype in ['clean', 'final']:
        ds_dir = args.output_dir / dstype
        ds_dir.makedirs_p()
        viz_dir = args.output_dir / dstype + '_viz'
        viz_dir.makedirs_p()
        
        dataset = Sintel(args.root_sintel, type=dstype, split='test', with_flow=False, transform=input_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
        
        for i_step, data in enumerate(data_loader):
            def tensor2array(tensor):
                return tensor.detach().cpu().numpy().transpose([0, 2, 3, 1])

            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).cuda()

            # compute output
            flows = model(img_pair)['flows_fw']
            flow_pred = flows[0]
            
            b, h, w, _ = data['target']['flow'].shape
            flow_pred_up = resize_flow(flow_pred, (h, w))
            
            info = data['img1_path'][0].splitall()
            scene, filename = info[-2:]
            file_id = filename[-8:-4]
            output_file = ds_dir / scene / 'frame{}.flo'.format(file_id)
            writeFlow(output_file, tensor2array(flow_pred_up)[0])
            
            viz_file = viz_dir / scene / filename
            (viz_dir / scene).makedirs_p()
            flow_viz = flow_to_image(tensor2array(flow_pred_up)[0])
            imageio.imwrite(viz_file, flow_viz)

    print('Completed!')
    return      


@torch.no_grad()
def create_kitti_submission(model, args):
    """ Create submission for the KITTI leaderboard """
    
    input_transform = transforms.Compose([
        sep_transforms.Zoom(args.img_height, args.img_width),
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])  
    
    dataset_2012 = KITTIFlow(Path(args.root_kitti12) / 'testing', subsplit='test', name='kitti2012', label_ratio=0.0, transform=input_transform)
    dataset_2015 = KITTIFlow(Path(args.root_kitti15) / 'testing', subsplit='test', name='kitti2015', label_ratio=0.0, transform=input_transform)
    #dataset_2012 = KITTIFlow(Path(args.root_kitti12) / 'training', subsplit='val', name='kitti2012', label_ratio=0.0, is_val=True, transform=input_transform)
    #dataset_2015 = KITTIFlow(Path(args.root_kitti15) / 'training', subsplit='val', name='kitti2015', label_ratio=0.0, is_val=True, transform=input_transform)      
    
    # start inference
    model.eval()
    for ds in [dataset_2012, dataset_2015]:
        ds_dir = args.output_dir / ds.name
        ds_dir.makedirs_p()
        viz_dir = args.output_dir / ds.name + '_viz'
        viz_dir.makedirs_p()
    
        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=True, shuffle=False)
        for i_step, data in enumerate(data_loader):
            def tensor2array(tensor):
                return tensor.detach().cpu().numpy().transpose([0, 2, 3, 1])  
    
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).cuda()    
    
            # compute output
            flows = model(img_pair)['flows_fw']
            flow_pred = flows[0]
            h, w = data['im_shape']
            h, w = h.item(), w.item()
            flow_pred_up = resize_flow(flow_pred, (h, w))
            
            filename = os.path.basename(data['img1_path'][0])
            output_file = ds_dir / 'flow' / filename
            writeFlowKITTI(output_file, tensor2array(flow_pred_up)[0])

            viz_file = viz_dir / filename
            flow_viz = flow_to_image(tensor2array(flow_pred_up)[0])
            imageio.imwrite(viz_file, flow_viz)
    
    print('Completed!')
    return      

@torch.no_grad()
def main():
    args = parser.parse_args()
    args.model_folder = Path(args.model_folder)
    
    if args.output_dir is None:
        args.output_dir = args.model_folder / 'test_flow_' + args.dataset
    args.output_dir.makedirs_p()
    
    ## set up the model
    config_file = args.model_folder / 'config.json'
    model_file = args.model_folder / args.trained_model
    with open(config_file) as f:
        cfg = EasyDict(json.load(f))
        
    model = get_model(cfg.model).cuda()

    model = restore_model(model, model_file)
    model.eval()
    
    if args.dataset == 'sintel':
        args.img_height, args.img_width = 448, 1024
        args.root_sintel = "/PATH/TO/YOUR/MPI-Sintel"
        create_sintel_submission(model, args)
    elif args.dataset == 'kitti':
        args.img_height, args.img_width = 384, 1216
        args.root_kitti12 = "/PATH/TO/YOUR/KITTI-2012"
        args.root_kitti15 = "/PATH/TO/YOUR/KITTI-2015"
        create_kitti_submission(model, args)

    
            
if __name__ == '__main__':
    main()



