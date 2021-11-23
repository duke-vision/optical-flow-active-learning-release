#!/bin/bash

## Sintel

# export CKPT_FOLDER=/YOUR/EXP/FOLDER
# python3 test.py --dataset=sintel --model-folder=$CKPT_FOLDER --trained-model=model_ep1000_ckpt.pth.tar
# cd /PATH/TO/YOUR/MPI-Sintel/bundler/linux-x64
# ./bundler $CKPT_FOLDER/test_flow_sintel/clean/ $CKPT_FOLDER/test_flow_sintel/final/ $CKPT_FOLDER/test_flow_sintel/submit.lzma

## KITTI

# export CKPT_FOLDER=/YOUR/EXP/FOLDER
# python3 test.py --dataset=kitti --model-folder=$CKPT_FOLDER --trained-model=model_ep1000_ckpt.pth.tar
# cd $CKPT_FOLDER/test_flow_kitti/kitti2012/flow
# zip ../../flow2012.zip *
# cd $CKPT_FOLDER/test_flow_kitti/kitti2015
# zip -r ../kitti2015.zip *

