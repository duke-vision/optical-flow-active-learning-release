#!/bin/bash

user=$(whoami)

### Part one: running semi-supervised training 

#python3 train.py -c configs/chairs.json --n_gpu=2 --user=$user --name=chairs

#python3 train.py -c configs/things3d.json --n_gpu=2 --user=$user --name=things3d

#python3 train.py -c configs/kitti_raw.json --n_gpu=2 -u $user --name=kitti_raw
#python3 train.py -c configs/kitti_ft_ar.json --n_gpu=2 -u $user --name=kitti_ft_ar

#python3 train.py -c configs/sintel_raw.json --n_gpu=2 -u $user --sintel_raw
#python3 train.py -c configs/sintel_ft_ar.json --n_gpu=2 -u $user --name=sintel_ft_ar

### Part two: running active learning

#python3 train.py -c configs/kitti_ft_ar_active.json --n_gpu=2 -u $user --name=kitti_ft_ar_active
#python3 train.py -c configs/sintel_ft_ar_active.json --n_gpu=2 -u $user --name=sintel_ft_ar_active
