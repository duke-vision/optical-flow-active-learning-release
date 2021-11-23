import json
import pprint
from shutil import copyfile
import datetime
import argparse

import os
from path import Path
from easydict import EasyDict

import basic_train
from logger import init_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/sintel_ft.json')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-n', '--name', default='')
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-u', '--user', default='user')
    parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--DEBUG', action='store_true')
    args = parser.parse_args()

    if args.resume is not None:
        args.resume = Path(args.resume)
        args.config = args.resume / 'config.json'
        
    with open(args.config) as f:
        cfg = EasyDict(json.load(f))
        
    cfg.train.DEBUG = args.DEBUG
        
    if args.DEBUG:
        cfg.train.update({
            'epoch_num': 10,
            'epoch_size': 2,
            'print_freq': 1,
            'record_freq': 1,
            'val_epoch_size': 1,
            'valid_size': 2
        })
        
    if args.evaluate:
        cfg.train.update({
            'epochs': 1,
            'epoch_size': -1,
            'valid_size': 0,
            'workers': 1,
            'val_epoch_size': 1,
        })

    if args.model is not None:
        cfg.train.pretrained_model = args.model
    cfg.train.n_gpu = args.n_gpu

    # store files day by day
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.resume is not None:
        cfg.save_root = args.resume
    else:
        if args.name == '':
            args.name = os.path.basename(args.config)[:-5]
        cfg.save_root = Path('/YOUR/SAVE/ROOT/' +  args.user) / curr_time + "_" + args.name
        cfg.save_root.makedirs_p()
        copyfile(args.config, cfg.save_root / 'config.json')

    # init logger
    _log = init_logger(log_dir=cfg.save_root, filename='train_' + curr_time + '.log')
    _log.info('=> slurm jobid: {}'.format(os.environ.get('SLURM_JOBID')))
    _log.info('will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)

    basic_train.main(cfg, _log, resume=args.resume is not None)
