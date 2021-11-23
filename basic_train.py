import torch
torch.backends.cudnn.benchmark = True
from utils.torch_utils import init_seed

from datasets.get_dataset import get_dataset
from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer


def main(cfg, _log, resume=False):
    init_seed(cfg.seed)

    _log.info("=> fetching img pairs.")
    train_set, valid_set = get_dataset(cfg)

    _log.info('{} samples found, {} train samples and {} test samples '.format(
        len(valid_set) + len(train_set),
        len(train_set),
        len(valid_set)))
    
    if type(train_set) is torch.utils.data.ConcatDataset:
        n_labeled_samples = sum([s.n_labeled_samples for s in train_set.datasets])
    else:
        n_labeled_samples = train_set.n_labeled_samples
    
    _log.info('train set has {} labeled and {} unlabeled (label ratio = {})'.format(
        n_labeled_samples,
        len(train_set) - n_labeled_samples,
        n_labeled_samples / len(train_set)))

    cfg.train.workers = max(cfg.train.workers, cfg.train.n_gpu * 4)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.workers, pin_memory=True, shuffle=True)

    max_test_batch = 8
    if type(valid_set) is torch.utils.data.ConcatDataset:
        valid_loader = [torch.utils.data.DataLoader(
            s, batch_size=min(max_test_batch, cfg.train.batch_size),
            num_workers=min(4, cfg.train.workers),
            pin_memory=True, shuffle=False) for s in valid_set.datasets]
        valid_size = sum([len(l) for l in valid_loader])
    else:
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=min(max_test_batch, cfg.train.batch_size),
            num_workers=min(4, cfg.train.workers),
            pin_memory=True, shuffle=False)
        valid_size = len(valid_loader)

    if cfg.train.epoch_size == 0:
        cfg.train.epoch_size = len(train_loader)
    if cfg.train.valid_size == 0:
        cfg.train.valid_size = valid_size
    cfg.train.epoch_size = min(cfg.train.epoch_size, len(train_loader))
    cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    model = get_model(cfg.model)
    loss = get_loss(cfg.loss)
    trainer = get_trainer(cfg.trainer)(
        train_loader, valid_loader, model, loss, _log, cfg.save_root, cfg.train, resume=resume)

    trainer.train()
