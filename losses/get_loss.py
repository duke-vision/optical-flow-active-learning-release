from .flow_loss import unFlowLoss, FlowLoss, semiFlowLoss

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'flow':
        loss = FlowLoss(cfg)
    elif cfg.type == 'semiflow':
        loss = semiFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
