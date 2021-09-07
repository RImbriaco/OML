import math
import torch.optim as optim


def create_optimizer(params, opt='adam', base_lr=5e-7):
    if opt == 'adam':
        optimizer = optim.Adam(params, base_lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(params, base_lr, momentum=0.9)
    else:
        raise NotImplementedError(opt)

    exp_decay = math.exp(-0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    return optimizer, scheduler





