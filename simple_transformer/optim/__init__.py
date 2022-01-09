import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List
from .loss import TranslationLoss
from .scheduler import Scheduler


def make_optimizer(params: List[Parameter], name: str, **kwargs) -> Optimizer:
    optimizer_class = eval(name)
    optimizer = optimizer_class(params, **kwargs)
    return optimizer


def make_scheduler(optimizer: Optimizer, name: str, **kwargs) -> _LRScheduler:
    scheduler_class = eval(name)
    scheduler = scheduler_class(optimizer, **kwargs)
    return scheduler


def make_loss_function(name: str, **kwargs) -> nn.Module:
    loss_func_class = eval(name)
    loss_func = loss_func_class(**kwargs)
    return loss_func
