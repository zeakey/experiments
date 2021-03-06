import shutil, sys, os, torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from os.path import join, split, abspath, dirname, isfile, isdir
import yaml
import math
import numpy as np

def load_yaml(yaml_file):
    assert isfile(yaml_file), "File %s does'nt exist!" % yaml_file
    return yaml.load(open(yaml_file))

def merge_config(args, yaml_config):
    # data
    if hasattr(args, "data") and args.data and args.data != yaml_config["DATA"]["DIR"]:
        yaml_config["DATA"]["DIR"] = args.data

    if hasattr(args, "bs") and args.bs and args.bs != yaml_config["DATA"]["BS"]:
        yaml_config["DATA"]["BS"] = args.bs

    if hasattr(args, "num_classes") and args.num_classes and args.num_classes != yaml_config["DATA"]["NUM_CLASSES"]:
        yaml_config["DATA"]["NUM_CLASSES"] = args.num_classes

    # model
    if hasattr(args, "model") and args.model and args.model != yaml_config["MODEL"]["MODEL"]:
        yaml_config["MODEL"]["MODEL"] = args.model


    # Optimizer
    if hasattr(args, "lr") and args.lr and args.lr != yaml_config["OPTIMIZER"]["LR"]:
        yaml_config["OPTIMIZER"]["LR"] = args.lr
    
    if hasattr(args, "epochs") and args.epochs and args.epochs != yaml_config["OPTIMIZER"]["EPOCHS"]:
        yaml_config["OPTIMIZER"]["EPOCHS"] = args.epochs

    # CUDA
    if hasattr(args, "gpu") and args.gpu and args.gpu != yaml_config["CUDA"]["GPU_ID"]:
        yaml_config["CUDA"]["GPU_ID"] = args.gpu
    
    if hasattr(args, "visport") and args.visport != yaml_config["VISDOM"]["PORT"]:
        yaml_config["VISDOM"]["PORT"] = args.visport

    # MISC
    if yaml_config["MISC"]["TMP"] == "" or yaml_config["MISC"]["TMP"] is None:
        yaml_config["MISC"]["TMP"] = yaml_config["DATA"]["DATASET"] + "-" + yaml_config["MODEL"]["MODEL"]
        yaml_config["MISC"]["TMP"] = join("tmp", yaml_config["MISC"]["TMP"])

    if hasattr(args, "tmp") and args.tmp and args.tmp != yaml_config["MISC"]["TMP"]:
        yaml_config["MISC"]["TMP"] = args.tmp

    return yaml_config

class DayHourMinute(object):
  
  def __init__(self, seconds):
      
      self.days = int(seconds // 86400)
      self.hours = int((seconds- (self.days * 86400)) // 3600)
      self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)

def get_lr(epoch, base_lr, warmup_epochs=5, warmup_start_lr=0.001):
    lr = 0
    if epoch < warmup_epochs:
        lr = ((base_lr - warmup_start_lr) / warmup_epochs) * epoch
    else:
        lr = base_lr * (0.1 ** ((epoch-warmup_epochs) // 30))
    return lr

class LRScheduler(object):

    def __init__(self, base_lr, max_epochs, iters_per_epoch, gamma=0.1, warmup_epochs=5, warmup_start_lr=0.001):

        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.iters_per_epoch = iters_per_epoch
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        self._it = 0


    def step(self, step=1):

        self._it += step


def get_lr_per_iter(epoch, iter, iters_per_epoch, base_lr, stepsize=30, gamma=0.1, warmup_epochs=5, warmup_start_lr=0.001):
    lr = 0
    iter = epoch * iters_per_epoch + iter
    warmup_iters = warmup_epochs * iters_per_epoch
    stepsize_iters = stepsize * iters_per_epoch

    if iter < warmup_iters:
        lr = warmup_start_lr + ((base_lr - warmup_start_lr) / warmup_iters) * iter
    else:
        lr = base_lr * (0.1 ** ((iter-warmup_iters) // stepsize_iters))
    return lr

def set_lr(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_{t+1} = \eta_{min} + (\eta_t - \eta_{min})\frac{1 +
        \cos(\frac{T_{cur+1}}{T_{max}}\pi)}{1 + \cos(\frac{T_{cur}}{T_{max}}\pi)}
    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, restart_every=30, restart_factor=0.8, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.restart_every = restart_every
        self.restart_factor = restart_factor

        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if (self.last_epoch) % self.restart_every == 0:
            factor = self.restart_factor ** ((self.last_epoch + 1) // self.restart_every)
        else:
            factor = 1

        return [(1 + math.cos(math.pi * (self.last_epoch % self.restart_every) / self.restart_every)) /
                (1 + math.cos(math.pi * ((self.last_epoch - 1)%self.restart_every) / self.restart_every)) *
                (group['lr'] - self.eta_min) * factor + self.eta_min 
                for group in self.optimizer.param_groups]
