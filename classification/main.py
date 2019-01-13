import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, argparse, time, shutil, visdom
from os.path import join, split, isdir, isfile, dirname, abspath

import vltools
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, ilsvrc2012, accuracy
import vltools.pytorch as vlpytorch
import utils, models

from models import msnet, msnet1, msnet2, msnet3, msnet4, resnet_sandglass
from torchvision.models import resnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--config', default='configs/basic.yml', help='path to dataset')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--visport', default=8097, type=int, metavar='N', help='Visdom port')

# by default, arguments bellow will come from a config file
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--num_classes', default=None, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=None, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=None, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=None, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=None, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default=None)
parser.add_argument('--benchmark', dest='benchmark', action="store_true")
parser.add_argument('--fp16', dest='fp16', action="store_true")
parser.add_argument('--gpu', default=None, type=int, metavar='N', help='GPU ID')

args = parser.parse_args()

assert isfile(args.config)
CONFIGS = utils.load_yaml(args.config)
CONFIGS = utils.merge_config(args, CONFIGS)

# Fix random seed for reproducibility
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(CONFIGS["MISC"]["RAND_SEED"])
torch.manual_seed(CONFIGS["MISC"]["RAND_SEED"])
torch.cuda.manual_seed_all(CONFIGS["MISC"]["RAND_SEED"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if CONFIGS["VISDOM"]["VISDOM"] == True:
    import visdom
    vis = visdom.Visdom(port=CONFIGS["VISDOM"]["PORT"])

THIS_DIR = abspath(dirname(__file__))
os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)

logger = Logger(join(CONFIGS["MISC"]["TMP"], "log.txt"))

# model and optimizer
model = CONFIGS["MODEL"]["MODEL"] + "(num_classes=%d)" % (CONFIGS["DATA"]["NUM_CLASSES"])
model = eval(model)

if CONFIGS["CUDA"]["DATA_PARALLEL"]:
    logger.info("Model Data Parallel")
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda(device=CONFIGS["CUDA"]["GPU_ID"])

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONFIGS["OPTIMIZER"]["LR"],
    momentum=CONFIGS["OPTIMIZER"]["MOMENTUM"],
    weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
)
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

scheduler = lr_scheduler.MultiStepLR(optimizer,
                      milestones=CONFIGS["OPTIMIZER"]["MILESTONES"],
                      gamma=CONFIGS["OPTIMIZER"]["GAMMA"])

if args.fp16:
    from apex.fp16_utils import FP16_Optimizer, network_to_half
    from apex import amp
    torch.backends.cudnn.benchmark = True
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=1.0, dynamic_loss_scale=False)
    model = network_to_half(model)
    amp_handle = amp.init(enabled=args.fp16)
    model = network_to_half(model)

# loss function
criterion = torch.nn.CrossEntropyLoss()

def main():

    logger.info(CONFIGS)

    # dataset
    assert isdir(CONFIGS["DATA"]["DIR"]), CONFIGS["DATA"]["DIR"]
    start_time = time.time()

    if CONFIGS["DATA"]["DATASET"] == "ilsvrc2012":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            join(CONFIGS["DATA"]["DIR"], 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        assert len(train_dataset.classes) == CONFIGS["DATA"]["NUM_CLASSES"], \
            "%d vs %d" % (len(train_dataset.classes), CONFIGS["DATA"]["NUM_CLASSES"])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=CONFIGS["DATA"]["BS"],
            shuffle=True, num_workers=8, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                join(CONFIGS["DATA"]["DIR"], 'val'),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])),
            batch_size=100, shuffle=False,
            num_workers=8, pin_memory=True)

    elif CONFIGS["DATA"]["DATASET"] == "cifar10":
        train_loader, val_loader = vlpytorch.cifar10(CONFIGS["DATA"]["DIR"], bs=CONFIGS["DATA"]["BS"])

    elif CONFIGS["DATA"]["DATASET"] == "cifar100":
        train_loader, val_loader = vlpytorch.cifar100(CONFIGS["DATA"]["DIR"], bs=CONFIGS["DATA"]["BS"])

    else:
        raise ValueError("Unknown dataset: %s. (Supported datasets: ilsvrc2012 | cifar10 | cifar100)" % CONFIGS["DATA"]["DATASET"])

    logger.info("Data loading done, %.3f sec elapsed." % (time.time() - start_time))

    # records
    best_acc1 = 0
    acc1_record = []
    acc5_record = []
    loss_record = []
    lr_record = []
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    start_time = time.time()
    for epoch in range(args.start_epoch, CONFIGS["OPTIMIZER"]["EPOCHS"]):

        # train and evaluate
        loss = train(train_loader, epoch)
        acc1, acc5 = validate(val_loader)
        scheduler.step()

        # record stats
        loss_record.append(loss)
        acc1_record.append(acc1)
        acc5_record.append(acc5)
        lr_record.append(optimizer.param_groups[0]["lr"])

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1_record)
        best_acc5 = max(acc5_record)
        logger.info("Best acc1=%.3f" % best_acc1)
        logger.info("Best acc5=%.3f" % best_acc5)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer' : optimizer.state_dict(),
            }, is_best, path=CONFIGS["MISC"]["TMP"])

        # We continously save records in case of interupt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(acc1_record, color='r', linewidth=2)
        axes[0].plot(acc5_record, color='g', linewidth=2)
        axes[0].legend(['Top1 Accuracy (Best%.3f)' % max(acc1_record), 'Top5 Accuracy (Best%.3f)' % max(acc5_record)],
                       loc="lower right")
        axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Precision")

        axes[1].plot(loss_record)
        axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[1].legend(["Loss"], loc="upper right")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")

        axes[2].plot(lr_record)
        axes[2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[2].legend(["Learning Rate"], loc="upper right")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")


        plt.tight_layout()
        plt.savefig(join(CONFIGS["MISC"]["TMP"], 'record.pdf'))
        plt.close(fig)

        if CONFIGS["VISDOM"]["VISDOM"]:

            vis.line(np.array([acc1_record, acc5_record]).transpose(),
                     np.arange(len(acc1_record)).reshape(len(acc1_record), 1).repeat(2, axis=1),
                     opts=dict({
                         "legend": ["Top1 accuracy", "Top5 accuracy"],
                         "title": "Accuracy",
                         "ytickmin": 0,
                         "ytickmax": 100,
                    }), win=1)

            vis.line(loss_record, np.arange(len(loss_record)),
                     opts=dict({
                         "title": "Loss",
                         "ytickmin": 0,
                         "fillarea": True,
                    }), win=2)

            vis.line(lr_record, np.arange(len(lr_record)),
                     opts=dict({
                         "title": "Learning rate",
                         "ytickmax": CONFIGS["OPTIMIZER"]["LR"],
                         "ytickmin": 0}),
                         win=3)

        record = dict({'acc1': np.array(acc1_record), 'acc5': np.array(acc5_record),
                       'loss_record': np.array(loss_record), "lr_record": np.array(lr_record)})

        savemat(join(CONFIGS["MISC"]["TMP"], 'record.mat'), record)

        t = time.time() - start_time           # total seconds from starting
        hours_per_epoch = (t // 3600) / (epoch + 1 - args.start_epoch)
        elapsed = utils.DayHourMinute(t)
        t /= (epoch + 1) - args.start_epoch    # seconds per epoch
        t = (CONFIGS["OPTIMIZER"]["EPOCHS"] - epoch - 1) * t      # remaining seconds
        remaining = utils.DayHourMinute(t)

        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, CONFIGS["OPTIMIZER"]["EPOCHS"], CONFIGS["MISC"]["TMP"], elapsed=elapsed, remaining=remaining))

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]["TMP"])

def train(train_loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if CONFIGS["CUDA"]["DATA_PARALLEL"]:
            target = target.cuda(non_blocking=True)
            data = data.cuda()
        else:
            target = target.cuda(device=CONFIGS["CUDA"]["GPU_ID"], non_blocking=True)
            data = data.cuda(device=CONFIGS["CUDA"]["GPU_ID"])

        if args.fp16:
            data = data.half()

        output = model(data)
        loss = criterion(output, target)

        # # adjust learning rate
        # lr = utils.get_lr_per_iter(epoch, i, len(train_loader),
        #                            base_lr=CONFIGS["OPTIMIZER"]["LR"],
        #                            warmup_epochs=CONFIGS["OPTIMIZER"]["WARMUP_EPOCHS"])
        # utils.set_lr(optimizer, lr)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]["lr"]
        if i % CONFIGS["MISC"]["LOGFREQ"] == 0:

            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.4f}'.format(
                   epoch, CONFIGS["OPTIMIZER"]["EPOCHS"], i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   lr=lr))

    return losses.avg

def validate(val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if CONFIGS["CUDA"]["DATA_PARALLEL"]:
                target = target.cuda(non_blocking=True)
                data = data.cuda()
            else:
                target = target.cuda(device=CONFIGS["CUDA"]["GPU_ID"], non_blocking=True)
                data = data.cuda(device=CONFIGS["CUDA"]["GPU_ID"])
            # compute output
            output = model(data)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, len(val_loader), loss=losses, top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
