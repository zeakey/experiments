import torch, torchvision
import torch.nn as nn
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
import utils

from models import msnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='configs/basic.yml', help='path to dataset')
parser.add_argument('--data', metavar='DIR', default="/media/data2/dataset/ilsvrc12/", help='path to dataset')
parser.add_argument('--num_classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('--model', metavar='STR', default="resnet18",
                    help='model name (resnet18|squeezenet), default="resnet18"')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=30, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/tmp')
parser.add_argument('--benchmark', dest='benchmark', action="store_true")
parser.add_argument('--gpu', default=0, type=int, metavar='N', help='GPU ID')

parser.add_argument('--visport', default=8097, type=int, metavar='N', help='Visdom port')
args = parser.parse_args()
args.model = args.model.lower()
CONFIGS = utils.load_yaml(args.config)
CONFIGS = utils.merge_config(args, CONFIGS)

if CONFIGS["VISDOM"]["VISDOM"] == True:
    import visdom
    vis = visdom.Visdom(port=CONFIGS["VISDOM"]["PORT"])

THIS_DIR = abspath(dirname(__file__))
os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)

logger = Logger(join(CONFIGS["MISC"]["TMP"], "log.txt"))

# model and optimizer
# model = torchvision.models.resnet.resnet34(num_classes=args.num_classes)
model = msnet.MSNet34(num_classes=args.num_classes)
if CONFIGS["CUDA"]["DATA_PARALLEL"]:
    logger.info("Model Data Parallel")
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda(device=CONFIGS["CUDA"]["GPU_ID"])

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CONFIGS["OPTIMIZER"]["LR"],
    weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"],
    nesterov=False
)
criterion = torch.nn.CrossEntropyLoss()

logger.info("Model details:")
logger.info(model)

def main():

    logger.info(CONFIGS)
    # dataset
    assert isdir(CONFIGS["DATA"]["DIR"]), CONFIGS["DATA"]["DIR"]
    start_time = time.time()
    train_loader, val_loader = ilsvrc2012(CONFIGS["DATA"]["DIR"], bs=CONFIGS["OPTIMIZER"]["BS"])
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
    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate
        lr = utils.get_lr(epoch, base_lr=CONFIGS["OPTIMIZER"]["LR"])
        utils.set_lr(optimizer, lr)

        # train and evaluate
        loss_record.append(train(train_loader, epoch))
        acc1, acc5 = validate(val_loader)

        # record stats
        acc1_record.append(acc1)
        acc5_record.append(acc5)
        lr_record.append(lr)

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
                     opts=dict({"legend": ["Top1 accuracy", "Top5 accuracy"], "title": "Accuracy"}), win=1)

            vis.line(loss_record, opts=dict({"title": "Loss"}), win=2)

            vis.line(lr_record, opts=dict({"title": "Learning rate"}), win=3)

        record = dict({'acc1': np.array(acc1_record), 'acc5': np.array(acc5_record),
                       'loss_record': np.array(loss_record), "lr_record": np.array(lr_record)})

        savemat(join(args.tmp, 'precision.mat'), record)

        t = time.time() - start_time           # total seconds from starting
        t /= (epoch + 1) - args.start_epoch    # seconds per epoch
        t = (args.epochs - epoch - 1) * t      # remaining seconds
        day= t // 86400
        hour= (t- (day * 86400)) // 3600
        logger.info("Epoch %d finished, remaining %d days %d hours." % (epoch, day, hour))

    logger.info("Optimization done!")

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
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % CONFIGS["MISC"]["LOGFREQ"] == 0:

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader),
                   batch_time=batch_time, loss=losses, top1=top1, top5=top5))
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
