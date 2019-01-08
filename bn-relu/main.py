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
import resnet_bnrelu, resnet_relubn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', help='model type: relubn | bnrelu', default="relubn")
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
parser.add_argument('--epochs', default=180, type=int, metavar='N', help='Maximal epochs')
parser.add_argument('--tmp', help='tmp folder', default=None)
parser.add_argument('--benchmark', dest='benchmark', action="store_true")
parser.add_argument('--gpu', default=None, type=int, metavar='N', help='GPU ID')
args = parser.parse_args()

THIS_DIR = abspath(dirname(__file__))
if args.tmp is None:
    args.tmp = join(THIS_DIR, args.model)
os.makedirs(args.tmp, exist_ok=True)

logger = Logger(join(args.tmp, "log.txt"))

# model and optimizer

if args.model == "relubn":
    model = resnet_relubn.resnet18().cuda(device=args.gpu)
elif args.model == "bnrelu":
    model = resnet_bnrelu.resnet18().cuda(device=args.gpu)
else:
    raise ValueError("Unknown model: %s" % args.model)
# model = resnet_relubn.resnet18()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

# loss function
criterion = torch.nn.CrossEntropyLoss()

def main():

    # dataset
    train_loader, val_loader = vlpytorch.cifar10('/home/kai/.torch/data', 128)

    # records
    best_acc1 = 0
    acc1_record = []
    acc5_record = []
    loss_record = []
    lr_record = []

    start_time = time.time()
    for epoch in range(180):

        # train and evaluate
        loss = train(train_loader, epoch)
        acc1, acc5 = validate(val_loader)

        # adjust learning rate
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
            }, is_best, path=args.tmp)

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
        plt.savefig(join(args.tmp, 'record.pdf'))
        plt.close(fig)

        record = dict({'acc1': np.array(acc1_record), 'acc5': np.array(acc5_record),
                       'loss_record': np.array(loss_record), "lr_record": np.array(lr_record)})

        savemat(join(args.tmp, 'record.mat'), record)

        t = time.time() - start_time           # total seconds from starting
        hours_per_epoch = (t // 3600) / (epoch + 1 - args.start_epoch)
        elapsed = utils.DayHourMinute(t)
        t /= (epoch + 1) - args.start_epoch    # seconds per epoch
        t = (args.epochs - epoch - 1) * t      # remaining seconds
        remaining = utils.DayHourMinute(t)

        logger.info("Epoch {0}/{1} finishied, {2} hours per epoch on average.\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, args.epochs, hours_per_epoch, elapsed=elapsed, remaining=remaining))

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
        target = target.cuda(args.gpu, non_blocking=True)
        data = data.cuda(args.gpu)
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

        lr = optimizer.param_groups[0]["lr"]

        if i % 20 == 0:

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.5f}'.format(
                   epoch, i, len(train_loader),
                   batch_time=batch_time, loss=losses, top1=top1, top5=top5, lr=lr))

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

            target = target.cuda(args.gpu, non_blocking=True)
            data = data.cuda(args.gpu)
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
            if i % 20 == 0:
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
