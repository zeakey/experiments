import torch, torchvision
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from scipy.io import savemat
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath, expanduser
if int(torch.__version__.split(".")[1]) >= 14:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter
import vlkit
from vlkit import get_logger
from vlkit import image as vlimage
from vlkit.pytorch import save_checkpoint, AverageMeter, accuracy
from vlkit.pytorch import datasets
import vlkit.pytorch as vlpytorch
#sys.path.insert(0, abspath("../"))
import resnetv1_cifar
from forest import Forest


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--num_classes', default=10, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=50, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')

parser.add_argument('--tmp', help='tmp folder', default="work_dir")
parser.add_argument('--benchmark', dest='benchmark', action="store_true")
parser.add_argument('--use-forest', dest='use_forest', action="store_true")


args = parser.parse_args()

THIS_DIR = abspath(dirname(__file__))
os.makedirs(args.tmp, exist_ok=True)

logger = get_logger(join(args.tmp, "log.txt"))
writer = SummaryWriter(log_dir=args.tmp)

model = resnetv1_cifar.resnet18(num_classes=100)
if args.use_forest:
    model = nn.Sequential(model, Forest(in_features=512, num_trees=8, tree_depth=7, num_classes=10))
else:
    model = nn.Sequential(model, nn.Linear(512, 100))

#model = nn.Sequential(*model)
model = model.cuda()



optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

# loss function
criterion = torch.nn.CrossEntropyLoss()

def main():

    logger.info(args)

    start_time = time.time()

    train_loader, val_loader = datasets.cifar10(path=join(expanduser("~"), ".torch", "data"), bs=args.bs)

    logger.info("Data loading done, %.3f sec elapsed." % (time.time() - start_time))

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        loss = train(train_loader, epoch)
        acc1, acc5 = validate(val_loader)

        # adjust lr
        scheduler.step()

        # record stats
        writer.add_scalar('test/acc1', acc1, epoch)
        writer.add_scalar('test/acc5', acc5, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('train/loss', loss, epoch)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

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
        target = target.cuda(non_blocking=True)
        data = data.cuda()

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

        if i % args.print_freq == 0:

            logger.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.5f}'.format(
                   epoch, args.epochs, i, len(train_loader),
                   batch_time=batch_time, loss=losses, top1=top1,
                   top5=top5, lr=optimizer.param_groups[0]["lr"]))

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
            target = target.cuda(non_blocking=True)
            data = data.cuda()
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
