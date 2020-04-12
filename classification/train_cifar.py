
import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vlkit import Logger, run_path
from vlkit.pytorch import save_checkpoint, AverageMeter, accuracy
from vlkit.pytorch import datasets
from torch.optim.lr_scheduler import MultiStepLR

from models import preact_resnet_cifar, vgg_cifar
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--dataset', default="cifar10", help='dataset')
parser.add_argument('--bs', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default="80,120", type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/prune")
parser.add_argument('--randseed', type=int, help='random seed', default=None)
args = parser.parse_args()

milestones = [int(i) for i in args.milestones.split(',')]


if args.randseed == None:
    args.randseed = np.random.randint(1000)

args.tmp = args.tmp.strip("/")
args.tmp = "tmp"
args.tmp = run_path(args.tmp)

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

THIS_DIR = abspath(dirname(__file__))
os.makedirs(args.tmp, exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()

tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

def main():

    logger.info(args)

    if args.dataset == "cifar10":
        train_loader, val_loader = datasets.cifar10(abspath("datasets/"), bs=args.bs)
        num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = datasets.cifar100(abspath("datasets/"), bs=args.bs)
        num_classes = 100
    elif args.dataset == "svhn":
        train_loader, val_loader = datasets.svhn(abspath("datasets/"), bs=args.bs)
        num_classes = 10
    else:
        raise ValueError("Invalid dataset %s" % args.dataset)

    # model = preact_resnet_cifar.Resnet(depth=20, num_classes=num_classes).cuda()
    model = vgg_cifar.vgg13_bn(num_classes=num_classes).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    logger.info("Model details:")
    logger.info(model)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # records
    best_acc1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            shutil.copy(args.resume, args.tmp)
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

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp)

        logger.info("Best acc1=%.5f" % best_acc1)


        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)


    logger.info("Optimization done, ALL results saved to %s." % args.tmp)



def train(train_loader, model, optimizer, epoch):
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

        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)

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

        if i % args.print_freq == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.3E}'.format(
                   epoch, args.epochs, i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   lr=lr))

    return losses.avg

def validate(val_loader, model, epoch):
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

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
