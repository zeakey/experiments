import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch.datasets import ilsvrc2012
import vltools.pytorch as vlpytorch

import resnet_cifar, vgg_cifar
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--config', default='configs/basic.yml', help='path to dataset')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--num_classes', default=None, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=None, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/tmp-1")
parser.add_argument('--sparsity', type=float, default=1e-3, help='random seed')
parser.add_argument('--randseed', type=int, help='random seed', default=None)
parser.add_argument('--retrain', action="store_true")
args = parser.parse_args()

milestones = [100, 150, 180]

if args.randseed == None:
    args.randseed = np.random.randint(1000)
args.tmp = args.tmp.strip("/")
args.tmp = args.tmp+"-seed%d"%args.randseed

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

THIS_DIR = abspath(dirname(__file__))
os.makedirs(args.tmp, exist_ok=True)

tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

# loss function
criterion = torch.nn.CrossEntropyLoss()

def main():

    logger.info(args)
    train_loader, val_loader = vlpytorch.datasets.cifar100(abspath("/home/kai/.torch/data"), bs=args.bs)

    # model and optimizer
    model = vgg_cifar.vgg16_bn(num_classes=100).cuda()
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

    logger.info("Initial parameters details:")
    for name, p in model.named_parameters():
        logger.info("%s, shape=%s, std=%f, mean=%f" % (name, str(p.shape), p.std().item(), p.mean().item()))

    # records
    best_acc1 = 0

    # save initial weights
    save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'best_acc1': -1,
            }, False, path=args.tmp, filename="initial-weights.pth")

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

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                      milestones=milestones,
                      gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        bn_l1 = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_l1 += m.weight.abs().mean()

        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('train/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('train/BN-L1', bn_l1, epoch)

        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1

        if is_best:
            best_acc1 = acc1

        # log parameters details
        logger.info("Epoch %d parameters details:" % epoch)

        for name, p in model.named_parameters():
            logger.info("%s, shape=%s, std=%f, mean=%f" % \
                    (name, str(p.shape), p.std().item(), p.mean().item()))

        logger.info("Best acc1=%.5f" % best_acc1)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

    # evaluate before pruning
    logger.info("evaluating before pruning...")
    validate(val_loader, model, args.epochs)

    bnfactors = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bnfactors += m.weight.data.cpu().abs().numpy().tolist()
    bnfactors = np.sort(np.array(bnfactors))
    bnthres = bnfactors[int(len(bnfactors) * 0.7)]

    prune_mask = {}
    named_modules = dict(model.named_modules())
    modules = list(named_modules.values())
    previous_conv = None
    for name, m in named_modules.items():
        if not isinstance(m, nn.BatchNorm2d):
            continue
        bnfactor = m.weight.data.abs()
        prune_mask[name] = (bnfactor >= bnthres)
        m.weight.data[1-prune_mask[name]] = 0
        m.bias.data[1-prune_mask[name]] = 0

    logger.info("evaluating after masking...")
    validate(val_loader, model, args.epochs)

    # resume initial weights
    init_weights = torch.load(join(args.tmp, "initial-weights.pth"))
    model.load_state_dict(init_weights['state_dict'])

    # do real pruning
    for idx, (name, m) in enumerate(named_modules.items()):
        if not isinstance(m, nn.BatchNorm2d):
            continue

        previous_conv = modules[idx-1]
        
        nextid = idx+1
        next_conv = modules[nextid]
        while not (isinstance(next_conv, nn.Conv2d) or isinstance(next_conv, nn.Linear)):
            next_conv = modules[nextid]
            nextid += 1

        assert isinstance(previous_conv, nn.Conv2d), type(previous_conv)
        assert isinstance(next_conv, nn.Conv2d) or isinstance(next_conv, nn.Linear), type(next_conv)

        mask = prune_mask[name]

        m.weight.data = m.weight.data[mask]
        m.bias.data = m.bias.data[mask]
        m.running_mean = m.running_mean[mask]
        m.running_var = m.running_var[mask]

        previous_conv.weight.data = previous_conv.weight.data[mask]
        previous_conv.bias.data = previous_conv.bias.data[mask]

        next_conv.weight.data = next_conv.weight.data[:, mask]

    logger.info("evaluating after real pruning...")
    validate(val_loader, model, args.epochs)

    if args.retrain:

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                      milestones=milestones,
                      gamma=0.1)
        best_acc1 = 0
        for epoch in range(0, args.epochs):

            # train and evaluate
            loss = train(train_loader, model, optimizer, epoch)
            acc1, acc5 = validate(val_loader, model, epoch)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]

            tfboard_writer.add_scalar('retrain0/loss_epoch', loss, epoch)
            tfboard_writer.add_scalar('retrain0/lr_epoch', lr, epoch)

            tfboard_writer.add_scalar('retest0/acc1_epoch', acc1, epoch)
            tfboard_writer.add_scalar('retest0/acc5_epoch', acc5, epoch)

            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1

            if is_best:
                best_acc1 = acc1

            # log parameters details
            logger.info("Epoch %d parameters details:" % epoch)

            for name, p in model.named_parameters():
                logger.info("%s, shape=%s, std=%f, mean=%f" % \
                        (name, str(p.shape), p.std().item(), p.mean().item()))

            logger.info("Best acc1=%.5f" % best_acc1)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.tmp, filename="checkpoint-retrain0.pth")

        # reinitiate parameters and retrain
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        best_acc1 = 0
        for epoch in range(0, args.epochs):

            # train and evaluate
            loss = train(train_loader, model, optimizer, epoch)
            acc1, acc5 = validate(val_loader, model, epoch)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]

            tfboard_writer.add_scalar('retrain1/loss_epoch', loss, epoch)
            tfboard_writer.add_scalar('retrain1/lr_epoch', lr, epoch)

            tfboard_writer.add_scalar('retest1/acc1_epoch', acc1, epoch)
            tfboard_writer.add_scalar('retest1/acc5_epoch', acc5, epoch)

            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1

            if is_best:
                best_acc1 = acc1

            # log parameters details
            logger.info("Epoch %d parameters details:" % epoch)

            for name, p in model.named_parameters():
                logger.info("%s, shape=%s, std=%f, mean=%f" % \
                        (name, str(p.shape), p.std().item(), p.mean().item()))

            logger.info("Best acc1=%.5f" % best_acc1)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.tmp, filename="checkpoint-retrain1.pth")

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
        data = data.cuda()

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # impose L1 penalty to BN factors
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(args.sparsity*torch.sign(m.weight.data))  # L1
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]["lr"]

        if i % 20 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.4f}'.format(
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

            tfboard_writer.add_scalar('test/loss_iter', loss.item(), epoch * len(val_loader)  + i)
            
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
