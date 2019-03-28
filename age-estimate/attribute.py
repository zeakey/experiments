import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataloader import UTKFaceDataset
import vltools
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
import vltools.pytorch as vlpytorch
from vltools.tcm import CosAnnealingLR
import resnet
from utils import MAE

from models.attribute_model import AttrModel

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
# by default, arguments bellow will come from a config file
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--imsize', type=int, default=224, help='Image Size')
parser.add_argument('--num_classes', default=100, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='LR', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/attribute")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dex', action="store_true", default=False)

args = parser.parse_args()
os.makedirs(args.tmp, exist_ok=True)
logger = Logger(join(args.tmp, "log.txt"))

# Fix random seed for reproducibility
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = AttrModel()
model = nn.DataParallel(model).cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = args.lr,
    weight_decay = args.wd,
    momentum = 0.9
)
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

# loss function
criterion = torch.nn.CrossEntropyLoss()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

train_dataset = UTKFaceDataset(args.data, split="train",
            transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
test_dataset = UTKFaceDataset(args.data, split="test",
            transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                           num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                           num_workers=8, pin_memory=True)

train_loader_len = len(train_loader)
scheduler = CosAnnealingLR(args.epochs*train_loader_len, lr_max=args.lr,
            lr_min=0.0001, warmup_iters=5*train_loader_len)

def main():

    logger.info(args)

    # records
    gender_acc_record = []
    race_acc_record = []
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

    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        train_loss0, train_loss1, train_gender_acc, train_race_acc = train(train_loader, epoch)
        test_loss0, test_loss1, test_gender_acc, test_race_acc = validate(test_loader)

        gender_acc_record.append(train_gender_acc)
        race_acc_record.append(train_race_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, is_best=(gender_acc_record[-1]==max(gender_acc_record)), path=args.tmp)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(gender_acc_record)
        axes[0].set_title("Gender Acc")

        axes[1].plot(race_acc_record)
        axes[1].set_title("Race Acc")

        plt.savefig(join(args.tmp, "record.pdf"))
        plt.close(fig)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

def train(train_loader, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss0 = AverageMeter()
    loss1 = AverageMeter()
    race_acc = AverageMeter()
    gender_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, _, target_gender, target_race) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_gender = target_gender.cuda(device=args.gpu, non_blocking=True)
        target_race = target_race.cuda(device=args.gpu, non_blocking=True)
        data = data.cuda(device=args.gpu)

        gender, race = model(data)
        loss0_ = criterion(gender, target_gender)
        loss1_ = criterion(race, target_race)

        # measure accuracy and record loss
        gender_acc_ = accuracy(gender, target_gender, topk=(1,))[0]
        race_acc_ = accuracy(race, target_race, topk=(1,))[0]

        loss0.update(loss0_.item(), data.size(0))
        loss1.update(loss1_.item(), data.size(0))

        gender_acc.update(gender_acc_.item(), data.size(0))
        race_acc.update(race_acc_.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        (loss0_ + loss1_).backward()
        optimizer.step()

        lr_ = scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Gender Loss {loss0.val:.3f} ({loss0.avg:.3f})\t'
                  'Race Loss {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'Gender Acc {gender_acc.val:.3f} ({gender_acc.avg:.3f})\t'
                  'Race Acc {race_acc.val:.3f} ({race_acc.avg:.3f})\t'
                  'LR: {lr:}'.format(
                   epoch, args.epochs, i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss0=loss0, loss1=loss1, 
                   gender_acc=gender_acc, race_acc=race_acc, lr=lr_))

    return loss0.avg, loss1.avg, gender_acc.avg, race_acc.avg

def validate(test_loader):

    batch_time = AverageMeter()
    loss0 = AverageMeter()
    loss1 = AverageMeter()
    race_acc = AverageMeter()
    gender_acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, _, target_gender, target_race) in enumerate(test_loader):
            
            target_gender = target_gender.cuda(device=args.gpu, non_blocking=True)
            target_race = target_race.cuda(device=args.gpu, non_blocking=True)
            data = data.cuda(device=args.gpu)

            gender, race = model(data)
            loss0_ = criterion(gender, target_gender)
            loss1_ = criterion(race, target_race)

            # measure accuracy and record loss
            gender_acc_ = accuracy(gender, target_gender, topk=(1,))[0]
            race_acc_ = accuracy(race, target_race, topk=(1,))[0]

            loss0.update(loss0_.item(), data.size(0))
            loss1.update(loss1_.item(), data.size(0))

            gender_acc.update(gender_acc_.item(), data.size(0))
            race_acc.update(race_acc_.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Gender Loss {loss0.val:.3f} (avg={loss0.avg:.3f})\t'
                      'Race Loss {loss1.val:.3f} (avg={loss1.avg:.3f})\t'
                      'Gender Loss {loss0.val:.3f} ({loss0.avg:.3f})\t'
                      'Race Loss {loss1.val:.3f} ({loss1.avg:.3f})\t'
                      'Gender Acc {gender_acc.val:.3f} ({gender_acc.avg:.3f})\t'
                      'Race Acc {race_acc.val:.3f} ({race_acc.avg:.3f})'.format(
                       i, len(test_loader), loss0=loss0, loss1=loss1, gender_acc=gender_acc,
                       race_acc=race_acc))

        logger.info(' * Gender Acc {gender_acc.avg:.3f} Race Acc {race_acc.avg:.3f}'
              .format(gender_acc=gender_acc, race_acc=race_acc))

    return loss0.avg, loss1.avg, gender_acc.avg, race_acc.avg

if __name__ == '__main__':
    main()
