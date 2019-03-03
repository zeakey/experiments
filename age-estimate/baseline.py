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
from dataloader import CACDDataset
import vltools
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
import vltools.pytorch as vlpytorch
from vltools.tcm import CosAnnealingLR
import resnet

def MAE(prediction, target):
    assert prediction.squeeze().ndimension() == 2, prediction.squeeze().shape
    assert target.squeeze().ndimension() == 1
    return torch.abs(target - torch.argmax(prediction, dim=1)).float().mean().item()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# arguments from command line
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
# by default, arguments bellow will come from a config file
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--imsize', type=int, default=224, help='Image Size')
parser.add_argument('--num_classes', default=62-14+1, type=int, metavar='N', help='Number of classes')
parser.add_argument('--bs', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='LR', help='weight decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/baseline")
parser.add_argument('--gpu', type=int, default=0)

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

# Switch between VGG16 and ResNet50
if False:
    model = torchvision.models.vgg.vgg16(pretrained=False)
    for name, p in model.named_parameters():
        print(name, ": ", p.data.mean(), p.data.std())
    print("========================")
    model = torchvision.models.vgg.vgg16(pretrained=True)
    for name, p in model.named_parameters():
        print(name, ": ", p.data.mean(), p.data.std())
    print("========================")
    classifier = list(model.classifier.children())
    classifier[-1] = nn.Linear(4096, args.num_classes)
    model.classifier = nn.Sequential(*classifier)
    for name, p in model.named_parameters():
        print(name, ": ", p.data.mean(), p.data.std())
    print("========================")
else:
    model = resnet.resnet18(pretrained=True)
    model.fc =  nn.Linear(512, args.num_classes)

model = nn.DataParallel(model).cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd
)
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# loss function
criterion = torch.nn.CrossEntropyLoss()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

train_dataset = CACDDataset("cacd-guoyilu", "train.txt",
            transforms.Compose([
                transforms.Resize((args.imsize, args.imsize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
test_dataset = CACDDataset("cacd-guoyilu", "test.txt",
            transforms.Compose([
                transforms.Resize((args.imsize, args.imsize)),
                transforms.ToTensor(),
                normalize,
            ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                           num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                           num_workers=8, pin_memory=True)
train_loss_all = []
test_loss_all = []
train_mae_all = []
test_mae_all = []
lr_all = []

scheduler = CosAnnealingLR(len(train_loader)*args.epochs, lr_max=args.lr,
                           warmup_iters=len(train_loader)*2)

def main():

    logger.info(args)

    # records
    best_acc1 = 0

    test_acc1_record = []
    test_acc5_record = []

    train_mae_record = []
    test_mae_record = []

    train_loss_record = []
    test_loss_record = []

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

    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        train_loss, train_acc1, train_acc5, train_mae = train(train_loader, epoch)
        test_loss, test_acc1, test_acc5, test_mae = validate(test_loader)

        # record stats
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        test_acc1_record.append(test_acc1)
        test_acc5_record.append(test_acc5)

        train_mae_record.append(train_mae)
        test_mae_record.append(test_mae)

        lr_record.append(optimizer.param_groups[0]["lr"])

        # remember best prec@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1_record)
        best_acc5 = max(test_acc5_record)
        logger.info("Best acc1=%.3f, best train-mae=%.3f, best test-mae=%.3f" % \
                    (best_acc1, min(train_mae_record), min(test_mae_record)))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer' : optimizer.state_dict(),
            }, is_best, path=args.tmp)

        # continously save records in case of interupt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axid = 0
        axes[axid].plot(test_acc1_record, color='r', linewidth=2)
        axes[axid].plot(test_acc5_record, color='g', linewidth=2)
        axes[axid].legend(['Top1 Accuracy (Best%.3f)' % max(test_acc1_record),
                           'Top5 Accuracy (Best%.3f)' % max(test_acc5_record)],
                           loc="lower right")

        axes[axid].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[axid].set_xlabel("Epoch")
        axes[axid].set_ylabel("Accuracy")

        axid += 1
        axes[axid].plot(train_mae_record, color='r', linewidth=2)
        axes[axid].plot(test_mae_record, color='g', linewidth=2)
        axes[axid].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[axid].legend(["Train-MAE (Best=%.3f)" % min(train_mae_record),
                           "Test-MAE (Best=%.3f)" % min(test_mae_record)], loc="upper right")
        axes[axid].set_xlabel("Epoch")
        axes[axid].set_ylabel("MAE")

        axid += 1
        axes[axid].plot(train_loss_record, color='r', linewidth=2)
        axes[axid].plot(test_loss_record, color='g', linewidth=2)
        axes[axid].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[axid].legend(["Train Loss", "Test Loss"], loc="upper right")
        axes[axid].set_xlabel("Epoch")
        axes[axid].set_ylabel("Loss")

        axid += 1
        axes[axid].plot(lr_record, color="r", linewidth=2)
        axes[axid].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[axid].legend(["Learning Rate"], loc="upper right")
        axes[axid].set_xlabel("Epoch")
        axes[axid].set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.savefig(join(args.tmp, 'epoch-record.pdf'))
        plt.savefig(join(args.tmp, 'epoch-record.svg'))
        plt.close(fig)

        record = dict({'acc1': np.array(test_acc1_record),
                       'acc5': np.array(test_acc5_record),
                       'train_loss_record': np.array(train_loss_record),
                       'test_loss_record': np.array(test_loss_record),
                       'lr_record': np.array(lr_record),
                       'train_loss_all': np.array(train_loss_all),
                       'test_loss_all': np.array(test_loss_all),
                       'train_mae_all': np.array(train_mae_all),
                       'test_mae_all': np.array(test_mae_all)})

        savemat(join(args.tmp, 'record.mat'), record)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].plot(train_loss_all, color='r', linewidth=2)
    axes[0, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 0].set_xlabel("Iter")
    axes[0, 0].set_ylabel("Train Loss")

    axes[0, 1].plot(test_loss_all, color='r', linewidth=2)
    axes[0, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 1].set_xlabel("Iter")
    axes[0, 1].set_ylabel("Test Loss")

    axes[1, 0].plot(train_mae_all, color='r', linewidth=2)
    axes[1, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 0].set_xlabel("Iter")
    axes[1, 0].set_ylabel("Train MAE")

    axes[1, 1].plot(test_mae_all, color='r', linewidth=2)
    axes[1, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 1].set_xlabel("Iter")
    axes[1, 1].set_ylabel("Test MAE")

    plt.tight_layout()
    plt.savefig(join(args.tmp, 'iter-record.pdf'))
    plt.savefig(join(args.tmp, 'iter-record.svg'))
    plt.close(fig)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

def train(train_loader, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mae = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(device=args.gpu, non_blocking=True)
        data = data.cuda(device=args.gpu)

        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))
        mae.update(MAE(output, target), data.size(0))

        train_loss_all.append(losses.val)
        train_mae_all.append(mae.val)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        # adjust lr and update params
        lr = scheduler.step()
        lr_all.append(lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
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
                  'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                  'LR: {lr:}'.format(
                   epoch, args.epochs, i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   mae=mae, lr=lr))

    return losses.avg, top1.avg, top5.avg, mae.avg

def validate(test_loader):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mae = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(test_loader):
            
            target = target.cuda(device=args.gpu, non_blocking=True)
            data = data.cuda(device=args.gpu)
            # compute output
            output = model(data)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            mae.update(MAE(output, target), data.size(0))

            test_loss_all.append(losses.val)
            test_mae_all.append(mae.val)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})\t'
                      'MAE {mae.val:.3f} (avg={mae.avg:.3f})'.format(
                       i, len(test_loader), loss=losses, top1=top1, top5=top5, mae=mae))

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} MAE {mae.avg:.3f}'
              .format(top1=top1, top5=top5, mae=mae))
    return losses.avg, top1.avg, top5.avg, mae.avg

if __name__ == '__main__':
    main()
