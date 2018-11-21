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
import matplotlib.pyplot as plt
import os, sys, argparse, time, shutil
from os.path import join, split, abspath, isdir, isfile, dirname
from utils import save_checkpoint, AverageMeter, accuracy
import pytools, cv2
from utils import save_checkpoint, Logger, ilsvrc2012, cifar10, cifar100
import models
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', metavar='STR', default="resnet18",
                    help='model name (resnet18|squeezenet), default="resnet18"')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch-size', default=512, type=int,
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
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='places2subset', type=str, metavar='PATH',
                    help='checkpoint path (default: checkpoint)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--tmp', help='tmp folder', default='tmp')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='benchmark time')
args = parser.parse_args()
args.model = args.model.lower()

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
args.checkpoint = join(TMP_DIR, args.checkpoint)
if not isdir(args.checkpoint):
  os.makedirs(args.checkpoint)

def main():
    print(args)
    log = Logger(join(args.checkpoint, 'log.txt'))
    sys.stdout = log
    print("Loading training data...")
    start = time.time()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='~/.torch/data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='~/.torch/data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    
    print("Data loading done, %.3f sec elapsed." % (time.time() - start))

    # model
    from models import resnet_affine
    model = resnet_affine.resnet18()
    model.cuda()
    print(model)
    log.flush()

    criterion = nn.CrossEntropyLoss().cuda()
    
    if True:
        others = []
        rnn = []
        for name, p in model.named_parameters():
            if 'rnn' in name:
                rnn.append(p)
                # print(name, 'fc')
            else:
                others.append(p)
        optimizer = torch.optim.SGD([{"params": others, "lr": args.lr},
                                     {"params": rnn, "lr": 0}], args.lr,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    best_prec1 = 0
    prec1s = []
    prec5s = []
    loss_record = []
    if args.evaluate:
        # Others 0    Sports 1     Music 2 
        model.cuda()
        
        prec1, prec5 = validate(val_loader, model, criterion)
        
        print("Prec@1%.3f, Prec@5%.3f" % (prec1, prec5))
        return
        
    for epoch in range(args.start_epoch, args.epochs):
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, False, filename=join(args.checkpoint, 'checkpoint.pth'))

        if args.benchmark:
            print("Start to benchmark inference time...")
            bench_time = AverageMeter()
            model.cpu()
            model.eval()
            with torch.no_grad():
                bench_start = time.time()
                for i, (data, _) in enumerate(val_loader):
                    # target = target.cuda(non_blocking=True)
                    # input = input.cuda()
                    # compute output
                    data = data.cpu()
                    _ = model(data)
                    bench_end = time.time()
                    bench_time.update(bench_end - bench_start)
                    print("---> Running batch%d (batchsize%d), time elapsed %.3f, %.3fms per sample" % \
                         (i+1, args.bs, bench_end - bench_start, 1000*(bench_end - bench_start)/args.bs/(i+1)))
                    if i >= 3:
                        break
            model.cuda()
            print("Inference %d samples, time elapsed %f" % (args.bs * (i+1), bench_time.avg))
            input("Press Enter to Continue...")
        
        if epoch == 0:
            prec1, prec5 = validate(val_loader, model, criterion)
        
        # train for one epoch
        loss_record += train(train_loader, model, criterion, optimizer, epoch)
        
        scheduler.step()

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)
        prec1s.append(prec1)
        prec5s.append(prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, filename=join(args.checkpoint, 'checkpoint.pth'))
        if is_best:
            shutil.copyfile(join(args.checkpoint, 'checkpoint.pth'), join(args.checkpoint, 'model_best.pth'))
        print("Evaluating done, best prec1@%.3f" % max(prec1s))
        log.flush()
    loss_record = np.array(loss_record)
    _, axes = plt.subplots(1, 2)
    axes[0].plot(prec1s, color='r', linewidth=2)
    axes[0].plot(prec5s, color='g', linewidth=2)
    axes[0].legend(['Top1 Accuracy (Best%.3f)' % max(prec1s), 'Top5 Accuracy (Best%.3f)' % max(prec5s)])
    axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0].plot(np.argmax(prec1s), max(prec1s), "*r", linewidth=8) # dot best acc

    axes[1].plot(loss_record)
    axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1].legend(["Loss"])

    plt.savefig(join(args.checkpoint, 'record.pdf'))

    record = dict({'prec1': np.array(prec1s), 'prec5': np.array(prec5s), 'loss_record': np.array(loss_record)})
    savemat(join(args.checkpoint, 'precision.mat'), record)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    loss_record = []
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record los
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), input.size(0))
        top1_meter.update(prec1.item(), input.size(0))
        top5_meter.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} (avg={batch_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=loss_meter, top1=top1_meter, top5=top5_meter))
            loss_record.append(loss_meter.avg)
    return loss_record

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda()
            # compute output
            output = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} (avg={batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
