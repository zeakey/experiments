import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from scipy.io import savemat
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
import vltools
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch.datasets import ilsvrc2012
import vltools.pytorch as vlpytorch
import utils, models
from models import *

from models.scalenet_cifar import MSConv

from tensorboardX import SummaryWriter

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
parser.add_argument('--gpu', default=None, type=int, metavar='N', help='GPU ID')

parser.add_argument('--l1lambda', default=1e-5, type=float, help='lambda for L-1 penalty')
parser.add_argument('--allocate-freq', default=20, type=int, help='allocate frequency')
parser.add_argument('--debug', action="store_true")

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

THIS_DIR = abspath(dirname(__file__))
os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)

tfboard_writer = writer = SummaryWriter(log_dir=CONFIGS["MISC"]["TMP"])

logger = Logger(join(CONFIGS["MISC"]["TMP"], "log.txt"))

# model and optimizer
model = CONFIGS["MODEL"]["MODEL"] + "(num_classes=%d, logger=logger)" % (CONFIGS["DATA"]["NUM_CLASSES"])
logger.info("Model: %s" % model)
model = eval(model)

if CONFIGS["CUDA"]["DATA_PARALLEL"]:
    logger.info("Model Data Parallel")
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda(device=CONFIGS["CUDA"]["GPU_ID"])

for name, p in model.named_parameters():
    print(name, p.shape)

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

logger.info("Initial parameters details:")
for name, p in model.named_parameters():
    logger.info("%s, shape=%s, std=%f, mean=%f" % (name, str(p.shape), p.std().item(), p.mean().item()))

scheduler = lr_scheduler.MultiStepLR(optimizer,
                      milestones=CONFIGS["OPTIMIZER"]["MILESTONES"],
                      gamma=CONFIGS["OPTIMIZER"]["GAMMA"])

# loss function
criterion = torch.nn.CrossEntropyLoss()

# initiate l1 loss weight
delta_l1weight = 5e-6
l1weight = {}
uratio = {}
delta_ur = 0.01
def uratio_func(x):
    return (x >= x.max()*0.01).float().mean()
for name, m in model.named_modules():
    if isinstance(m, MSConv):
        l1weight[name] = 0
        uratio[name] = uratio_func(m.bn.weight.data)

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
        train_loader, val_loader = vlpytorch.datasets.cifar10(CONFIGS["DATA"]["DIR"], bs=CONFIGS["DATA"]["BS"])

    elif CONFIGS["DATA"]["DATASET"] == "cifar100":
        train_loader, val_loader = vlpytorch.datasets.cifar100(CONFIGS["DATA"]["DIR"], bs=CONFIGS["DATA"]["BS"])

    else:
        raise ValueError("Unknown dataset: %s. (Supported datasets: ilsvrc2012 | cifar10 | cifar100)" % CONFIGS["DATA"]["DATASET"])

    logger.info("Data loading done, %.3f sec elapsed." % (time.time() - start_time))

    # records
    best_acc1 = 0
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

        if args.debug:
            temperature = model.allocate()
            for k, v in temperature.items():
                factors0, factors1, triggered = v
                uratio0 = (factors0 >= (factors0.max() * 0.01)).float().mean()
                uratio1 = (factors1 >= (factors1.max() * 0.01)).float().mean()
                tfboard_writer.add_scalar('temperature/'+k+"0", uratio0.item(), epoch-1)
                tfboard_writer.add_scalar('temperature/'+k+"1", uratio1.item(), epoch-1)
                tfboard_writer.add_scalar('channels/'+k+"0", factors0.numel(), epoch-1)
                tfboard_writer.add_scalar('channels/'+k+"0", factors0.numel(), epoch-1)

                if triggered:
                    assert k+".conv2" in l1weight.keys()
                    l1weight[k+".conv2"] = 0

        # record init l1weight
        if epoch == 0:
            for k, v in l1weight.items():
                tfboard_writer.add_scalar('l1weight/'+k, v, -1)
            for k, v in uratio.items():
                tfboard_writer.add_scalar('uratio/'+k, v, -1)

        # train and evaluate
        loss, L1 = train(train_loader, epoch)
        acc1, acc5 = validate(val_loader, epoch)
        scheduler.step()

        # adjust l1 loss weight:
        for name, m in model.named_modules():
            if isinstance(m, MSConv):
                ur = uratio_func(m.bn.weight.data)
                # uratio[name]: utilization ratio of last epoch
                # ur: current utilization ratio
                if uratio[name] - ur < delta_ur:
                    l1weight[name] += delta_l1weight
                uratio[name] = ur
                tfboard_writer.add_scalar('l1weight/'+name, l1weight[name], epoch)
                tfboard_writer.add_scalar('uratio/'+name, uratio[name], epoch)

        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('train/BN-L1', L1, epoch)

        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)

        # allocate channels
        temperature = model.allocate(threshold=0.7)
        for k, v in temperature.items():
            factors0, factors1, triggered = v
            uratio0 = (factors0 >= (factors0.max() * 0.01)).float().mean()
            uratio1 = (factors1 >= (factors1.max() * 0.01)).float().mean()
            tfboard_writer.add_scalar('temperature/'+k+"0", uratio0.item(), epoch)
            tfboard_writer.add_scalar('temperature/'+k+"1", uratio1.item(), epoch)
            tfboard_writer.add_scalar('channels/'+k+"-0", factors0.numel(), epoch)
            tfboard_writer.add_scalar('channels/'+k+"-1", factors1.numel(), epoch)

            if triggered:
                # if channel transfer is triggered,
                # reset l1 loss weight to 0
                assert k+".conv2" in l1weight.keys()
                l1weight[k+".conv2"] = 0

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1

        if is_best:
            best_acc1 = acc1

        # log parameters details
        if args.debug:
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
            }, is_best, path=CONFIGS["MISC"]["TMP"])

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
    l1penalty = AverageMeter()

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
        # L1 loss
        L1 = torch.zeros_like(loss)
        for name, m in model.named_modules():
            if isinstance(m, MSConv):
                l1w = l1weight[name]
                L1 += l1w * torch.abs(m.bn.weight).sum()
        l1penalty.update(L1.item())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        (loss + L1).backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tfboard_writer.add_scalar('train/L1_iter', L1.item(), epoch * len(train_loader)  + i)
        tfboard_writer.add_scalar('train/loss_iter', loss.item(), epoch * len(train_loader)  + i)
        tfboard_writer.add_scalar('train/lr', lr, epoch * len(train_loader)  + i)
        
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

        if i == 50 and args.debug:
            break

    return losses.avg, l1penalty.avg

def validate(val_loader, epoch):
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

            if i == 10 and args.debug:
                break

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
