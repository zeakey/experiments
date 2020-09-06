# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from vlkit.training import get_logger
from vlkit import image as vlimage
from vlkit.pytorch import save_checkpoint, AverageMeter, accuracy
from vlkit.pytorch.datasets import ilsvrc2012
import vlkit.pytorch as vlpytorch
from vlkit.lr import CosAnnealingLR, MultiStepLR
# distributed
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
from drn_seg import DRNSeg
from dataset import SODDataset
from utils import save_maps, get_full_image_names, load_batch_images, batch_mva, evaluate_maps, init_from_pretrained
from crf import par_batch_crf_dataloader
from floss import FLoss, F_cont


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--arch', '--a', metavar='STR', default="drn_d_105", help='model')
# data
parser.add_argument('--train-data', metavar='DIR', default="data/ecssd", help='path to dataset')
parser.add_argument('--test-data', metavar='DIR', default="data/ecssd", help='path to dataset')
parser.add_argument('--use-rec', action='store_true', help='Use mxnet record.')
parser.add_argument('--batch-size', default=20, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default=432, type=int, metavar='N', help='im crop size')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
# optimization
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=1, help="warmup epochs")
parser.add_argument('--milestones', default="10,15", type=str)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp")
parser.add_argument('--pretrained', help='pretrained model', type=str, default="Pretrained_Models/drn_pretraining/drn-d-105_ms_cityscapes.pth")
# Loss function
parser.add_argument('--loss', help='loss function', type=str, default="ce")
parser.add_argument('--floss-beta', help='floss beta', type=float, default=1)
# FP16
parser.add_argument('--fp16', action='store_true',
                    help='half precision training.')
parser.add_argument('--static-loss-scale', type=float, default=128,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync-bn', action='store_true',
                    help='Use sync BN.')
args = parser.parse_args()


if args.local_rank == 0:
    os.makedirs(args.tmp, exist_ok=True)

args.milestones = [int(i) for i in args.milestones.split(',')]

torch.backends.cudnn.benchmark = True

if args.local_rank == 0:
    tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
    logger = get_logger(join(args.tmp, "log.txt"))

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    args.world_size = torch.distributed.get_world_size()

DTYPE = torch.float
if args.fp16:
    from apex import amp
    DTYPE = torch.half
    assert args.distributed, "FP16 can be only used in Distributed mode"
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."



floss = FLoss(beta=args.floss_beta)
criterion = nn.CrossEntropyLoss()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
image_transform = transforms.Compose([transforms.Resize([args.imsize]*2), transforms.ToTensor(), normalize])
label_transform = transforms.Compose([transforms.Resize([args.imsize]*2, interpolation=Image.NEAREST)])
train_dataset = SODDataset(img_dir=join(args.train_data, "images"),
                label_dir=join(args.train_data, "gt-masks"),
                name_list=join(args.train_data, "train_names.txt"),
                flip=True, image_transform=image_transform, label_transform=label_transform)
test_dataset = SODDataset(img_dir=join(args.test_data, "images/"),
                label_dir=join(args.test_data, "gt-masks/"),
                name_list=join(args.test_data, "test_names.txt"),
                flip=False, image_transform=image_transform, label_transform=label_transform)

if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
else:
    train_sampler = None
    val_sampler = None
    test_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers,
    pin_memory=True, sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=100, shuffle=False, num_workers=args.workers,
    pin_memory=True, drop_last=False, sampler=test_sampler)

train_loader_len = len(train_loader)

def main():
    if args.local_rank == 0:
        logger.info(args)
    # model and optimizer
    model = DRNSeg(args.arch, classes=1)

    if args.pretrained is not None:
        if args.local_rank == 0:
            model = init_from_pretrained(model, args.pretrained, verbose=True)
        else:
            model = init_from_pretrained(model, args.pretrained, verbose=False)

    if args.sync_bn:
        if args.local_rank == 0:
            logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model)

    if args.local_rank == 0:
        logger.info(optimizer)

    # records
    best_mae = 10000

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            shutil.copy(args.resume, args.tmp)
            if args.local_rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank == 0:
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    lrscheduler = CosAnnealingLR(loader_len=len(train_loader), epochs=args.epochs, max_lr=args.lr)
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, lrscheduler, epoch)
        test_mae, test_f = test(model, test_loader, epoch)
        

        if args.local_rank == 0:
            tfboard_writer.add_scalar("train/F-measure", train_loss, epoch)
            tfboard_writer.add_scalar("test/mae", test_mae, epoch)
            tfboard_writer.add_scalar("test/F-measure", test_f, epoch)


    if args.local_rank == 0:
        logger.info("Optimization done, ALL results saved to %s." % args.tmp)
        tfboard_writer.close()
        for h in logger.handlers:
            h.close()


def train_epoch(model, train_loader, optimizer, lrscheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    train_loader_len = len(train_loader)
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):

        image = data["image"]
        target = data["label"].long()
        if args.distributed:
            image = image.cuda(non_blocking=True).to(dtype=DTYPE)
        target = target.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)
        output = model(image)

        target = target.to(dtype=output.dtype)
        # loss = criterion(torch.sigmoid(output), target)
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), target)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), image.size(0))

        # compute and adjust lr
        if lrscheduler is not None:
            lr = lrscheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # scale loss before backward
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if args.distributed:
            torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if args.local_rank == 0 and i % args.print_freq == 0:
            logger.info('Epoch[{0}/{1}] It[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'LR: {lr:.2E}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, loss=losses, lr=lr))

        # if args.local_rank == 0 and i == 0:
        #     data = torchvision.utils.make_grid(image, normalize=True)
        #     pred = torchvision.utils.make_grid(torch.sigmoid(output), normalize=True)
        #     # tfboard_writer.add_image("images", data, epoch)
        #     # tfboard_writer.add_image("predictions", pred, epoch)

    return losses.avg

def test(model, test_loader, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae =  AverageMeter()
    fmeasure = AverageMeter()
    # switch to evaluate mode
    model.eval()
    test_loader_len = len(test_loader)

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):
            image = data["image"].cuda()
            target = data["label"].cuda()

            # compute output
            output = model(image)
            output = torch.sigmoid(output)
            target = target.to(dtype=output.dtype)
            loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), target)

            # mae and F-measure
            mae_ = (output - target).abs().mean()
            f_ = floss(output, target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_mae = reduce_tensor(mae_)
                reduced_f = reduce_tensor(f_)
            else:
                reduced_loss = loss.data
                reduced_mae = mae_.data
                reduced_f = f_.data
            

            losses.update(reduced_loss.item(), image.size(0))
            mae.update(reduced_mae.item(), image.size(0))
            fmeasure.update(reduced_f.item(), image.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'MAE {mae.val:.3f} (avg={mae.avg:.3f}) F {fmeasure.val:3f} (avg={fmeasure.avg:.3f})'.format(
                       i, test_loader_len, loss=losses, mae=mae, fmeasure=fmeasure))

    if args.local_rank == 0:
        logger.info(' * MAE {mae.avg:.5f} F-measure {fmeasure.avg:.5f}'.format(mae=mae, fmeasure=fmeasure))

    return mae.avg, fmeasure.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
