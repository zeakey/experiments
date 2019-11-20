# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
import torch, torchvision
import torch.nn as nn
import torchvision
# logger and auxliaries
import numpy as np
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vltools import Logger
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
import vltools.pytorch as vlpytorch
from vltools.tcm.lr import CosAnnealingLR, MultiStepLR
from vltools import image as vlimage
# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

# distributed
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import FP16_Optimizer, network_to_half

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
# data
parser.add_argument('--use-rec', action="store_true")
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--num_classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
# optimizer
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=5, help="warmup epochs")
parser.add_argument('--milestones', default="30,60,80", type=str)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/resnet50")
# FP16
parser.add_argument('--fp16', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
# distributed
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--dali-cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

args = parser.parse_args()
args.milestones = [int(i) for i in args.milestones.split(',')]

torch.backends.cudnn.benchmark = True

# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if args.use_rec:
            # MXnet rec reader
            self.input = ops.MXNetReader(path=join(data_dir, "train.rec"), index_path=join(data_dir, "train.idx"),
                                        random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
        else:
            # image reader
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if args.use_rec:
            self.input = ops.MXNetReader(path=join(data_dir, "val.rec"), index_path=join(data_dir, "val.idx"),
                                     random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
        else:
            self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]  

# loss function
criterion = torch.nn.CrossEntropyLoss()

if args.local_rank == 0:
    tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
    logger = Logger(join(args.tmp, "log.txt"))

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.fp16:
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

def main():
    if args.local_rank == 0:
        logger.info(args)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    if args.use_rec:
        traindir = args.data
        valdir = args.data
    else:
        traindir = join(args.data, "train")
        valdir = join(args.data, "val")
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
           data_dir=traindir, crop=224, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
           data_dir=valdir, crop=224, size=256)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    train_loader_len = int(train_loader._size / args.batch_size)

    # model and optimizer
    model = torchvision.models.resnet.resnet50().cuda()

    if args.fp16:
        model = network_to_half(model)
    model = DDP(model, delay_allreduce=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.local_rank == 0:
        logger.info("Optimizer:")
        logger.info(optimizer)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                            static_loss_scale=args.static_loss_scale,
                            dynamic_loss_scale=args.dynamic_loss_scale, verbose=False)
    # records
    best_acc1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            shutil.copy(args.resume, args.tmp)
            if args.local_rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank == 0:
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(loader_len=train_loader_len,
                   milestones=args.milestones, gamma=args.gamma, warmup_epochs=args.warmup_epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train and evaluate
        loss = train(train_loader, model, optimizer, scheduler, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)

        # # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1

        if args.local_rank == 0:
            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()},
                is_best, path=args.tmp)
            logger.info("Best acc1=%.5f" % best_acc1)
            tfboard_writer.add_scalar('train/loss', loss, epoch)
            tfboard_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)
            tfboard_writer.add_scalar('test/acc1', acc1, epoch)
            tfboard_writer.add_scalar('test/acc5', acc5, epoch)

    if args.local_rank == 0:
        logger.info("Optimization done, ALL results saved to %s." % args.tmp)

def train(train_loader, model, optimizer, lrscheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_loader_len = int(np.ceil(train_loader._size/args.batch_size))
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):

        target = data[0]["label"].squeeze().cuda().long()
        data = data[0]["data"]

        # measure data loading time
        data_time.update(time.time() - end)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        losses.update(reduced_loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute and adjust lr
        if lrscheduler is not None:
            lr = lrscheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if args.local_rank == 0 and i % args.print_freq == 0:
            tfboard_writer.add_scalar("train/iter-lr", lr, epoch*train_loader_len+i)
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.2E}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   lr=lr))

        if i%1000 == 0:
            norm = torch.norm(output, p=2, dim=1)
            _, norm_sort = torch.sort(norm)
            selected = torch.cat((norm_sort[0:8], norm_sort[-8::]))
            selected = data[selected,].cpu().detach()
            save_path = join(args.tmp, "images", "epoch%d-iter%d.jpg"%(epoch, i))
            os.makedirs(dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(selected, filename=save_path, normalize=True)

    train_loader.reset()

    return losses.avg

def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    val_loader_len = int(np.ceil(val_loader._size/args.batch_size))

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):

            target = data[0]["label"].squeeze().cuda().long()
            data = data[0]["data"]

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, val_loader_len, loss=losses, top1=top1, top5=top5))

    if args.local_rank == 0:
        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
                    .format(top1=top1, top5=top5))

    val_loader.reset()

    return top1.avg, top5.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
