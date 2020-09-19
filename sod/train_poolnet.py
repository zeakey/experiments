# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath, split, splitext
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from vlkit import get_logger, run_path
from vlkit import image as vlimage
from vlkit.pytorch import save_checkpoint, AverageMeter
from vlkit.lr import CosAnnealingLR, MultiStepLR
# distributed
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
from models.drn_seg import DRNSeg
from models.poolnet import poolnet
from dataset import SODDataset
from utils import accuracy, init_from_pretrained, save_maps
from crf import par_batch_crf_dataloader
from floss import FLoss, F_cont

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--arch', '--a', metavar='STR', default="drn_d_54", help='model')
# data
parser.add_argument('--train-data', metavar='DIR', default="data/DUTS/DUTS-TR", help='path to dataset')
parser.add_argument('--test-data', metavar='DIR', default="data/ECSSD", help='path to dataset')
parser.add_argument('--use-rec', action='store_true', help='Use mxnet record.')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default="360,360", type=str, help='image size')
parser.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers')
# optimization
parser.add_argument('--epochs', default=24, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=2, help="warmup epochs")
parser.add_argument('--milestones', default="15,1000000", type=str)
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-scheduler', default="cos", type=str, help='LR scheduler')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/poolnet")
parser.add_argument('--pretrained', help='pretrained model', type=str, default=None)
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

parser.add_argument("--setting", default=0, type=int)

args = parser.parse_args()


if args.local_rank == 0:
    args.tmp = run_path(args.tmp)
    os.makedirs(args.tmp, exist_ok=True)

args.milestones = [int(i) for i in args.milestones.split(',')]
args.imsize = [int(i) for i in args.imsize.split(',')]

# torch.backends.cudnn.benchmark = True

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


CELoss = nn.CrossEntropyLoss()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
# image_transform = transforms.Compose([transforms.Resize(args.imsize), transforms.ToTensor(), normalize])
image_transform = None # transforms.Compose([transforms.ToTensor()])
label_transform = None # transforms.Compose([transforms.ToTensor()])
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
    batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    pin_memory=True, drop_last=False, sampler=test_sampler)

train_loader_len = len(train_loader)

def main():
    if args.local_rank == 0:
        logger.info(args)

    model = poolnet.build_model("resnet")
    model.base.resnet.load_state_dict(torch.load("resnet50_caffe.pth"))

    if args.pretrained is not None and isfile(args.pretrained):
        if args.local_rank == 0:
            model = init_from_pretrained(model, args.pretrained, verbose=True)
        else:
            model = init_from_pretrained(model, args.pretrained, verbose=False)

    if args.sync_bn:
        if args.local_rank == 0:
            logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()

    # poolnet optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    # else:
    #     model = torch.nn.DataParallel(model)

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

    lr_scheduler = None

    for epoch in range(args.epochs):
        train_loss, ori_losses = train_epoch(model, train_loader, optimizer, lr_scheduler, epoch)
        test_mae, ori_acc = test(model, test_loader, epoch)

        # adjust lr
        if epoch in args.milestones:
            args.lr = args.lr * 0.1
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
            logger.info("decreasing learning rate to %f."%args.lr)

        if args.local_rank == 0:
            tfboard_writer.add_scalar("train/seg-loss", train_loss, epoch)
            tfboard_writer.add_scalar("train/ori-loss", ori_losses, epoch)
            tfboard_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            tfboard_writer.add_scalar("test/mae", test_mae, epoch)
            tfboard_writer.add_scalar("test/Orientation-Accuracy", ori_acc, epoch)

    if args.local_rank == 0:
        logger.info("Optimization done, ALL results saved to %s." % args.tmp)
        tfboard_writer.close()
        for h in logger.handlers:
            h.close()


def train_epoch(model, train_loader, optimizer, lr_scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    seg_losses = AverageMeter()
    ori_losses = AverageMeter()

    train_loader_len = len(train_loader)
    # switch to train mode
    model.eval()

    end = time.time()
    for i, data in enumerate(train_loader):

        image = data["image"]
        target = data["label"]
        edge = data["edge"]
        orientation = data["orientation"]
        mask = data["mask"]

        image = torch.flip(image, dims=(1,)) # RGB to BGR
        image -= torch.tensor([104.00699, 116.66877, 122.67892]).reshape(1,3,1,1)

        if args.distributed:
            image = image.cuda(non_blocking=True).to(dtype=DTYPE)

        image = image.cuda()#(non_blocking=True)
        target = target.cuda().float()#(non_blocking=True).float()
        # edge = edge.cuda(non_blocking=True).float()
        # orientation = orientation.cuda(non_blocking=True).long()
        # mask = mask.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)
        output = model(image)

        # seg_loss = F.binary_cross_entropy(torch.sigmoid(output["seg"]), target)
        seg_loss = F.binary_cross_entropy_with_logits(output, target, reduction='sum')

        # ori_loss = F.cross_entropy(output["orientation"], orientation, reduction="none")
        # ori_loss = (ori_loss * mask).sum() / mask.sum()
        ori_loss = torch.zeros_like(seg_loss)

        # edge_loss = F.binary_cross_entropy(torch.sigmoid(output["edge"]), edge, reduction="none")
        # mask = torch.where(edge==0, edge.mean(), 1-edge.mean())
        # mask /= max(mask.mean(), 1e-5)
        # edge_loss = (edge_loss * mask).mean()
        ori_loss = torch.zeros_like(seg_loss)

        if args.distributed:
            reduced_seg_loss = reduce_tensor(seg_loss.data)
            reduced_ori_loss = reduce_tensor(ori_loss.data)
        else:
            reduced_seg_loss = seg_loss.data
            reduced_ori_loss = ori_loss.data

        seg_losses.update(reduced_seg_loss.item(), image.size(0))
        ori_losses.update(reduced_ori_loss.item(), image.size(0))

        # compute and adjust lr
        if lr_scheduler is not None:
            lr = lr_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # compute gradient and do SGD step
        loss = seg_loss# + ori_loss
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
                  'SegLoss {seg_loss.val:.3f} ({seg_loss.avg:.3f})\t'
                  'OriLoss {ori_loss.val:.3f} ({ori_loss.avg:.3f})\t'
                  'LR: {lr:.2E}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, seg_loss=seg_losses, ori_loss=ori_losses, lr=lr))

            tfboard_writer.add_scalar("train/iter-segloss", seg_losses.val, epoch*len(train_loader)+i)

        # if args.local_rank == 0 and i == 0:
        #     data = torchvision.utils.make_grid(image, normalize=True)
        #     pred = torchvision.utils.make_grid(torch.sigmoid(output), normalize=True)
        #     # tfboard_writer.add_image("images", data, epoch)
        #     # tfboard_writer.add_image("predictions", pred, epoch)

    return seg_losses.avg, ori_losses.avg

def test(model, test_loader, epoch):
    batch_time = AverageMeter()
    seg_losses = AverageMeter()
    mae =  AverageMeter()
    ori_acc1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    test_loader_len = len(test_loader)

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):
            metas = data["metas"]
            image = data["image"].cuda()
            target = data["label"].cuda().float()
            mask = data["mask"].cuda()
            orientation = data["orientation"].cuda().long()

            # compute output
            output = model(image)
            seg_pred = torch.sigmoid(output)
            seg_loss = torch.nn.functional.binary_cross_entropy(seg_pred, target)

            # ori_pred = output["orientation"].transpose(0,1)[:, mask].t()
            # ori_gt = orientation[mask]
            # ori_top1, ori_top2 = accuracy(ori_pred, ori_gt, topk=(1,2))
            ori_top1 = torch.zeros_like(seg_loss)

            # mae and F-measure
            mae_ = (seg_pred - target).abs().mean()

            filenames = [splitext(split(i)[-1])[0] for i in metas["filename"]]
            diff = ((seg_pred >= 0.5).float() - target).detach().cpu().numpy()
            diff = (diff * 127 + 128)

            if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
                seg_pred = (seg_pred.cpu().numpy()*255).astype(np.uint8)
                save_maps(seg_pred, filenames, join(args.tmp, "epoch-%d"%epoch, "prediction"))
                # seg_pred = (seg_pred >= 0.5).cpu().numpy()*255
                # gt = target.cpu().numpy()*255
                # diff = np.concatenate((seg_pred, gt, diff), axis=3).astype(np.uint8)
                # save_maps(diff, filenames, join(args.tmp, "epoch-%d"%epoch, "diff"))

            if args.distributed:
                reduced_seg_loss = reduce_tensor(seg_loss.data)
                reduced_mae = reduce_tensor(mae_)
            else:
                reduced_seg_loss = seg_loss.data
                reduced_mae = mae_.data

            seg_losses.update(reduced_seg_loss.item(), image.size(0))
            mae.update(reduced_mae.item(), image.size(0))
            ori_acc1.update(ori_top1.item(), image.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'SegLoss {seg_loss.val:.3f} (avg={seg_loss.avg:.3f})\t'
                      'MAE {mae.val:.3f} (avg={mae.avg:.3f})\t'
                      'OriAcc={ori_acc.avg:.3f}\t'.format(
                       i, test_loader_len, seg_loss=seg_losses, mae=mae, ori_acc=ori_acc1))

    if args.local_rank == 0:
        logger.info(' * MAE {mae.avg:.5f} OriAcc={ori_acc.avg:.3f}'\
            .format(mae=mae, ori_acc=ori_acc1))

    return mae.avg, ori_acc1.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
