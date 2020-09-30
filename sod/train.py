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
from vlkit import get_logger
from vlkit import image as vlimage
from vlkit.pytorch import save_checkpoint, AverageMeter
from vlkit.lr import CosAnnealingLR, MultiStepLR
import apex
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
from models.drn_seg import DRNSeg
from models.poolnet import poolnet
from dataset import SODDataset
from utils import accuracy, init_from_pretrained, save_maps

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test-freq', default=2, type=int, help='testing frequency')
parser.add_argument('--arch', '--a', metavar='STR', default="drn_d_54", help='model')
# data
parser.add_argument('--train-data', metavar='DIR', default="data/DUTS/DUTS-TR", help='path to dataset')
parser.add_argument('--test-data', metavar='DIR', default="data/ECSSD", help='path to dataset')
parser.add_argument('--backend', default="cv2", help='Image decode backend (cv2 | pil)')
parser.add_argument('--use-rec', action='store_true', help='Use mxnet record.')
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default="320, 320", type=str, help='image size')
parser.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers')
# optimization
parser.add_argument('--epochs', default=24, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=0, help="warmup epochs")
parser.add_argument('--milestones', default="12,18", type=str)
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-scheduler', default="cos", type=str, help='LR scheduler')
parser.add_argument('--gamma', default=0.1, type=float, help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float, help='weight decay')
parser.add_argument('--freeze-bn', action='store_true', help='Freeze BN parameters')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp")
parser.add_argument('--pretrained', help='pretrained model', type=str, default="drn_d_54-0e0534ff.pth")
# Loss function
parser.add_argument('--loss', help='loss function', type=str, default="ce")
parser.add_argument('--reduction', help='loss reduction', type=str, default="mean")
# FP16
parser.add_argument('--fp16', action='store_true', help='half precision training.')
parser.add_argument('--static-loss-scale', type=float, default=128,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
args = parser.parse_args()

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(args.tmp, exist_ok=True)

args.milestones = [int(i) for i in args.milestones.split(',')]
args.imsize = [int(i) for i in args.imsize.split(',')]

torch.backends.cudnn.benchmark = True

tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
logger = get_logger(join(args.tmp, "log.txt"))

test_datasets = {
    "DUTS": {"image": "data/DUTS/DUTS-TE/images", "mask": "data/DUTS/DUTS-TE/gt-masks"},
    "DUT-OMRON": {"image": "data/DUT-OMRON/image", "mask": "data/DUT-OMRON/mask"},
    "HKU-IS": {"image": "data/HKU-IS/image", "mask": "data/HKU-IS/mask"},
    "PASCAL-S": {"image": "data/PASCAL-S/image", "mask": "data/PASCAL-S/mask"},
    }

def image_transform(image):
    assert isinstance(image, np.ndarray) and image.ndim == 3, "%s, %s"%(type(image), image.shape)
    image = image - np.array((104.00699, 116.66877, 122.67892))
    image = image / 255.0
    return image.astype(np.float32).transpose((2,0,1))
def label_transform(label):
    assert isinstance(label, np.ndarray) and label.ndim == 2
    label = label[np.newaxis, :, :]
    return label
train_dataset = SODDataset(img_dir=join(args.train_data, "images"),
                label_dir=join(args.train_data, "gt-masks"), imsize=args.imsize,
                flip=True, backend=args.backend, image_transform=image_transform, label_transform=label_transform)

test_datasets = {
    k: SODDataset(img_dir=v["image"], label_dir=v["mask"], imsize=args.imsize, flip=False, backend=args.backend,
               image_transform=image_transform, label_transform=label_transform) \
    for k, v in test_datasets.items()
}

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
    pin_memory=True, sampler=None)
train_loader_len = len(train_loader)

test_loaders = {
    k: torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    pin_memory=True, drop_last=False, sampler=None) \
    for k, dataset in test_datasets.items()
}

def main():
    logger.info(args)
    # model and optimizer
    model = DRNSeg("drn_d_54", classes=1)
    if args.freeze_bn:
        for name, p in model.named_parameters():
            if "base" in name and "bn" in name:
                p.requires_grad = False
                logger.info("Set requires_grad=False: %s"%name)

    if isfile(args.pretrained):
        model = init_from_pretrained(model, args.pretrained, verbose=True)

    model = model.cuda()

    if True:
        # poolnet optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = None
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=args.momentum)
        lr_scheduler = CosAnnealingLR(epoch_size=len(train_loader), epochs=args.epochs, max_lr=args.lr, min_lr=0, warmup_epochs=args.warmup_epochs)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    logger.info(optimizer)

    for epoch in range(args.epochs):
        # train and record logs
        train_loss, ori_losses = train_epoch(model, train_loader, optimizer, lr_scheduler, epoch)
        tfboard_writer.add_scalar("train/seg-loss", train_loss, epoch)
        tfboard_writer.add_scalar("train/ori-loss", ori_losses, epoch)
        tfboard_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # test and record logs
        if epoch % args.test_freq == 0 or epoch == args.epochs-1:
            for k, loader in test_loaders.items():
                logger.info("Testing on '%s' dataset..."%k)
                test_loss, test_mae, ori_acc = test(model, loader, epoch)

                tfboard_writer.add_scalar("test-seg-loss/" + k, test_loss, epoch)
                tfboard_writer.add_scalar("test-mae/" + k, test_mae, epoch)
                tfboard_writer.add_scalar("test-ori-acc" + k, ori_acc, epoch)

        # adjust lr
        if epoch in args.milestones:
            args.lr = args.lr * args.gamma
            logger.info("Epoch %d, decay learning rate to %f"%(epoch, args.lr))
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

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
    if args.freeze_bn:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, data in enumerate(train_loader):

        image = data["image"]
        target = data["label"]
        # edge = data["edge"]
        orientation = data["orientation"]
        mask = data["mask"]


        image = image.cuda() # (non_blocking=True)
        target = target.cuda().float() # (non_blocking=True).float()
        # edge = edge.cuda(non_blocking=True).float()
        orientation = orientation.cuda(non_blocking=True).long().squeeze(dim=1)
        mask = mask.cuda(non_blocking=True).squeeze(dim=1)

        # measure data loading time
        data_time.update(time.time() - end)
        output = model(image)

        seg_loss = F.binary_cross_entropy_with_logits(output["seg"], target, reduction=args.reduction)

        ori_loss = F.cross_entropy(output["orientation"], orientation, reduction="none")
        ori_loss = (ori_loss * mask).sum() / mask.sum()

        seg_losses.update(seg_loss.item(), image.size(0))
        ori_losses.update(ori_loss.item(), image.size(0))

        # compute and adjust lr
        if lr_scheduler is not None:
            lr = lr_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # compute gradient and do SGD step
        loss = seg_loss + ori_loss
        optimizer.zero_grad()

        # scale loss before backward
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]
        if i % args.print_freq == 0:
            logger.info('Epoch[{0}/{1}] It[{2}/{3}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'SegLoss {seg_loss.val:.3f} ({seg_loss.avg:.3f})\t'
                    'OriLoss {ori_loss.val:.3f} ({ori_loss.avg:.3f})\t'
                    'LR: {lr:.2E}'.format(
                    epoch, args.epochs, i, train_loader_len,
                    batch_time=batch_time, data_time=data_time, seg_loss=seg_losses, ori_loss=ori_losses, lr=lr))

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
            mask = data["mask"].cuda().squeeze(dim=1)
            orientation = data["orientation"].cuda().long().squeeze(dim=1)

            # compute output
            output = model(image)
            seg_pred = torch.sigmoid(output["seg"])
            seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(output["seg"], target, reduction=args.reduction)

            ori_pred = output["orientation"].transpose(0,1)[:, mask].t()
            ori_gt = orientation[mask]
            ori_top1, ori_top2 = accuracy(ori_pred, ori_gt, topk=(1,2))

            # mae and F-measure
            mae_ = (seg_pred - target).abs().mean()

            filenames = [splitext(split(i)[-1])[0] for i in metas["filename"]]
            diff = (seg_pred - target).detach().cpu().numpy()
            diff = (diff * 127 + 128)

            if (epoch+1) % 10 == 0 or epoch == args.epochs-1:
                seg_pred = (seg_pred >= 0.5).cpu().numpy()*255
                gt = target.cpu().numpy()*255
                diff = np.concatenate((seg_pred, gt, diff), axis=3).astype(np.uint8)
                save_maps(diff, filenames, join(args.tmp, "epoch-%d"%epoch, "diff"))

            seg_losses.update(seg_loss.item(), image.size(0))
            mae.update(mae_.item(), image.size(0))
            ori_acc1.update(ori_top1.item(), image.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger.info('Test: [{0}/{1}]\t'
                    'SegLoss {seg_loss.val:.3f} (avg={seg_loss.avg:.3f})\t'
                    'MAE {mae.val:.3f} (avg={mae.avg:.3f})\t'
                    'OriAcc={ori_acc.avg:.3f}\t'.format(
                    i, test_loader_len, seg_loss=seg_losses, mae=mae, ori_acc=ori_acc1))

    logger.info(' * MAE {mae.avg:.5f} OriAcc={ori_acc.avg:.3f}'\
        .format(mae=mae, ori_acc=ori_acc1))

    return seg_losses.avg, mae.avg, ori_acc1.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
