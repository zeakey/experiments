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
from vlkit.training import get_logger, run_path
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
from utils import save_maps, get_full_image_names, load_batch_images
from crf import batch_crf, par_batch_crf


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--arch', '--a', metavar='STR', default="drn_d_105", help='model')
# data
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--use-rec', action='store_true', help='Use mxnet record.')
parser.add_argument('--batch-size', default=20, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--imsize', default=432, type=int, metavar='N', help='im crop size')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
# optimization
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=2, help="warmup epochs")
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
parser.add_argument('--tmp', help='tmp folder', default="tmp/pre-resnet50")
parser.add_argument('--pretrained', help='pretrained model', type=str, default="Pretrained_Models/drn_pretraining/drn-d-105_ms_cityscapes.pth")
# Loss function
parser.add_argument('--loss', help='loss function', type=str, default="ce")
parser.add_argument('--floss-beta', help='floss beta', type=float, default=1)
# FP16
parser.add_argument('--fp16', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
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

args.tmp = run_path(args.tmp)
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

DTYPE = torch.float
if args.fp16:
    from apex import amp
    DTYPE = torch.half
    assert args.distributed, "FP16 can be only used in Distributed mode"
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

if args.loss == "ce":
    criterion = torch.nn.BCEWithLogitsLoss()
elif args.loss == "floss":
    criterion = FLoss(beta=args.floss_beta)


def main():
    if args.local_rank == 0:
        logger.info(args)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.Resize([args.imsize]*2), transforms.ToTensor(), normalize])
    label_transform = transforms.Compose([transforms.Resize([args.imsize]*2, interpolation=Image.NEAREST), transforms.ToTensor()])

    train_dataset = SODDataset(img_dir="/media/ssd0/deep-usps-data/01_img/",
                        label_dirs=["/media/ssd0/deep-usps-data/02_gt/",
                                    "/media/ssd0/deep-usps-data/03_mc/",
                                    "/media/ssd0/deep-usps-data/04_hs/",
                                    "/media/ssd0/deep-usps-data/05_dsr/",
                                    "/media/ssd0/deep-usps-data/06_rbd/"],
                        name_list="/home/kai/Code1/deep-usps/Parameters/train_names.txt",
                        flip=False, image_transform=image_transform, label_transform=label_transform)

    test_dataset = SODDataset(img_dir="/media/ssd0/deep-usps-data/01_img/",
                    label_dirs=["/media/ssd0/deep-usps-data/02_gt/"],
                    name_list="/home/kai/Code1/deep-usps/Parameters/test_names.txt",
                    flip=False, image_transform=image_transform, label_transform=label_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=4,
        pin_memory=True, drop_last=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=4,
        pin_memory=True, drop_last=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False, sampler=test_sampler)

    train_loader_len = len(train_loader)

    # model and optimizer
    model = DRNSeg(args.arch, classes=1)

    # add model graph
    if args.local_rank == 0:
        tfboard_writer.add_graph(model, torch.zeros(1, 3, args.imsize, args.imsize))

    if args.pretrained is not None:
        assert isfile(args.pretrained)
        load_dict = torch.load(args.pretrained)
        own_dict=model.state_dict()
        for name, param in load_dict.items():
            if name not in own_dict:
                if args.local_rank == 0:
                    logger.info('Parameter %s not found in own model.'%name)
                continue
            if own_dict[name].size() != load_dict[name].size():
                if args.local_rank == 0:
                    logger.info('Parameter shape does not match: {} ({} vs. {}).'.format(name, own_dict[name].size(), load_dict[name].size()))
            else:
                own_dict[name].copy_(param)
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

    scheduler = MultiStepLR(loader_len=train_loader_len, base_lr=args.lr,
                   milestones=args.milestones, gamma=args.gamma, warmup_epochs=args.warmup_epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train and evaluate
        loss, predictions, indice = train(train_loader, model, optimizer, scheduler, epoch)
        mae = test(test_loader, model, epoch)

        assert predictions.min() >= 0 and predictions.max() <= 1
        assert predictions.shape[0] == indice.size, "predictions shape: {} v.s. indice size: {}".format(predictions.shape, indice.size)

        if args.local_rank == 0:
            logger.info("Loading images...")
        names = np.array(train_dataset.get_item_names())[indice].tolist()
        full_image_names = get_full_image_names(train_dataset.img_dir, names=names, ext=".jpg", warn_exist=True)
        images = load_batch_images(full_image_names, H=args.imsize, W=args.imsize)

        if args.local_rank == 0:
            logger.info("Saving predictions to %s" % join(args.tmp, "epoch-%d"%epoch, "predictions"))
        save_maps(maps=(predictions*255).astype(np.uint8), names=names, dir=join(args.tmp, "epoch-%d"%epoch, "predictions"))

        if args.local_rank == 0:
            start = time.time()
            logger.info("Applying crf to predictions (results saved to %s)" % join(args.tmp, "epoch-%d"%epoch, "crf"))
        crf_results = par_batch_crf(images, predictions)
        save_maps(maps=(crf_results*255).astype(np.uint8), names=names, dir=join(args.tmp, "epoch-%d"%epoch, "crf"), ext=".png")
        if args.local_rank == 0:
            logger.info("Done! (%f sec elapsed.)" % (time.time()-start))

        crf_results3 = (np.repeat(crf_results, axis=1, repeats=3)*255).astype(np.uint8)
        predictions3 = (np.repeat(predictions, axis=1, repeats=3)*255).astype(np.uint8)
        crf_results3 = np.concatenate((images, predictions3, crf_results3), axis=3)
        save_maps(maps=crf_results3, names=names, dir=join(args.tmp, "epoch-%d"%epoch, "crf"), ext=".jpg")

        mva_dir = join(args.tmp, "epoch-%d"%epoch, "mva-crf")
        full_mva_names = get_full_image_names(mva_dir, names=names, ext=".png", warn_exist=False)
        alpha = 0.7
        if epoch == 0:
            mva_crf = crf_results.copy()
        else:
            mva_crf = load_batch_images(full_mva_names, H=args.imsize, W=args.imsize)
            mva_crf = alpha * mva_crf + (1 - alpha) *  crf_results
        save_maps(maps=(mva_crf*255).astype(np.uint8), names=names, dir=mva_dir, ext=".png")


        is_best = mae < best_mae
        if is_best:
            best_mae = mae

        if args.local_rank == 0:
            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae': best_mae,
                'optimizer' : optimizer.state_dict()},
                is_best, path=args.tmp)

            if (epoch+1) in args.milestones:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, path=args.tmp, filename="checkpoint-epoch%d.pth"%epoch)

            logger.info("Best mae=%.5f" % best_mae)
            tfboard_writer.add_scalar('train/loss', loss, epoch)
            tfboard_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)
            tfboard_writer.add_scalar('test/mae', mae, epoch)

    if args.local_rank == 0:
        logger.info("Optimization done, ALL results saved to %s." % args.tmp)
        tfboard_writer.close()
        for h in logger.handlers:
            h.close()

threshold = {
    "mc": 0.31,
    "hs": 0.36
}


def train(train_loader, model, optimizer, lrscheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_loader_len = len(train_loader)
    # switch to train mode
    model.train()

    predictions = torch.empty(0, 1, args.imsize, args.imsize, dtype=DTYPE).cuda()
    indice = torch.empty(0, dtype=torch.long).cuda()

    end = time.time()
    for i, data in enumerate(train_loader):
        # data = [d.cuda() for d in data]
        data, gt, mc, hs, drs, rbd, idx = data
        if args.distributed:
            data = data.cuda(non_blocking=True).to(dtype=DTYPE)
        idx = idx.cuda().to(dtype=torch.long)
        mc = mc.cuda(non_blocking=True)
        target = mc>threshold["mc"]

        # measure data loading time
        data_time.update(time.time() - end)
        output = model(data)

        predictions = torch.cat((predictions, torch.sigmoid(output.detach()).to(dtype=DTYPE)), dim=0)
        indice = torch.cat((indice, idx), dim=0)

        target = target.to(dtype=output.dtype)
        loss = criterion(output, target)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), data.size(0))

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
            tfboard_writer.add_scalar("train/iter-lr", lr, epoch*train_loader_len+i)
            tfboard_writer.add_scalar("train/iter-loss", losses.val, epoch*train_loader_len+i)

            logger.info('Epoch[{0}/{1}] It[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'LR: {lr:.2E}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, loss=losses, lr=lr))

        if args.local_rank == 0 and i == 0:
            data = torchvision.utils.make_grid(data, normalize=True)
            pred = torchvision.utils.make_grid(torch.sigmoid(output), normalize=True)
            tfboard_writer.add_image("images", data, epoch)
            tfboard_writer.add_image("predictions", pred, epoch)

    return losses.avg, predictions.cpu().numpy(), indice.cpu().numpy()

def test(test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae =  AverageMeter()
    # switch to evaluate mode
    model.eval()
    test_loader_len = len(test_loader)

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(test_loader):
            names = data.pop()
            data = [d.cuda() for d in data]
            data, gt = data

            # compute output
            output = model(data)
            loss = criterion(output, gt)
            # mae
            mae_ = (torch.sigmoid(output) - gt).abs().mean()

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_mae = reduce_tensor(mae_)
            else:
                reduced_loss = loss.data
                reduced_mae = mae_.data

            losses.update(reduced_loss.item(), data.size(0))
            mae.update(reduced_mae.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'MAE {mae.val:.3f} (avg={mae.avg:.3f})'.format(
                       i, test_loader_len, loss=losses, mae=mae))


    if args.local_rank == 0:
        logger.info(' * MAE {mae.avg:.5f}'.format(mae=mae))

    return mae.avg

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss

if __name__ == '__main__':
    main()
