# https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
import torch, torchvision
import torch.nn as nn
import torchvision, cv2
# logger and auxliaries
import numpy as np
import math
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vlkit import get_logger, run_path
from vlkit.pytorch import save_checkpoint, AverageMeter, accuracy
import vlkit.pytorch as vlpytorch
from vlkit.lr import CosAnnealingLR, MultiStepLR
from vlkit import image as vlimage

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

# distributed
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import FP16_Optimizer, BN_convert_float

from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

from models import preact_resnet, modules, sphere_face, sphereface
import face_verification

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Legacy autograd function with non-static forward method is deprecated and will be removed", UserWarning)


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
# model
parser.add_argument('--model', default="preact_resnet.resnet34", help='backbone model')
parser.add_argument('--linear', default="modules.NormLinear", help='linear layer')
# data
parser.add_argument('--use-rec', action="store_true")
parser.add_argument('--dali-cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--lfwdir', metavar='DIR', default="", help='path to LFW dataset')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--num-classes', default=10572, type=int, metavar='N', help='Number of classes (10572|85742)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
# optimizer
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=2, help="warmup epochs")
parser.add_argument('--milestones', default="10,15,18", type=str)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/resnet34-linear")
parser.add_argument('--label-smoothing', type=float, default=0.0, help="label-smoothing temperature")
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
# margin
# cos(m1\theta + m2)
parser.add_argument("--m1", default=1, type=int)
parser.add_argument("--m2", default=0, type=float)
parser.add_argument("--s", default=64, type=float)
parser.add_argument("--max-lam", default=0.1666, type=float)
parser.add_argument("--max-lam-iter", default=2000, type=int)
# data aug
parser.add_argument("--hole", action="store_true")
# seed
parser.add_argument("--seed", default=7, type=int)
args = parser.parse_args()

args.milestones = [int(i) for i in args.milestones.split(',')]

# https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=args.seed+device_id)
        
        # MXnet rec reader
        self.input = ops.MXNetReader(path=join(data_dir, "train.rec"), index_path=join(data_dir, "train.idx"),
                                    random_shuffle=True, shard_id=args.local_rank, num_shards=args.world_size)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        self.resize = ops.Resize(device=dali_device, resize_x=112, resize_y=112, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(112, 112),
                                            image_type=types.RGB)
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

# loss function
if args.label_smoothing == 0:
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = torch.nn.KLDivLoss(reduction="batchmean")

if args.local_rank == 0:
    tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
    logger = get_logger(join(args.tmp, "log.txt"))

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
    else:
        args.world_size = 1
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
    train_loader_len = int(train_loader._size / args.batch_size)

    # model and optimizer
    # model = preact_resnet.resnet34()
    model = args.model+"().cuda()"
    # linear = args.linear+"(in_features=512, out_features=%d).cuda()" % (args.num_classes)
    # linear = "nn.Linear(512, args.num_classes).cuda()"
    linear = "modules.NormLinear(512, args.num_classes, s=args.s).cuda()"
    model = eval(model)
    linear = eval(linear)

    if args.fp16:
        model = BN_convert_float(model.half())
        linear = BN_convert_float(linear.half())
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
        linear = DDP(linear)

    optimizer = torch.optim.SGD(
        [{"params": model.parameters()},
        {"params": linear.parameters()}],
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
    best_lfw = 0

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            if args.local_rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_lfw = checkpoint['best_lfw']
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
    # scheduler = CosAnnealingLR(loader_len=train_loader_len,
    #                            max_lr=args.lr, min_lr=1e-5,
    #                            epochs=args.epochs,
    #                            warmup_epochs=args.warmup_epochs)

    if args.local_rank == 0:
        lfwacc, lfwthres = test_lfw(model)
        tfboard_writer.add_scalar('test/lfw-acc', lfwacc, -1)
        tfboard_writer.add_scalar('test/lfw-thres', lfwthres, -1)
        logger.info("Initial LFW accuracy %f" % lfwacc)

    for epoch in range(args.start_epoch, args.epochs):
        # train and evaluate
        loss = train(train_loader, (model, linear), optimizer, scheduler, epoch)

        if args.local_rank == 0:
            lfwacc, lfwthres = test_lfw(model)
            if lfwacc > best_lfw:
                best_lfw = lfwacc

            logger.info("Epoch %d: LFW accuracy %f (best=%f)"%(epoch, lfwacc, best_lfw))
            tfboard_writer.add_scalar('test/lfw-acc', lfwacc, epoch)
            tfboard_writer.add_scalar('test/best-lfw-acc', best_lfw, epoch)
            tfboard_writer.add_scalar('test/lfw-thres', lfwthres, epoch)

        if args.local_rank == 0:
            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': {**model.state_dict(), **linear.state_dict()},
                'best_lfw': best_lfw,
                'optimizer' : optimizer.state_dict()},
                True, path=args.tmp)

            if args.distributed:
                weight_norm = linear.module.weight.data
            else:
                weight_norm = linear.weight.data
            weight_norm = torch.norm(weight_norm, p=2, dim=1).mean().item()
            tfboard_writer.add_scalar('train/weight-norm', weight_norm, epoch)
            tfboard_writer.add_scalar('train/loss', loss, epoch)
            tfboard_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)

    if args.local_rank == 0:
        logger.info("Optimization done (best lfw-acc=%.4f), ALL results saved to %s." % (best_lfw, args.tmp))
        tfboard_writer.close()
        for h in logger.handlers:
            h.close()

def train(train_loader, model, optimizer, lrscheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_loader_len = int(np.ceil(train_loader._size/args.batch_size))

    model, linear = model
    # switch to train mode
    model.train()
    linear.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        target = data[0]["label"].squeeze().cuda().long()
        data = data[0]["data"]
        data = data - 127.5
        data = data * 0.0078125
        bs = data.shape[0]
        assert target.max() < args.num_classes, "%d vs %d" % (target.max(), args.num_classes)

        if args.fp16:
            data = data.half()

        # measure data loading time
        data_time.update(time.time() - end)
        iter_index = epoch * train_loader_len + i

        if epoch < args.warmup_epochs:
            lam = 0
        else:
            lam_effective_iter = (epoch - args.warmup_epochs) * train_loader_len + i
            if False:
                lam = max(5, 1000 / (lam_effective_iter * 0.1 + 1))
                lam = 1 / (1 + lam)
            else:
                if lam_effective_iter >= args.max_lam_iter:
                    lam = args.max_lam
                else:
                    lam = (1 - math.cos(lam_effective_iter * math.pi / args.max_lam_iter)) / 2 * args.max_lam

        feature = model(data)
        output = linear(feature)#, target)
        if args.label_smoothing == 0:
            loss = criterion(output, target)
        else:
            with torch.no_grad():
                label_distr = torch.zeros_like(output)
                noise = args.label_smoothing / (1 - args.num_classes)
                confidence = 1 - args.label_smoothing
                label_distr.fill_(noise)
                label_distr.scatter_(1, target.unsqueeze(1), confidence)
            loss = criterion(torch.log_softmax(output, dim=1), label_distr)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            losses.update(reduced_loss.item(), data.size(0))
        else:
            losses.update(loss.item(), data.size(0))
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

        # do gradient-clipping
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.grad.clamp_(-1, 1)

        optimizer.step()

        torch.cuda.synchronize()

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:

            lr = optimizer.param_groups[0]["lr"]
            fnorm = torch.norm(feature.float(), p=2, dim=1).detach().cpu().numpy()
            if args.distributed:
                wnorm = torch.norm(linear.module.weight.data.float(), p=2, dim=1).detach().cpu().numpy()
            else:
                wnorm = torch.norm(linear.weight.data.float(), p=2, dim=1).detach().cpu().numpy()

            # output orientation
            mask = torch.zeros(output.shape, dtype=torch.bool, device=target.device)
            mask[torch.arange(bs).to(device=target.device), target] = True
            output = output.detach()
            cos_pos = output[mask].view(-1)
            cos_neg = output[~mask].view(-1)
            # cos(gradient, output)
            if linear.module.weight.grad is not None:
                gn = torch.index_select(linear.module.weight.grad, dim=0, index=target)
                gn = torch.nn.functional.normalize(gn, p=2, dim=1)
                fn = torch.nn.functional.normalize(feature.detach(), p=2, dim=1)
                cos_grad = (gn*fn).sum(dim=1)
            else:
                cos_grad = torch.zeros_like(target)

            tfboard_writer.add_scalar("train/iter-lr", lr, epoch*train_loader_len+i)
            tfboard_writer.add_scalar("train/iter-lambda", lam, epoch*train_loader_len+i)
            tfboard_writer.add_scalar("train/iter-acc1", top1.val, epoch*train_loader_len+i)
            tfboard_writer.add_scalar("train/iter-loss", losses.val, epoch*train_loader_len+i)
            tfboard_writer.add_scalar('train/iter-feature-norm', fnorm.mean(), epoch*train_loader_len+i)
            tfboard_writer.add_scalar('train/iter-weight-norm', wnorm.mean(), epoch*train_loader_len+i)

            tfboard_writer.add_histogram('train/iter-cos-pos', cos_pos, epoch*train_loader_len+i)
            tfboard_writer.add_histogram('train/iter-cos-neg', cos_neg, epoch*train_loader_len+i)
            tfboard_writer.add_histogram('train/iter-cos-grad', cos_grad, epoch*train_loader_len+i)

            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'BTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'LR: {lr:.2E} lambda: {lambd:.2E}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1,
                   lr=lr, lambd=lam))

        if args.local_rank == 0 and iter_index % 1000 == 0:
            N = 16

            norm = torch.norm(feature.float(), p=2, dim=1).detach()
            loss = torch.nn.functional.cross_entropy(output.detach(), target, reduction="none")
            _, predict = torch.max(output, dim=1)
            probability = torch.nn.functional.softmax(output, dim=1)

            _, norm_sort = torch.sort(norm)
            _, loss_sort = torch.sort(loss)

            selected_idx = torch.cat((norm_sort[0:N], norm_sort[-N::], loss_sort[0:N], loss_sort[-N::]))
            selected = data[selected_idx,].cpu().detach().numpy().astype(np.float32)
            for idx in range(selected.shape[0]):
                im1 = np.squeeze(selected[idx,]).transpose((1,2,0))
                im1 = vlimage.norm255(im1)
                im1 = Image.fromarray(im1)
                draw = ImageDraw.Draw(im1)
                font = ImageFont.truetype("FreeSans.ttf", 12)

                idx1 = selected_idx[idx]
                pred = int(predict[idx1])
                tgt = int(target[idx1])
                prob1 = probability[idx1, pred]
                prob2 = probability[idx1, tgt]

                if pred == tgt:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                x, y = 2, 10
                draw.text((x, y), "loss:%.4f"%loss[idx1], color, font=font); y += 25
                draw.text((x, y), "norm:%.4f"%norm[idx1], color, font=font); y += 25
                draw.text((x, y), "pred:%d/%.5f"%(pred, prob1), color, font=font); y += 25
                draw.text((x, y), "target:%d/%.5f"%(tgt, prob2), color, font=font); y += 25

                im1 = np.array(im1).transpose((2,0,1))
                selected[idx,] = im1
            selected = torch.from_numpy(selected)
            selected = torchvision.utils.make_grid(selected, nrow=N, normalize=True)
            tfboard_writer.add_image("train/images", selected, epoch*train_loader_len+i)
            tfboard_writer.add_histogram('train/norm-distr', norm.cpu().numpy(), epoch*train_loader_len+i)
            tfboard_writer.add_histogram('train/loss-distr', loss.cpu().numpy(), epoch*train_loader_len+i)
            selected = selected.numpy().transpose((1,2,0))
            selected = vlimage.norm255(selected)
            save_path = join(args.tmp, "images/train", "epoch%d-iter%d.jpg"%(epoch, i))
            os.makedirs(dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, selected)

            # draw loss-norm scatter figure
            loss = loss.cpu().numpy(); norm = norm.cpu().numpy()
            fig, ax = plt.subplots(1, 1)
            ax.scatter(loss, norm)
            ax.set_xlabel("loss")
            ax.set_ylabel("norm")
            ax.set_title("feature norm v.s. loss value")
            plt.tight_layout()
            save_path = join(args.tmp, "images/norm-loss", "epoch%d-iter%d.jpg"%(epoch, i))
            os.makedirs(dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)

    train_loader.reset()

    return losses.avg

@torch.no_grad()
def test_lfw(model):
    model.eval()
    data = torch.load("lfw-112x112.pth", map_location=torch.device('cpu'))
    label = data["label"].numpy()
    data = data["data"].float()
    data = data - 127.5
    data = data * 0.0078125
    if args.fp16:
        data = data.half()
    feature = np.zeros((12000, 512), dtype=np.float32)
    for i in range(0, 12000, 100):
        d1 = data[i:i+100,].cuda()
        d2 = torch.flip(d1, dims=(3,))
        o1 = model(d1)
        o2 = model(d2)

        if True:
            o = o1+o2
        else:
            o = torch.cat((o1, o2), dim=1)
        o = o.detach().cpu().numpy()

        feature[i:i+100,] = o

    acc, thres = face_verification.verification(feature, label)
    return acc.mean(), thres.mean()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
