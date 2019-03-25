import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
torch.backends.cudnn.bencmark = True

import os, sys, random, datetime, time, argparse, math
from os.path import isdir, isfile, isdir, join, dirname, abspath
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.io import savemat

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import vltools
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch.datasets import ilsvrc2012
import vltools.pytorch as vlpytorch
from vltools.tcm.lr import CosAnnealingLR

from models.resnet import Resnet20
from utils import LFWDataset

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=512)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=0.2)
parser.add_argument('--scheduler', type=str, help='lr scheduler', default="step")
parser.add_argument('--warmup-epochs', type=int, help='warmup epochs', default=5)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=18)
parser.add_argument('--gamma', type=float, help='gamma', default=0.1)
parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
parser.add_argument('--start-epoch', type=int, help='max epoch', default=0)
parser.add_argument('--epochs', type=int, help='max epoch', default=20)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--tmp', type=str, help='tmp files', default="tmp/exclusive")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# ex related
parser.add_argument('--ex-all', action="store_true", default=False)
parser.add_argument('--ex-euclidean', action="store_true", default=False)
parser.add_argument('--ex-reduction', type=str, default="sum")
parser.add_argument('--ex-weight', type=float, help='exclusive loss weight', default=1)
parser.add_argument('--ex-start-epoch', type=float, help='epoch when exclusive-reg starts', default=16)
# datasets
parser.add_argument('--dali', action="store_true", default=False)
parser.add_argument('--data', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

if args.dali:
    from dali_loader import HybridTrainPipe
    import nvidia.dali.plugin.pytorch as plugin_pytorch

# Fix random seed for reproducibility
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert isfile(args.lfwlist)
assert isdir(args.lfw)
assert args.ex_start_epoch < args.epochs

args.tmp = join(abspath(args.tmp))
os.makedirs(args.tmp, exist_ok=True)

logger = Logger(join(args.tmp, "log.txt"))
logger.info(args)


    
logger.info("Pre-loading training data...")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader_len = -1
if args.dali:
    NUM_GPUS = 1
    NUM_THREADS = 2
    pipes = [HybridTrainPipe(batch_size=int(args.bs/NUM_GPUS), num_threads=NUM_THREADS, device_id=device_id,
    data_dir=args.data, crop=224, num_gpus=NUM_GPUS, dali_cpu=False) for device_id in range(NUM_GPUS)]
    pipes[0].build()
    train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))
    train_loader_len = int(train_loader._size / args.bs)
else:
    train_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=8, pin_memory=True)
    train_loader_len = len(train_loader)
    
logger.info("Done!")

lfw_dataset = LFWDataset("data/lfw-112X96", "data/LFW_imagelist.txt",
        transforms= transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

lfw_loader = torch.utils.data.DataLoader(
    lfw_dataset, batch_size=400, shuffle=False,
    num_workers=8, pin_memory=True)

# transforms for LFW testing data
test_transform = transforms.Compose([
  transforms.ToTensor(),
  normalize
])

# model and optimizer
logger.info("Loading model...")

class NormalizedLinear(nn.Module):
    
    def __init__(self, in_features, out_features, radius=10):

        super(NormalizedLinear, self).__init__()
        self.radius = float(radius)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.reset_parameters()
  
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        
        weight_norm = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

        x_norm = x_norm * self.radius
        # see https://github.com/pytorch/pytorch/blob/372d1d67356f054db64bdfb4787871ecdbbcbe0b/torch/nn/modules/linear.py#L55
        return torch.nn.functional.linear(x_norm, weight_norm, None)

class Model(nn.Module):

    def __init__(self, dim=512, num_class=10572, radius=10):
        super(Model, self).__init__()
        self.num_class = num_class
        self.base = Resnet20()
        self.radius = radius
        self.fc6 = NormalizedLinear(dim, num_class, radius=self.radius)

    def forward(self, x):
        
        features = self.base(x)
        if self.training:
            logits = self.fc6(features)
            return features, logits
        else:
            return features

model = Model(num_class=10575)
logger.info("Transporting model to GPU(s)...")
model.cuda()
logger.info("Done!")

# optimizer related
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)


def exclusive_loss(weight, reduction="sum"):
    assert weight.ndimension() == 2 or weight.ndimension() == 4, weight.ndimension()
    weight = weight.view(weight.size(0), -1)
    
    # N: number of kernels
    # K: kernel dimension
    N, K = weight.shape
    
    normed_weight = torch.nn.functional.normalize(weight, p=2, dim=1)
    cos = torch.mm(normed_weight, normed_weight.t())
    cos = torch.clamp(cos, min=-1, max=1)
    if args.ex_euclidean:
        cos = -2 * torch.sqrt((1 - cos) / 2)

    if reduction == "sum":
        mask = torch.eye(N).cuda()
        mask = 1 - mask
        loss = (cos * mask).sum() / (N * (N - 1))
    elif reduction == "nearest":
        mask = cos.clone().detach()
        mask.scatter_(0, torch.arange(N).view(1, -1).cuda(), -1)
        _, indices = torch.max(mask, dim=0)
        mask.zero_()
        mask.scatter_(0, indices.view(1, -1), 1.0)
        loss = (mask * cos).sum() / N
    else:
        raise ValueError("bad reduction %s" % reduction)
    
    return loss

def decorrelated_loss(weight, reduction="mean"):
    weight = weight.view(weight.size(0), -1)
    weight = weight - torch.mean(weight, dim=1, keepdim=True)
    weight = torch.nn.functional.normalize(weight, p=2, dim=1)
    corr = torch.mm(weight, weight.t())
    corr = corr - corr.diag().diag()
    return torch.pow(corr, 2).mean()

lr_record = []
train_loss_record = []
train_exloss_record = []
train_acc1_record = []
lfw_acc_record = []

if args.resume:
    if isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # records
        lr_record = checkpoint["lr_record"]
        train_loss_record = checkpoint["train_loss_record"]
        train_exloss_record = checkpoint["train_exloss_record"]
        train_acc1_record = checkpoint["train_acc1_record"]
        lfw_acc_record = checkpoint["lfw_acc_record"]

        logger.info("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

model.load_state_dict(torch.load("tmp/normalize/checkpoint.pth")["state_dict"])
    
if args.scheduler == "step":
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8,], gamma=args.gamma)
elif args.scheduler == "cos":
    scheduler = CosAnnealingLR(args.epochs*train_loader_len, lr_max=args.lr,
                    lr_min=0.001, warmup_iters=args.warmup_epochs*train_loader_len)
else:
    raise ValueError("Unsupported scheduler %s" % args.scheduler)

def train(train_loader, model, optimizer, epoch):
    # recording
    loss = AverageMeter()
    ex_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train_loader_len = int(train_loader._size / args.bs)
    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        if args.dali:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
        else:
            input = data[0].cuda()
            target = data[1].cuda().long()

        data_time.update(time.time() - end)

        prob = model(input)
        loss0 = criterion(prob, target)

        # exclusive_loss
        ex_weight = args.ex_weight if epoch >= args.ex_start_epoch else 0
        if ex_weight != 0:
            if args.ex_all:
                loss1 = []
                for name, p in model.named_parameters():
                    if ("conv" in name or "fc" in name) and "weight" in name:
                        loss1.append(decorrelated_loss(p, reduction=args.ex_reduction))
                loss1 = sum(loss1)
            else:
                loss1 = decorrelated_loss(dict(model.named_parameters())["fc6.weight"])
        else:
            loss1 = torch.zeros_like(loss0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(prob, target, topk=(1, 5))
        loss.update(loss0.item(), input.size(0))
        ex_loss.update(loss1.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        train_loss_record.append(loss.val)
        train_exloss_record.append(ex_loss.val)
        train_acc1_record.append(top1.val)

        # clear cached gradient
        optimizer.zero_grad()


        # backward gradient
        if ex_weight == 0:
            loss0.backward()
        else:
            (loss0 + ex_weight * loss1).backward()

        # update parameters
        optimizer.step()

        if args.scheduler == "cos":
            
            lr = scheduler.step()
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        lr_record.append(optimizer.param_groups[0]["lr"])

        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            logger.info("Epoch [{0}/{1}] Iter[{2}/{3}]\t"
                        "Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Train loss {loss.val:.3f} ({loss.avg:.3f})\t"
                        "Ex loss (*{4}) {ex_loss.val:.2e} ({ex_loss.avg:.2e})\t"
                        "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Acc@5 {top5.val:.3f} ({top5.avg:.3f}) LR {lr:.2e}".format(
                        epoch, args.epochs, batch_idx, train_loader_len, ex_weight,
                        batch_time=batch_time, data_time=data_time, loss=loss,
                        ex_loss=ex_loss, top1=top1, top5=top5, lr=lr_record[-1]))

    return loss.avg, ex_loss.avg, top1.avg

def test_lfw(model, imglist, epoch):

    model.eval() # switch to evaluate mode
    features = np.zeros((len(imglist), 512 * 2), dtype=np.float32)

    logger.info("Extracting features for evaluation...")

    with torch.no_grad():
        for idx, input in enumerate(tqdm(lfw_loader)):
            
            feature0 = model(input.cuda())
            feature1 = model(torch.flip(input.cuda(), dims=(3,)))

            feature = torch.cat((feature0, feature1), dim=1)

            if idx == 0:
                features = torch.cat((feature0, feature1), dim=1)
            else:
                features = torch.cat((features, torch.cat((feature0, feature1), dim=1)), dim=0)

        features = features.detach().cpu().numpy()
            

    from test_lfw import fold10
    lfw_acc = fold10(features, cache_fn=join(args.tmp, "epoch(%d)-lfw-acc.txt" % epoch))
    savemat(join(args.tmp, "epoch(%d)-features.mat" % epoch), dict({"features": features}))
    return lfw_acc

def main():

    with open(args.lfwlist, 'r') as f:
        imglist = f.readlines()
        imglist = [join(args.lfw, i.rstrip()) for i in imglist]

    for epoch in range(args.start_epoch, args.epochs):
        

        start = time.time()
        train_loss, train_exloss, train_acc1 = train(train_loader, model, optimizer, epoch)
        if args.dali:
            train_loader.reset()
        logger.info("Epoch %d finished training (%.3f sec)" % (epoch, time.time() - start))

        # adjust lr       
        if args.scheduler == "step":
            scheduler.step()

        # test
        lfw_acc = test_lfw(model, imglist, epoch)

        # save records
        lfw_acc_record.append(lfw_acc)

        is_best = lfw_acc_record[-1] == max(lfw_acc_record)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "lr_record": lfw_acc_record,
            "train_loss_record": train_loss_record,
            "train_exloss_record": train_exloss_record,
            "train_acc1_record": train_acc1_record,
            "lfw_acc_record": lfw_acc_record},
            is_best=is_best, path=args.tmp)

        if epoch == args.ex_start_epoch - 1:
            torch.save({"epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_record": lfw_acc_record,
                        "train_loss_record": train_loss_record,
                        "train_exloss_record": train_exloss_record,
                        "train_acc1_record": train_acc1_record,
                        "lfw_acc_record": lfw_acc_record},
                       join(args.tmp, "checkpoint-before-ex.pth"))

        logger.info("Epoch %d best LFW accuracy is %.5f." % (epoch, max(lfw_acc_record)))


        plt.style.use('seaborn')
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].plot(train_loss_record) # loss
        axes[0].set_title("Train Loss")

        axes[1].plot(train_exloss_record) # loss
        axes[1].set_title("Ex Loss")

        axes[2].plot(train_acc1_record) # top1acc
        axes[2].set_title("Train Acc1")

        axes[3].plot(lfw_acc_record)
        axes[3].set_title("LFW Acc (best=%.3f)" % max(lfw_acc_record))

        axes[4].plot(lr_record)
        axes[4].set_title("LR")

        plt.tight_layout()
        plt.savefig(join(args.tmp, 'record.pdf'))
        plt.close()

        savemat(join(args.tmp, 'record.mat'),
                dict({"train_acc1_record": train_acc1_record,
                      "train_loss_record": train_loss_record,
                      "train_exloss_record": train_exloss_record,
                      "lfw_acc_record": lfw_acc_record}))

if __name__ == '__main__':
  main()

