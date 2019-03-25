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

from models.resnet import Resnet20
from utils import LFWDataset

parser = argparse.ArgumentParser(description='PyTorch Implementation of HED.')
parser.add_argument('--bs', type=int, help='batch size', default=600)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=18)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--wd', type=float, help='weight decay', default=5e-4)
parser.add_argument('--epochs', type=int, help='max epoch', default=30)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
parser.add_argument('--debug', type=str, help='debug mode', default='false')
parser.add_argument('--tmp', type=str, help='tmp files', default="tmp/normalized")
parser.add_argument('--resume', type=str, help='checkpoint path', default=None)
# datasets
parser.add_argument('--data', type=str, help='root folder of CASIA-WebFace dataset', default="data/CASIA-WebFace-112X96")
parser.add_argument('--lfw', type=str, help='LFW dataset root folder', default="data/lfw-112X96")
parser.add_argument('--lfwlist', type=str, help='lfw image list', default='data/LFW_imagelist.txt')
args = parser.parse_args()

# Fix random seed for reproducibility
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert isfile(args.lfwlist)
assert isdir(args.lfw)

args.tmp = join(abspath(args.tmp))
os.makedirs(args.tmp, exist_ok=True)

logger = Logger(join(args.tmp, "log.txt"))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    
logger.info("Pre-loading training data...")

train_dataset = datasets.ImageFolder(
    args.data,
    transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.bs, shuffle=True,
    num_workers=8, pin_memory=True)
    
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
        
        x = self.base(x)
        if self.training:
            x = self.fc6(x)

        return x

model = Model(num_class=10575)
logger.info("Done!")

# optimizer related
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

logger.info("Transporting model to GPU(s)...")
model.cuda()
logger.info("Done!")


def train(train_loader, model, optimizer, epoch):
    # recording
    loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # train_loader_len = int(train_loader._size / args.bs)
    train_loader_len = len(train_loader)
    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, data in enumerate(train_loader):

        input = data[0].cuda()
        target = data[1].cuda().long()
        data_time.update(time.time() - end)

        prob = model(input)
        loss0 = criterion(prob, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(prob, target, topk=(1, 5))
        loss.update(loss0.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # clear cached gradient
        optimizer.zero_grad()
        # backward gradient
        loss0.backward()
        # update parameters
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            logger.info("Epoch [{0}/{1}] Iter[{2}/{3}]\t"
                        "Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Train loss {loss.val:.3f} ({loss.avg:.3f})\t"
                        "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch, args.epochs, batch_idx, train_loader_len,
                        batch_time=batch_time, data_time=data_time, loss=loss,
                        top1=top1, top5=top5))
    return loss.avg, top1.avg

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

train_loss_record = []
train_acc1_record = []
lfw_acc_record = []

def main():

    with open(args.lfwlist, 'r') as f:
        imglist = f.readlines()
        imglist = [join(args.lfw, i.rstrip()) for i in imglist]

    for epoch in range(args.epochs):
        
        scheduler.step() # will adjust learning rate

        start = time.time()
        train_loss, train_acc1 = train(train_loader, model, optimizer, epoch)
        logger.info("Epoch %d finished training (%.3f sec)" % (epoch, time.time() - start))

        # test
        lfw_acc = test_lfw(model, imglist, epoch)
        # save records
        lfw_acc_record.append(lfw_acc)
        train_acc1_record.append(train_acc1)
        train_loss_record.append(train_loss)

        is_best = lfw_acc_record[-1] == max(lfw_acc_record)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=is_best, path=args.tmp)

        logger.info("Epoch %d best LFW accuracy is %.5f." % (epoch, max(lfw_acc_record)))


        fig, axes = plt.subplots(1, 3, figsize=(6, 2))

        axes[0].plot(train_loss_record) # loss
        axes[0].set_title("Train Loss")
        
        axes[1].plot(train_acc1_record) # top1acc
        axes[1].set_title("Train Acc1")
        
        axes[2].plot(lfw_acc_record)
        axes[2].set_title("LFW Acc (best=%.3f)" % max(lfw_acc_record))

        plt.savefig(join(args.tmp, 'record.pdf'))

        savemat(join(args.tmp, 'record.mat'),
                dict({"train_acc1_record": train_acc1_record,
                      "train_loss_record": train_loss_record,
                      "lfw_acc_record": lfw_acc_record}))

if __name__ == '__main__':
  main()

