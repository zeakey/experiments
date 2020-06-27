import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import drn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir="tmp")

net = drn.drn_d_22(num_classes=64).cuda()
load_dict=torch.load("../../deep-usps/Pretrained_Models/drn_pretraining/drn_d_22_cityscapes.pth")
for name, p in net.state_dict().items():
    new_name = name.replace("layer", "base.")
    if new_name in load_dict.keys():
        if load_dict[new_name].shape == p.shape:
            p.copy_(load_dict[new_name])
#             print("name %s copied!"%name)

class Model(nn.Module):
    def __init__(self, num_classes=16):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.cnn = net
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=3)
        self.linear = nn.Linear(128, num_classes*3)
    def forward(self, x):
        x = self.cnn(x)
        N, C, H, W = x.shape

        # N C H W -> H N W C
        x = x.permute((2, 0, 3, 1))
        x = x.reshape(H, N*W, C)

        # input of shape (seq_len, batch, input_size)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        x, _ = self.lstm(x) # [H, N*W, C]
        x = self.linear(x) # [H, N*W, 3*num_classes]
        # H N W 3 num_classes -> N H W 3 num_classes
        x = x.view(H, N, W, 3, self.num_classes).permute(1, 0, 2, 3, 4)

        return x

model = Model().cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
        "/media/ssd1/ilsvrc12/train/",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#             normalize,
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=32, pin_memory=True, sampler=None)

train_loader_len = len(train_loader)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


def train(model, train_loader, epoch):
    losses = AverageMeter('Loss', ':.4e')
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time

        images = images.cuda(non_blocking=True)
        target = torch.floor(images * 15).long().flatten()

        N, _, H, W = images.shape

        pred = model(images)
        
        pred = pred.reshape(-1, 16)


        loss = torch.nn.functional.cross_entropy(pred, target, reduction='none')

        edge = loss.detach()
        edge = (edge.reshape(N, H, W, 3)**2).sum(dim=3).sqrt()
        edge.unsqueeze_(dim=1)
        edge /= edge.max()

        loss = loss.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print("Iter [%d/%d]: loss=%f"%(i+1, len(train_loader), losses.avg))
            writer.add_images("edges", edge, epoch*train_loader_len+i)
    return edge

train(model, train_loader, epoch=0)