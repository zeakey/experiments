import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from forest import Forest
from vltools import Logger
from vltools.pytorch import AverageMeter, accuracy
import os

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = torchvision.datasets.MNIST('/home/kai/.torch/data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.MNIST('/home/kai/.torch/data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.ip1 = nn.Linear(800, 128)
        self.ip2 = Forest(in_features=128, num_trees=8, tree_depth=5, num_classes=10)
    
    def forward(self, x):

        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.ip1(x))
        x = self.ip2(x)

        return x

os.makedirs("tmp", exist_ok=True)
logger = Logger("tmp/mnist.log.txt")

model = LeNet().cuda()

weights = []
pi = []
for name, p in model.named_parameters():
    if 'pi' in name:
        pi.append(p)
    else:
        weights.append(p)

optimizer = optim.Adam([{"params": pi, "weight_decay": 0},
                        {"params": weights}], lr=0.001, weight_decay=1e-5)

def main():

    for epoch in range(30):
        
        trn_acc1 = AverageMeter()
        trn_loss = AverageMeter()

        for it, (data, target) in enumerate(train_loader):

            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = F.nll_loss(torch.log(output), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target)[0]
            trn_acc1.update(acc1.item(), data.size(0))
            trn_loss.update(loss.item(), data.size(0))

            if it % 20 == 0:
                logger.info("Training: Epoch %d, iter %d, Loss=%.3f, Accuracy=%.3f" % (epoch, it, trn_loss.val, trn_acc1.val))
        
        logger.info("Training epoch %d done, avg loss=%.3f, avg accuracy=%.3f" % (epoch, trn_loss.avg, trn_acc1.avg))
        
        val_loss = AverageMeter()
        val_acc1 = AverageMeter()

        for it, (data, target) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = F.nll_loss(torch.log(output), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            acc1 = accuracy(output, target)[0]
            val_acc1.update(acc1.item(), data.size(0))
            val_loss.update(loss.item(), data.size(0))
            
        logger.info("Testing epoch %d done, avg loss=%.3f, avg accuracy=%.3f" % (epoch, val_loss.avg, val_acc1.avg))

if __name__ == "__main__":
    main()

