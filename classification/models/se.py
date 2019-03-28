import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_conv1x1(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer_conv1x1, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avgpool(x)
        y = self.fc(y)
        return x * y.expand_as(x)
