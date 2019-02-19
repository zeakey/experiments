
import torch
import torch.nn as nn

class PReLU(nn.Module):

    def __init__(self, planes):
        super(PReLU, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pos_slope = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1,
                                groups=planes, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):

        w = self.bn(self.pos_slope(self.avgpool(x)))
        w = torch.sigmoid(w)

        y = x * w.expand_as(x)
        y[x < 0] *= 0

        return y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)