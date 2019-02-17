
import torch
import torch.nn as nn

class PReLU(nn.Module):

    def __init__(self, planes):
        super(PReLU, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1,
                                groups=planes, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):

        w = self.bn(self.conv1x1(self.avgpool(x)))
        w = torch.sigmoid(w)
        w = w.expand_as(x)

        y = x * w
        y[x < 0] *= 0

        return y