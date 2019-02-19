
import torch
import torch.nn as nn

class PReLU(nn.Module):

    def __init__(self, planes, neg_slope=0.25, reduction=4):
        super(PReLU, self).__init__()
        self.neg_slope = neg_slope
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc0 = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        w = self.avgpool(x).view(b, c)

        w0 = self.fc0(w).view(b, c, 1, 1)
        w0 = w0.expand_as(x)
        
        w1 = self.fc1(w).view(b, c, 1, 1)
        w1 = w1.expand_as(x)
        w1 = w1 * self.neg_slope

        y = torch.zeros_like(x)

        y[x > 0] = x[x > 0] * w0[x > 0]
        y[x < 0] = x[x < 0] * w1[x < 0]

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
