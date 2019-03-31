import torch
import torch.nn as nn
from torch.autograd import Function

class RepReLU_conv1x1(nn.Module):

    def __init__(self, planes, neg_slope=0.25, zero_weights=False):
        super(RepReLU_conv1x1, self).__init__()

        self.neg_slope = neg_slope

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_pos = nn.Sequential(
            nn.Conv2d(planes, planes, groups=planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fc_neg = nn.Sequential(
            nn.Conv2d(planes, planes, groups=planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.__initiate_params__(zero_weights)
    
    def __initiate_params__(self, zero_weights):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if zero_weights:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        b, c, _, _ = x.size()
        w = self.avgpool(x)

        w_pos = self.fc_pos(w).view(b, c, 1, 1).expand_as(x)
        w_neg = self.fc_neg(w).view(b, c, 1, 1).expand_as(x)

        w_neg = w_neg * self.neg_slope

        w = torch.where(x >= 0, w_pos, w_neg)
        result = x * w

        return result

class RepReLU_conv1x1bn(nn.Module):

    def __init__(self, planes, neg_slope=0.25, zero_weights=False):
        super(RepReLU_conv1x1bn, self).__init__()

        self.neg_slope = neg_slope

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_pos = nn.Sequential(
            nn.Conv2d(planes, planes, groups=planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Sigmoid()
        )

        self.fc_neg = nn.Sequential(
            nn.Conv2d(planes, planes, groups=planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Sigmoid()
        )

        self.__initiate_params__(zero_weights)

    def __initiate_params__(self, zero_weights):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if zero_weights:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        b, c, _, _ = x.size()
        w = self.avgpool(x)

        w_pos = self.fc_pos(w).view(b, c, 1, 1).expand_as(x)
        w_neg = self.fc_neg(w).view(b, c, 1, 1).expand_as(x)

        w_neg = w_neg * self.neg_slope

        w = torch.where(x >= 0, w_pos, w_neg)
        result = x * w

        return result

class RePReLU_fc(nn.Module):

    def __init__(self, planes, neg_slope=0.25, reduction=4):
        super(RePReLU_fc, self).__init__()
        self.neg_slope = neg_slope
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_pos = nn.Sequential(
            nn.Linear(planes, planes//reduction, bias=False),
            nn.Linear(planes//reduction, planes, bias=False),
            nn.Sigmoid()
        )

        self.fc_neg = nn.Sequential(
            nn.Linear(planes, planes//reduction, bias=False),
            nn.Linear(planes//reduction, planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        w = self.avgpool(x).view(b, c)

        w_pos = self.fc_pos(w).view(b, c, 1, 1).expand_as(x)
        w_neg = self.fc_neg(w).view(b, c, 1, 1).expand_as(x)

        w_neg = w_neg * self.neg_slope

        w = torch.where(x >= 0, w_pos, w_neg)
        result = x * w

        return x
