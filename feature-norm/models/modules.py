import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class NormLinear(nn.Module):
    def __init__(self, in_features, out_features, s=64):
        super(NormLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, 1.0)

    def forward(self, input):

        input = F.normalize(input)
        self.weight.data = F.normalize(self.weight.data)

        return self.s * F.linear(input, self.weight)

class MarginLinear(nn.Module):
    def __init__(self, in_features, out_features, s=64):
        super(MarginLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s

        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, m1=1, m2=0, label=None, lambd=1):

        input = F.normalize(input)
        cosine = F.linear(input, F.normalize(self.weight))

        m1 = int(m1)

        if label is None or (m1 == 1 and m2 == 0):
            output = cosine * self.s
            return output

        elif m2 != 0 and m1 == 1:
            # additive margin

            # sin(theta)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            cos_m2 = math.cos(m2)
            sin_m2 = math.sin(m2)
            # psi = cos(theta + m2)
            psi_theta = cosine*cos_m2 - sine*sin_m2
            psi_theta = torch.where(cosine>-cos_m2, psi_theta, -psi_theta-2)

        elif m2 == 0 and m1 != 1:
            # multiplitive margin
            cos_m1_theta = self.mlambda[m1](cosine)
            theta = cosine.acos()
            k = (m1 * theta / math.pi).floor()
            psi_theta = (-1)**k * cos_m1_theta - 2*k

        else:
            theta = torch.acos(cosine)
            def psi(theta, m1, m2):
                cos_m1_theta = torch.cos(m1*theta)
                cos_m1_theta_plus_m2 = torch.cos(m1*theta+m2)

                k = ((m1 * theta + m2 ) / math.pi).floor()
                psi_theta = (-1)**k * cos_m1_theta_plus_m2 - 2*k

                return psi_theta

            psi_theta = psi(theta, m1, m2)

        one_hot = torch.zeros_like(cosine, dtype=bool)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = torch.where(one_hot, psi_theta*lambd+cosine*(1-lambd), cosine)

        output *= self.s

        return output

class ArcLinear(nn.Module):
    """
    ArcFace
    """
    def __init__(self, in_features, out_features, s=64):
        super(ArcLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, m=0.5):

        input = F.normalize(input)
        cosine = F.linear(input, F.normalize(self.weight))

        if label is None or m == 0:
            output = cosine * self.s
            return output

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        # psi = cos(theta + m)
        psi_theta = cosine*cos_m - sine*sin_m
        psi_theta = torch.where(cosine>-cos_m, psi_theta, -psi_theta-2)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot*psi_theta + (1-one_hot)*cosine
        output *= self.s
        return output

class ArcMarginModel(nn.Module):
    def __init__(self, in_features, out_features, s=64, m=0.5):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = m
        self.s = s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
