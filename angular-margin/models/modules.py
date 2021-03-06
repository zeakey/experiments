import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

def shift_softmax(x, eps=10-6):
    x = torch.exp(x+1) - 1 + eps
    assert torch.all(x > 0)
    y = x.sum(dim=1).view(-1, 1)
    return x / y


class NormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.01)

    def forward(self, input):
        input = F.normalize(input, dim=1, p=2)
        weight = F.normalize(self.weight, dim=1, p=2)
        output = F.linear(input, weight)
        return output

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
    def __init__(self, in_features, out_features, m=0.5):
        super(ArcLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None or self.m == 0:
            return cosine

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # psi = cos(theta + m)
        psi_theta = cosine*self.cos_m - sine*self.sin_m
        psi_theta = torch.where(cosine > -self.cos_m, psi_theta, -psi_theta-2)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot*psi_theta + (1-one_hot)*cosine
        return output

class ArcLinear2(nn.Module):
    """
    ArcFace
    """
    def __init__(self, in_features, out_features, m1=0.5, m2=0.2):
        super(ArcLinear2, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.m1 = m1
        self.m2 = m2
        self.cos_m1 = math.cos(self.m1)
        self.sin_m1 = math.sin(self.m1)

        self.cos_m2 = math.cos(self.m2)
        self.sin_m2 = math.sin(self.m2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        if self.m1 > 0:
            # psi1 = cos(theta + m1)
            psi_theta1 = cosine*self.cos_m1 - sine*self.sin_m1
            psi_theta1 = torch.where(cosine > -self.cos_m1, psi_theta1, -psi_theta1-2)
        else:
            psi_theta1 = cosine

        if self.m2 > 0:
            # psi2 = cos(theta - m2)
            psi_theta2 = cosine*self.cos_m2 + sine*self.sin_m2
            psi_theta2 = torch.where(cosine < self.cos_m2, psi_theta2, -psi_theta2+2)
        else:
            psi_theta2 = cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot * psi_theta1 + (1-one_hot) * psi_theta2

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
