import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class MarginLinear(nn.Module):
    def __init__(self, in_features, out_features, s=64, m1=1, m2=0.5, m3=0):
        super(MarginLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        assert self.m1 == 1
        assert self.m3 == 0

        self.sin_m2 = math.sin(self.m2)
        self.cos_m2 = math.cos(self.m2)
        self.min_cos = math.cos(math.pi - self.m2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = F.normalize(self.weight.data)
    

    def forward(self, input, label=None):

        input = F.normalize(input)
        self.weight.data = F.normalize(self.weight.data)

        # cos(theta)
        cosine = F.linear(input, self.weight)
        # print("input", input.mean(), "weight", weight.mean())

        if label is None:
            output = cosine * self.s
        else:
            # sin(theta)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            # phi = cos(theta + m2)
            phi = cosine*self.cos_m2 - sine*self.sin_m2

            phi = torch.where(cosine>self.min_cos, phi, -cosine-2)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

        return output
