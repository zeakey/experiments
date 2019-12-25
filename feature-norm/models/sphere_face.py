import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Sphere20(nn.Module):
    def __init__(self, num_classes=10572):
        super(Sphere20, self).__init__()
        self.num_classes = num_classes
        #input = B*3*112*112
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*7,512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)

        x = self.fc5(x)

        return x

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):

        super(AngleLinear, self).__init__()

        self.weight = Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, 1.0)

        self.m = m
        self.psi = Psi.apply


    def forward(self, input, label, lam=0):

        self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
        output = self.psi(input, self.weight, label, self.m, lam)

        return output



mlambda = [
    lambda x: x ** 0,
    lambda x: x,
    lambda x: 2 * x ** 2 - 1,
    lambda x: 4 * x ** 3 - 3 * x,
    lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
    lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
]
mcoeff_w = [
    lambda x: 0,
    lambda x: 1,
    lambda x: 4 * x,
    lambda x: 12 * x ** 2 - 3,
    lambda x: 32 * x ** 3 - 16 * x,
    lambda x: 80 * x ** 4 - 60 * x ** 2 + 5
]
mcoeff_x = [
    lambda x: -1,
    lambda x: 0,
    lambda x: 2 * x ** 2 + 1,
    lambda x: 8 * x ** 3,
    lambda x: 24 * x ** 4 - 8 * x ** 2 - 1,
    lambda x: 64 * x ** 5 - 40 * x ** 3
]

class Psi(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, label, m, lam):

        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        wn = F.normalize(weight, p=2, dim=1)
        xn = F.normalize(input, p=2, dim=1)
        cos_theta = xn.mm(wn.t())
        cos_m_theta = mlambda[m](cos_theta)

        k = (m * cos_theta.acos() / math.pi).floor()
        psi_theta = (-1) ** k * cos_m_theta - 2 * k

        onehot = torch.zeros_like(cos_theta, dtype=bool)
        onehot = onehot.scatter_(1, label.view(-1, 1), 1)

        output = torch.where(onehot,
                             psi_theta * lam + cos_theta * (1 - lam),
                             cos_theta)
        output = output * xlen

        # save for backward
        cos_theta = (cos_theta * onehot.to(input.dtype)).sum(dim=1).view(-1, 1)
        k = (k * onehot.to(input.dtype)).sum(dim=1).view(-1, 1)
        lam = torch.tensor(lam, device=input.device)
        m = torch.tensor(m, device=input.device)
        ctx.save_for_backward(input, label, weight, cos_theta, k, m, lam)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):

        input, label, weight , cos_theta, k, m, lam = ctx.saved_tensors
        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        onehot = torch.zeros_like(grad_outputs, dtype=bool)
        onehot = onehot.scatter_(1, label.view(-1, 1), 1)
        m = m.item()

        grad_input = grad_outputs.mm(weight) * (1-lam)
        grad_outputs_label = (grad_outputs*onehot.to(grad_outputs.dtype)).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** k) * mcoeff_w[m](cos_theta)
        coeff_x = (((-1) ** k) * mcoeff_x[m](cos_theta) + 2 * k) / xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) * lam
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input * lam
        grad_input += (grad_outputs * (1 - onehot.to(grad_outputs.dtype))).mm(weight) * lam
        grad_weight = grad_outputs.t().mm(input)

        return grad_input, grad_weight, None, None, None
