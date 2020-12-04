import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np
import math

class AngleSoftmax(nn.Module):
    def __init__(self, input_size, output_size, m=4):
        '''
        :param input_size: Input channel size.
        :param output_size: Number of Class.
        :param normalize: Whether do weight normalization.
        :param m: An integer, specifying the margin type, take value of [0,1,2,3,4,5].
        :param lambda_max: Starting value for lambda.
        :param lambda_min: Minimum value for lambda.
        :param power: Decreasing strategy for lambda.
        :param gamma: Decreasing strategy for lambda.
        :param loss_weight: Loss weight for this loss. 
        '''
        super(AngleSoftmax, self).__init__()

        self.weight = Parameter(torch.Tensor(int(output_size), input_size))
        nn.init.kaiming_uniform_(self.weight, 1.0)
        self.m = int(m)

        self.phi_kernel = PhiKernel4.apply

    def forward(self, x, y, lamb):

        self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
        # lamb = max(5, 1000 / (1 + 0.1 * self.it)**1.0)
        # lamb = 1 / (lamb+1)
        # self.it += 1
        # if self.it % 100 == 0:
        #     print("lambda = %f"%lamb)
        # phi_kernel = PhiKernel1(self.m, lamb)
        feat = self.phi_kernel(x, self.weight, y, self.m, lamb)

        return feat

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):

        super(AngleLinear, self).__init__()

        self.weight = Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, 1.0)

        self.m = m
        self.psi = PhiKernel4.apply


    def forward(self, input, label, lam=0):

        self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
        output = self.psi(input, self.weight, label, self.m, lam)

        return output

class PhiKernel(Function):
    def __init__(self, m, lamb):
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.mcoeff_w = [
            lambda x: 0,
            lambda x: 1,
            lambda x: 4 * x,
            lambda x: 12 * x ** 2 - 3,
            lambda x: 32 * x ** 3 - 16 * x,
            lambda x: 80 * x ** 4 - 60 * x ** 2 + 5
        ]
        self.mcoeff_x = [
            lambda x: -1,
            lambda x: 0,
            lambda x: 2 * x ** 2 + 1,
            lambda x: 8 * x ** 3,
            lambda x: 24 * x ** 4 - 8 * x ** 2 - 1,
            lambda x: 64 * x ** 5 - 40 * x ** 3
        ]
        self.m = m
        self.lamb = lamb

    def forward(self, input, weight, label):
        # xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        # wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        wn = F.normalize(weight, p=2, dim=1)
        xn = F.normalize(input, p=2, dim=1)
        #cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        cos_theta = xn.mm(wn.t())
        cos_m_theta = self.mlambda[self.m](cos_theta)

        k = (self.m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = torch.zeros_like(feat, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        feat[index] -= feat[index] / (1.0 + self.lamb)
        feat[index] += phi_theta[index] / (1.0 + self.lamb)
        self.save_for_backward(input, weight, label)
        self.cos_theta = (cos_theta * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        self.k = (k * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        self.xlen, self.index = xlen, index
        return feat

    def backward(self, grad_outputs):
        input, weight, label = self.saved_variables
        input, weight, label = input.data, weight.data, label.data
        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * self.lamb / (1.0 + self.lamb)
        grad_outputs_label = (grad_outputs*self.index.to(grad_outputs.dtype)).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** self.k) * self.mcoeff_w[self.m](self.cos_theta)
        coeff_x = (((-1) ** self.k) * self.mcoeff_x[self.m](self.cos_theta) + 2 * self.k) / self.xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) / (1.0 + self.lamb)
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input / (1.0 + self.lamb)
        grad_input += (grad_outputs * (1 - self.index.to(grad_outputs.dtype))).mm(weight) / (1.0 + self.lamb)
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None

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

class PhiKernel1(Function):
    def __init__(self, m, lamb):
        self.m = m
        self.lamb = lamb

    def forward(self, input, weight, label):
        # xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        # wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        wn = F.normalize(weight, p=2, dim=1)
        xn = F.normalize(input, p=2, dim=1)
        #cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        cos_theta = xn.mm(wn.t())
        cos_m_theta = mlambda[self.m](cos_theta)

        k = (self.m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = torch.zeros_like(feat, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        feat[index] -= feat[index] * self.lamb
        feat[index] += phi_theta[index] * self.lamb
        self.save_for_backward(input, weight, label)
        self.cos_theta = (cos_theta * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        self.k = (k * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        self.xlen, self.index = xlen, index
        return feat

    def backward(self, grad_outputs):
        input, weight, label = self.saved_variables
        input, weight, label = input.data, weight.data, label.data
        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * (1-self.lamb)
        grad_outputs_label = (grad_outputs*self.index.to(grad_outputs.dtype)).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** self.k) * mcoeff_w[self.m](self.cos_theta)
        coeff_x = (((-1) ** self.k) * mcoeff_x[self.m](self.cos_theta) + 2 * self.k) / self.xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) * self.lamb
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input * self.lamb
        grad_input += (grad_outputs * (1 - self.index.to(grad_outputs.dtype))).mm(weight) * self.lamb
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None

class PhiKernel2(Function):

    @staticmethod
    def forward(ctx, input, weight, label, m, lamb):
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        # wn = F.normalize(weight, p=2, dim=1)
        # xn = F.normalize(input, p=2, dim=1)
        cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        # cos_theta = xn.mm(wn.t())
        cos_m_theta = mlambda[m](cos_theta)

        k = (m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = torch.zeros_like(feat, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        feat[index] -= feat[index] * lamb
        feat[index] += phi_theta[index] * lamb

        lamb = torch.tensor(lamb, device=input.device)
        m = torch.tensor(m, device=input.device)
        cos_theta = (cos_theta * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        k = (k * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        
        ctx.save_for_backward(input, weight, label, cos_theta, k, m, lamb)
        return feat

    @staticmethod
    def backward(grad_outputs):

        input, weight, label, cos_theta, k, m, lamb = ctx.saved_variables
        input, weight, label = input.data, weight.data, label.data
        lamb = lamb.item()
        m = m.item()
        index = torch.zeros_like(feat, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)

        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * (1-lamb)
        grad_outputs_label = (grad_outputs*index.to(grad_outputs.dtype)).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** k) * mcoeff_w[m](cos_theta)
        coeff_x = (((-1) ** k) * mcoeff_x[m](cos_theta) + 2 * k) / xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) * lamb
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input * lamb
        grad_input += (grad_outputs * (1 - index.to(grad_outputs.dtype))).mm(weight) * lamb
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None, None

class PhiKernel3(Function):

    @staticmethod
    def forward(ctx, input, weight, label, m, lamb):
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        xlen = torch.norm(input, p=2, dim=1).view(-1, 1)
        # wn = F.normalize(weight, p=2, dim=1)
        # xn = F.normalize(input, p=2, dim=1)
        cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        # cos_theta = xn.mm(wn.t())
        cos_m_theta = mlambda[m](cos_theta)

        k = (m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = torch.zeros_like(feat, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        feat[index] -= feat[index] * lamb
        feat[index] += phi_theta[index] * lamb

        lamb = torch.tensor(lamb, device=input.device)
        m = torch.tensor(m, device=input.device)
        cos_theta = (cos_theta * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        k = (k * index.to(input.dtype)).sum(dim=1).view(-1, 1)
        
        ctx.save_for_backward(input, weight, label, cos_theta, k, m, lamb)
        return feat

    @staticmethod
    def backward(ctx, grad_outputs):

        input, weight, label, cos_theta, k, m, lamb = ctx.saved_variables
        input, weight, label = input.data, weight.data, label.data
        lamb = lamb.item()
        m = m.item()
        index = torch.zeros_like(grad_outputs, dtype=bool)
        index = index.scatter_(1, label.view(-1, 1), 1)
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B

        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * (1-lamb)
        grad_outputs_label = (grad_outputs*index.to(grad_outputs.dtype)).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** k) * mcoeff_w[m](cos_theta)
        coeff_x = (((-1) ** k) * mcoeff_x[m](cos_theta) + 2 * k) / xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) * lamb
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input * lamb
        grad_input += (grad_outputs * (1 - index.to(grad_outputs.dtype))).mm(weight) * lamb
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None, None, None

class PhiKernel4(torch.autograd.Function):

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
