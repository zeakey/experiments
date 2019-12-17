import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

class AngleSoftmax(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, m=4, lambda_max=1000.0, lambda_min=5.0, 
                 power=1.0, gamma=0.1, loss_weight=1.0):
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
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(int(output_size), input_size))
        nn.init.kaiming_uniform_(self.weight, 1.0)
        self.m = m

        self.it = 0
        self.LambdaMin = lambda_min
        self.LambdaMax = lambda_max
        self.gamma = gamma
        self.power = power

    def forward(self, x, y):

        if self.normalize:
            wl = self.weight.pow(2).sum(1).pow(0.5)
            wn = self.weight / wl.view(-1, 1)
            self.weight.data.copy_(wn.data)
        lamb = max(self.LambdaMin, self.LambdaMax / (1 + self.gamma * self.it)**self.power)
        self.it += 1
        phi_kernel = PhiKernel(self.m, lamb)
        feat = phi_kernel(x, self.weight, y)

        return feat

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
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
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
