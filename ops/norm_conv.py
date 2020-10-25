import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(MyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, kernel_size]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input):
        h_in, w_in = input.shape[2:]
        ks = self.kernel_size[0]
        h_out = math.floor((h_in + 2*self.padding - self.dilation*(ks-1)-1)/self.stride+1)
        w_out = math.floor((w_in + 2*self.padding - self.dilation*(ks-1)-1)/self.stride+1)

        # x: [bs ksize num_sliding]
        x = torch.nn.functional.unfold(input, kernel_size=ks, padding=self.padding, stride=self.stride)

        bs = input.shape[0]
        ksize = self.in_channels*ks*ks
        num_sliding = x.shape[2]

        assert x.shape[1] == ksize

        # x: [bs*num_sliding ksize]
        x = torch.transpose(x, 1, 2).reshape(-1, ksize)
        weight_flat = self.weight.view(self.out_channels, ksize)

        # normalize
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        weight_flat = torch.nn.functional.normalize(weight_flat, p=2, dim=1)


        x = torch.mm(x, weight_flat.t())

        x = x.reshape(bs, num_sliding, self.out_channels)
        x = torch.transpose(x, 1, 2)
        x = torch.nn.functional.fold(x, output_size=[h_out, w_out], kernel_size=1, padding=0, dilation=1, stride=1)
        return x


if __name__ == '__main__':
    input = torch.rand(10, 32, 1, 1)
    weight = torch.rand(64, 32, 3, 3)

    conv2d = MyConv2d(32, 64, kernel_size=3, padding=1)
    conv2d.weight.data.copy_(weight)

    y = conv2d(input)

    print(y.mean(), y.std())

    conv2d = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
    conv2d.weight.data.copy_(weight)

    y = conv2d(input)

    print(y.mean(), y.std())