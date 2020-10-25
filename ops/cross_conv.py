import torch
import torch.nn as nn

class CrossConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=0, padding=0, bias=True):
        super(CrossConv, self).__init__()
        assert out_channels % 2 == 0
        self.conv_h = nn.Conv2d(in_channels, out_channels//2, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
        self.conv_v = nn.Conv2d(in_channels, out_channels//2, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=bias)

    def forward(self, x):
        y1 = self.conv_h(x)
        y2 = self.conv_v(x)
        return torch.cat((y1, y2), dim=1)
