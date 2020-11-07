import torch
import torch.nn as nn

class CrossConvV1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=0, bias=True):
        super(CrossConvV1, self).__init__()
        assert out_planes % 2 == 0 and kernel_size == 3
        self.conv_h = nn.Conv2d(in_planes, out_planes//2, kernel_size=(1, kernel_size), stride=stride, groups=groups,
                                padding=(0, padding), bias=bias)

        self.conv_v = nn.Conv2d(in_planes, out_planes//2, kernel_size=(kernel_size, 1), stride=stride, groups=groups,
                                padding=(padding, 0), bias=bias)

        nn.init.kaiming_normal_(self.conv_h.weight.data, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_v.weight.data, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        y1 = self.conv_h(x)
        y2 = self.conv_v(x)
        return torch.cat((y1, y2), dim=1)

class CrossConvV2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=0, bias=True):
        super(CrossConvV2, self).__init__()
        assert out_planes % 2 == 0 and kernel_size == 3
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

        nn.init.kaiming_normal_(self.conv.weight.data, mode="fan_out", nonlinearity="relu")

        self.register_buffer("weight_mask", torch.ones_like(self.conv.weight.data))
        #
        self.weight_mask[:out_planes//2, :, 0, 0] = 0
        self.weight_mask[:out_planes//2, :, 2, 0] = 0
        self.weight_mask[:out_planes//2, :, 0, 2] = 0
        self.weight_mask[:out_planes//2, :, 2, 2] = 0
        #
        self.weight_mask[out_planes//2::, :, 1, 0] = 0
        self.weight_mask[out_planes//2::, :, 0, 1] = 0
        self.weight_mask[out_planes//2::, :, 2, 1] = 0
        self.weight_mask[out_planes//2::, :, 1, 2] = 0

        self.conv.weight.data = self.conv.weight.data * self.weight_mask

    def forward(self, x):
        self.conv.weight.data = self.conv.weight.data * self.weight_mask
        return self.conv(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    assert dilation == 1 and stride == 1 and groups == 1
    return CrossConvV2(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False)

if __name__ == "__main__":
    x = torch.rand(2, 3, 128, 128)
    conv = conv3x3(3, 64)

    print(conv.conv.weight[32,0,:,:])