import torch, math
import torch.nn as nn
import drn
import numpy as np

from mmcv.ops import DeformConv2d, MaskedConv2d
from mmcv.cnn import bias_init_with_prob, normal_init

def orientation2vec(orientation, num_classes=8):
    assert orientation.shape[1] == num_classes
    N, _, H, W = orientation.shape

    orientation = torch.argmax(orientation, dim=1)

    angle = orientation * 2*np.pi / num_classes

    vec = torch.zeros(N, 2, H, W, device=angle.device, dtype=angle.dtype)
    cos, sin = angle.sin(), angle.cos()
    
    vec[:, 0,] = torch.where(angle<=np.pi, sin, -sin)
    vec[:, 1,] = cos

    return vec

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deform_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, y):
        offset = self.conv_offset(y.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2],
            nn.Conv2d(model.out_dim, 256, kernel_size=1, bias=False))

        self.feature_adaption = FeatureAdaption(256, 256, kernel_size=3, deform_groups=4)

        self.seg = nn.Conv2d(256, classes, kernel_size=1, bias=False)

        self.orientation = nn.Conv2d(256, 8, kernel_size=1, bias=False)


        self.init_params()
        

        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up2 = nn.ConvTranspose2d(8, 8, 16, stride=8, padding=4,
                                    output_padding=0, groups=8,
                                    bias=False)
            up1 = nn.ConvTranspose2d(1, 1, 16, stride=8, padding=4,
                                    output_padding=0, groups=1,
                                    bias=False)
            fill_up_weights(up1)
            fill_up_weights(up2)
            up1.weight.requires_grad = False
            up2.weight.requires_grad = False
            self.up1 = up1
            self.up2 = up2

    def init_params(self):
        torch.nn.init.xavier_normal_(self.seg.weight.data)
        torch.nn.init.normal_(self.orientation.weight.data, mean=0, std=0.01)
        normal_init(self.base[-1], std=0.1)

    def forward(self, x):
        x = self.base(x)

        orientation = self.orientation(x)

        #
        vec = orientation2vec(orientation.detach())
        vec = vec.to(dtype=x.dtype)

        x = self.feature_adaption(x, vec)

        seg = self.seg(x)
        seg = self.up1(seg)
        orientation = self.up2(orientation)

        return seg, orientation


if __name__ == '__main__':
    model = DRNSeg("drn_d_22", classes=1)
    data = torch.rand(10, 3, 224, 224)
    y = model(data)