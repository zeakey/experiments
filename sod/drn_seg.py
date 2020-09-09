import torch, math
import torch.nn as nn
import drn
import numpy as np

# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/guided_anchor_head.py
from mmcv.ops import DeformConv2d, MaskedConv2d
from mmcv.cnn import bias_init_with_prob, normal_init

def orientation2flux(orientation, num_classes=8):
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
                 kernel_size=1,
                 deform_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            256, deform_groups * offset_channels, 1, bias=False)
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
    def __init__(self, model_name, classes, setting, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(out_middle=False, pretrained=False, num_classes=1000)
        # self.base = nn.Sequential(*list(model.children())[:-2],
        #     nn.Conv2d(model.out_dim, 256, kernel_size=1, bias=False))

        self.setting = setting

        self.base = model
        self.conv1x1 = nn.Conv2d(model.out_dim, 256, kernel_size=1)

        self.feature_adaption = FeatureAdaption(256, 256, kernel_size=3, deform_groups=4)

        self.seg_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.edge_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.ori_head = nn.Conv2d(256, 8, kernel_size=1, bias=False)

        self.def_conv = DeformConv2d(
            256,
            256,
            kernel_size=1,
            padding=0,
            deform_groups=1)

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
        # normal_init(self.base[-1], std=0.1)
        normal_init(self.seg_head, std=0.01)
        normal_init(self.edge_head, std=0.01)

        normal_init(self.ori_head, std=0.01)
        normal_init(self.def_conv, std=0.01)
        # self.def_conv.weight.data.copy_(torch.eye(256).unsqueeze_(dim=-1).unsqueeze_(dim=-1))
        # self.def_conv.weight.requires_grad = False

    def forward(self, x):
        x = self.conv1x1(self.base(x))
        orientation = self.ori_head(x)

        # #
        flux = orientation2flux(orientation.detach())
        flux = flux.to(dtype=x.dtype)

        if self.setting == 0:
            x_adapt = x
        elif self.setting == 1:
            x_adapt = self.feature_adaption(x, x.detach())
        elif self.setting == 2:
            x_adapt = self.feature_adaption(x, flux.detach())
        elif self.setting == 3:
            x_adapt = x + self.feature_adaption(x, flux.detach())

        seg = self.seg_head(x_adapt)
        seg = self.up1(seg)

        # edge = self.edge_head(x_adapt)
        # edge = self.up1(edge)
        orientation = self.up2(orientation)

        results = dict({
            "seg": seg,
            # "edge": edge,
            "orientation": orientation
        })

        return results


if __name__ == '__main__':
    model = DRNSeg("drn_d_22", classes=1)
    data = torch.rand(10, 3, 224, 224)
    y = model(data)