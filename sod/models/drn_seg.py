import torch, math
import torch.nn as nn
import torch.nn.functional as F
import drn
import numpy as np
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/guided_anchor_head.py
from mmcv.ops import DeformConv2d, MaskedConv2d
from mmcv.cnn import bias_init_with_prob, normal_init, kaiming_init, ConvModule, build_upsample_layer
from vlkit.ops import deconv_upsample
from mmseg.models.decode_heads import ASPPHead

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
                 offset_in_channels,
                 kernel_size=1,
                 deform_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            offset_in_channels, deform_groups * offset_channels, 1, bias=False)
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
    def __init__(self, model_name, classes, setting=0, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(pretrained=False)
        self.base = model
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        # self.seg_head = ASPPHead(in_channels=256, channels=128, num_classes=1, dilations=[1,2,4,8])
        self.seg_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)

        self.edge_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.ori_head = nn.Conv2d(256, 8, kernel_size=1, bias=False)

        self.def_conv = DeformConv2d(
            256,
            256,
            kernel_size=1,
            padding=0,
            deform_groups=1)

        self.up_seg = deconv_upsample(channels=1, stride=8)
        self.up_ori = deconv_upsample(channels=8, stride=8)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        # normal_init(self.seg_head.conv_seg, std=0.01)
        normal_init(self.seg_head, std=0.01)
        normal_init(self.edge_head, std=0.01)
        normal_init(self.ori_head, std=0.01)

    def forward(self, x):
        x = self.conv1x1(self.base(x))
        seg = self.seg_head(x)
        edge = self.edge_head(x)
        orientation = self.ori_head(x)

        seg = self.up_seg(seg)
        edge = self.up_seg(edge)
        orientation = self.up_ori(orientation)

        results = dict({
            "seg": seg,
            "edge": edge,
            "orientation": orientation
        })

        return results


if __name__ == '__main__':
    model = DRNSeg("drn_d_22", classes=1)
    data = torch.rand(10, 3, 224, 224)
    y = model(data)