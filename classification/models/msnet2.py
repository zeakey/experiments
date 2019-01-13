import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import __all__, model_urls, conv3x3, BasicBlock, Bottleneck

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample2(in_channels):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )

class MSModule(nn.Module):

    def __init__(self, block_high, block_low, block_merge=None, rescale=2):
        """
        block_high: high resolution block, usually it contains more layers
        block_low: low resolution block, usually it contains less layers compared to block_high
        block_merge: a block used to merge high-resolution features and low-resolution features
        """
        super(MSModule, self).__init__()
        self.block_high = block_high
        self.block_low = block_low
        self.block_merge = block_merge

        self.rescale = rescale
    
    def forward(self, x):
        high = self.block_high(x)
        low = self.block_low(x)

        if self.rescale != 1:
            low = nn.functional.interpolate(low, scale_factor=self.rescale, mode="bilinear", align_corners=True)

        if self.block_merge:
            merge = self.block_merge(high + low)
        else:
            merge = high + low

        return merge


class MSNet(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False, cifar=False):

        super(MSNet, self).__init__()
        
        self.inplanes = 64

        if cifar:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # layer1
        planes = 64
        block_high = self._make_layer(block, self.inplanes, planes, blocks=1, stride=1)
        block_low = self._make_layer(block, self.inplanes, planes, blocks=2, stride=1)
        block_merge = None
        self.layer1 = MSModule(block_high, block_low, block_merge, rescale=1)
        self.inplanes = planes*block.expansion

        # layer2
        planes = 128
        block_high = self._make_layer(block, self.inplanes, planes, blocks=1, stride=2)
        block_low = self._make_layer(block, self.inplanes, planes, blocks=3, stride=2)
        block_merge = None
        self.layer2 = MSModule(block_high, block_low, block_merge, rescale=1)
        self.inplanes = planes*block.expansion

        # layer3
        planes = 256
        block_high = self._make_layer(block, self.inplanes, planes, blocks=1, stride=2)
        block_low = self._make_layer(block, self.inplanes, planes, blocks=5, stride=2)
        block_merge = None
        self.layer3 = MSModule(block_high, block_low, block_merge, rescale=1)
        self.inplanes = planes*block.expansion

        self.layer4 = self._make_layer(block, planes*block.expansion, 512, blocks=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    conv1x1(inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                raise ValueError("stride=%d" % stride)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def msnet34(**kwargs):
    
    model = MSNet(block=BasicBlock, **kwargs)

    return model

def msnet50(**kwargs):
    
    model = MSNet(block=Bottleneck, **kwargs)

    return model
