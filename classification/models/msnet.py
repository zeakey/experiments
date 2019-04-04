import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import __all__, model_urls, conv3x3
from .resnet import BasicBlock, Bottleneck

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Downsample(in_channels, out_channels, stride):
    if stride == 2:
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    elif stride == 1:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        

class MSModule(nn.Module):

    def __init__(self, block_high, block_low, in_channels, high_channels, low_channels, stride=1, block_merge=None):
        """
        block_high: high resolution block, usually it contains more layers
        block_low: low resolution block, usually it contains less layers compared to block_high
        in_channels: channels of input tensor
        high_channels: output channels of high resolution block
        low_channels: output channels of low resolution block
        block_merge: a block used to merge high-resolution features and low-resolution features
        """
        super(MSModule, self).__init__()
        self.block_high = block_high
        self.block_low = block_low
        self.stride = stride

        assert stride == 1 or stride == 2

        if stride == 2:
            self.downsample = Downsample(in_channels, in_channels, 2)
        elif stride == 1:
            self.downsample = None
        else:
            raise ValueError("stride=%d" % stride)

        if high_channels != low_channels:
            self.match = nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False)
        else:
            self.match = None

        self.block_merge = block_merge

    def forward(self, x):

        if self.downsample:
            x = self.downsample(x)
        high = self.block_high(x)
        low = self.block_low(x)

        if low.size(2) != high.size(2):
            assert low.size(2) * 2 == high.size(2)
            assert low.size(3) * 2 == high.size(3)
            low = nn.functional.interpolate(low, scale_factor=2, mode="bilinear", align_corners=True)

        if self.match:
            high = self.match(high)

        if self.block_merge:
            merge = self.block_merge(high + low)
        else:
            merge = high + low


        return merge

class MSNet(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False):

        super(MSNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        block0a = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        block0b = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer0 = MSModule(block0a, block0b,
                        in_channels=64, low_channels=64, high_channels=64, block_merge=False)

        block_merge = None

        # layer1
        self.inplanes = 64
        planes = 64
        block_high = self._make_layer(block, self.inplanes, planes//2, blocks=1) # running at 56x56
        block_low = self._make_layer(block, self.inplanes, planes, blocks=2)
        block_low = nn.Sequential(
            Downsample(self.inplanes, self.inplanes, 2),
            block_low
        )
        block_merge = self._make_layer(block, 256, 64, blocks=1, stride=2)
        self.layer1 = MSModule(block_high, block_low,
                               in_channels=self.inplanes,
                               high_channels=128,
                               low_channels=256,
                               block_merge=block_merge,
                               stride=1)

        # layer2
        self.inplanes = planes*block.expansion
        planes = 128
        block_high = self._make_layer(block, self.inplanes, planes//2, blocks=1) # running at 28x28
        block_low = self._make_layer(block, self.inplanes, planes, blocks=3)
        block_low = nn.Sequential(
            Downsample(self.inplanes, self.inplanes, 2),
            block_low
        )
        block_merge = self._make_layer(block, 512, 128, blocks=1, stride=2)
        self.layer2 = MSModule(block_high, block_low,
                               in_channels=self.inplanes,
                               high_channels=256,
                               low_channels=512,
                               block_merge=block_merge,
                               stride=1)

        # layer3
        self.inplanes = planes*block.expansion
        planes = 256
        block_high = self._make_layer(block, self.inplanes, planes//2, blocks=2) # running at 14x14
        block_low = self._make_layer(block, self.inplanes, planes, blocks=4)
        block_low = nn.Sequential(
            Downsample(self.inplanes, self.inplanes, 2),
            block_low
        )
        self.layer3 = MSModule(block_high, block_low,
                               in_channels=self.inplanes,
                               high_channels=512,
                               low_channels=1024,
                               block_merge=None,
                               stride=1)
        
        # layer4
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
            downsample = Downsample(inplanes, planes*block.expansion, stride=stride)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
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
