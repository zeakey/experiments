import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import __all__, model_urls, conv3x3

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MSModule(nn.Module):

    def __init__(self, block, blocks, inplanes, planes, stride=[2, 1, 2]):
        super(MSModule, self).__init__()
        self.inplanes = inplanes
        self.stride = stride

        self.stream0 = self._make_layer(block, inplanes[0], planes[0], blocks[0], stride=stride[0])
        self.stream1 = self._make_layer(block, inplanes[1], planes[1], blocks[1], stride=stride[1])

        self.match = None

        if planes[0] != planes[1]:
            self.match = conv1x1(min(planes) * block.expansion, max(planes) * block.expansion)

        self.inplanes = max(planes) * block.expansion

        if stride[2] == 2:
            self.res = BasicBlock(self.inplanes, self.inplanes, stride=stride[2],
                        downsample=nn.Sequential(
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            conv1x1(self.inplanes, self.inplanes),
                            nn.BatchNorm2d(self.inplanes)
                        ))
        elif stride[2] == 1:
            self.res = BasicBlock(self.inplanes, self.inplanes, stride=stride[2])
        else:
            raise ValueError("stride = %d" % stride[2])

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            if stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2,stride=stride, ceil_mode=True),
                    conv1x1(inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif stride == 1:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                raise ValueError("stride=%d" % stride)


        layers = []
        layers.append(block(inplanes, planes, stride, downsample))

        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        stream0 = self.stream0(x)
        stream1 = self.stream1(x)
        if self.stride[0] != self.stride[1]:
            assert (self.stride[0] / self.stride[1]) % 2 == 0
            stream0 = nn.functional.interpolate(stream0, scale_factor=2, mode="bilinear", align_corners=True)

        return self.res(stream0 + stream1)

class MSNet34(nn.Module):

    def __init__(self, block=BasicBlock, num_classes=1000):

        super(MSNet34, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2a = conv3x3(64, 64, stride=2)

        self.conv2b1 = conv3x3(64, 32, stride=1)
        self.conv2b2 = conv3x3(32, 32, stride=2)
        self.conv2b3 = conv1x1(32, 64, stride=1)

        self.module1 = MSModule(block=block, blocks=[2, 1], inplanes = [64, 64], planes=[64, 64], stride=[2, 1, 2])
        self.module2 = MSModule(block=block, blocks=[3, 1], inplanes = [64, 64], planes = [128, 128], stride=[2, 1, 2])
        self.module3 = MSModule(block=block, blocks=[4, 2], inplanes = [128, 128], planes = [256, 256], stride=[2, 1, 1])

        self.inplanes = 256
        self.module4 = self._make_layer(block=block, planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2,stride=stride, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif stride == 1:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                raise ValueError("stride=%d" % stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        return nn.Sequential(*layers)

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x2a = self.conv2a(x)
        x2b = self.conv2b1(x)
        x2b = self.conv2b2(x2b)
        x2b = self.conv2b3(x2b)
        x = x2a + x2b

        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class MSNet50(nn.Module):

    def __init__(self, block=Bottleneck, num_classes=1000):

        super(MSNet50, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2a = conv3x3(64, 64, stride=2)
        
        self.conv2b1 = conv3x3(64, 32, stride=1)
        self.conv2b2 = conv3x3(32, 32, stride=2)
        self.conv2b3 = conv1x1(32, 64, stride=1)

        self.module1 = MSModule(block=block, blocks=[2, 1], inplanes = [64, 64], planes=[64, 64], stride=[2, 1, 2])
        self.module2 = MSModule(block=block, blocks=[3, 1], inplanes = [256, 256], planes = [128, 128], stride=[2, 1, 2])
        self.module3 = MSModule(block=block, blocks=[4, 2], inplanes = [512, 512], planes = [256, 256], stride=[2, 1, 1])
        
        self.inplanes = 1024
        self.module4 = self._make_layer(block=block, planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2,stride=stride, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif stride == 1:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                raise ValueError("stride=%d" % stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x2a = self.conv2a(x)
        x2b = self.conv2b1(x)
        x2b = self.conv2b2(x2b)
        x2b = self.conv2b3(x2b)
        x = x2a + x2b

        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x