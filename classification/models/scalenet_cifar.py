import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class MSConv(nn.Module):

    def __init__(self, inplanes, out_planes, kernel_size=3, stride=1, padding=1, capacity=1):
        super(MSConv, self).__init__()

        self.capacity = capacity

        self.ch0 = out_planes[0]
        self.ch1 = out_planes[1]
        self.inplanes = inplanes

        self.conv0 = nn.Conv2d(inplanes, self.ch0, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)

        self.conv1 = nn.Conv2d(inplanes, self.ch1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d(sum(out_planes))
    
    def forward(self, x):

        x0 = x
        x1 = self.downsample(x)

        y0 = self.conv0(x0)
        _, _, h, w = y0.shape

        y1 = self.conv1(x1)        
        y1 = torch.nn.functional.interpolate(y1, size=[h, w], mode="bilinear")

        return torch.cat((y0, y1), dim=1)
    
    def allocate(self):

        temperature0 = self.bn.weight.data[:self.ch0]
        temperature1 = self.bn.weight.data[self.ch0::]
        
        temp_diff = (temperature0.mean() - temperature1.mean()).item()

        ch0_old = self.ch0
        ch1_old = self.ch1

        ch_transfer = abs(int(temp_diff * self.capacity * self.ch1))

        if ch_transfer == 0:
            ch_transfer = 1

        if temp_diff > 0:
            # conv0 is more important than conv1
            # conv1 -> conv0

            if self.ch1 - ch_transfer <= 0:
                return temperature0, temperature1

            ch0_old = self.ch0
            ch1_old = self.ch1

            self.ch0 = self.ch0 + ch_transfer
            self.ch1 = self.ch1 - ch_transfer

            v1, idx_retained1 = torch.sort(temperature1, descending=True)
            idx_retained1 = idx_retained1[:self.ch1]

            _, inplanes, kernel_size, _ = self.conv0.weight.data.shape
            padding = self.conv0.padding
            stride = self.conv0.stride

            conv1w_retained = self.conv1.weight.data[idx_retained1, :, :, :]
            self.conv1 = nn.Conv2d(inplanes, self.ch1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv1.weight.data = conv1w_retained

            conv0w_old = self.conv0.weight.data
            self.conv0 = nn.Conv2d(inplanes, self.ch0, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            conv0w = torch.zeros(ch_transfer, inplanes, kernel_size, kernel_size).cuda()
            nn.init.kaiming_normal_(conv0w, mode='fan_out', nonlinearity='relu')
            self.conv0.weight.data = torch.cat((conv0w_old, conv0w), dim=0)

        elif temp_diff < 0:
            # conv1 is more important than conv0
            # conv0 -> conv1

            if self.ch0 - ch_transfer <= 0:
                return temperature0, temperature1

            ch0_old = self.ch0
            ch1_old = self.ch1

            self.ch0 = self.ch0 - ch_transfer
            self.ch1 = self.ch1 + ch_transfer

            v0, idx_retained0 = torch.sort(temperature0, descending=True)
            idx_retained0 = idx_retained0[:self.ch0]

            _, inplanes, kernel_size, _ = self.conv0.weight.data.shape
            padding = self.conv0.padding
            stride = self.conv0.stride

            conv0w_retained = self.conv0.weight.data[idx_retained0, :, :, :]
            self.conv0 = nn.Conv2d(inplanes, self.ch0, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv0.weight.data = conv0w_retained

            conv1w_old = self.conv1.weight.data
            self.conv1 = nn.Conv2d(inplanes, self.ch1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            conv1w = torch.zeros(ch_transfer, inplanes, kernel_size, kernel_size).cuda()
            torch.nn.init.kaiming_normal_(conv1w, mode='fan_out', nonlinearity='relu')
            self.conv1.weight.data = torch.cat((conv1w_old, conv1w), dim=0)

        return temperature0, temperature1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = MSConv(inplanes, out_planes=[planes//2]*2, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # out = self.bn1(out)
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
        
        # self.conv2 = conv3x3(planes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = MSConv(planes, out_planes=[planes//2]*2, kernel_size=3, stride=stride)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.constant_(m.weight, 1)
                nn.init.uniform_(m.weight, a=0, b=1)
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def allocate(self):
        temperature = {}
        for name, m in self.named_modules():
            if isinstance(m, MSConv):
                temperature0, temperature1 = m.allocate()
                temperature[name+'-0'] = temperature0
                temperature[name+'-1'] = temperature1
        return temperature
                


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
