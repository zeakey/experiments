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

        y = self.bn(torch.cat((y0, y1), dim=1))

        return y


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

    def allocate(self, threshold=0.7, logger=None):
        factors = self.conv2.bn.weight.data.clone().abs()
        factors0 = factors[:self.conv2.ch0]
        factors1 = factors[self.conv2.ch0::]

        # a channel is useful if its factor value is greater
        # than (or equal) 0.01 * maximal-factor
        useful_mask = factors >= (factors.max() * 0.01)
        useful_mask0 = (factors0 >= (factors0.max() * 0.01)).float()
        useful_mask1 = (factors1 >= (factors1.max() * 0.01)).float()

        if logger:
            logger.info("useless factors: %s" % str(factors[1-useful_mask].tolist()))

        # utilization ratio
        uratio = (factors >= (factors.max() * 0.01)).float().mean()
        uratio0 = useful_mask0.mean()
        uratio1 = useful_mask1.mean()

        if logger:
            logger.info("ch0: %d, ch1: %d" % (self.conv2.ch0, self.conv2.ch1))
            logger.info("conv0 useful: %d, useless: %d" % (
                useful_mask0.sum(),
                (1 - useful_mask0).sum()
                ))
            logger.info("conv1 useful: %d, useless: %d" % (
                useful_mask1.sum(),
                (1-useful_mask1).sum()
                ))
            logger.info("factors0: %s" % str(factors0.tolist()))
            logger.info("factors1: %s" % str(factors1.tolist()))
            logger.info("uratio0: %f, uratio1: %f" % (uratio0, uratio1))

        # channel allocation is triggered if eight uratio0
        # or uratio1 is less than  threshold
        triggered = uratio0 <= threshold or uratio1 <= threshold

        if not triggered:
            return factors0, factors1, False

        ch0 = self.conv2.ch0
        ch1 = self.conv2.ch1

        ch_transfer = -1
        if uratio0 > uratio1:
            # a: number of useless neurons in conv1
            # b: total number of neurons in conv1
            # c: number of useless neurons in conv0
            # d: total number of neurons in conv0
            a = (1 - useful_mask1).sum()
            b = useful_mask1.numel()
            c = (1 - useful_mask0).sum()
            d = useful_mask0.numel()

            # \frac{a-x}{b} = \frac{c+x}{d}
            # x = \frac{ad - cb}{b + d}
            ch_transfer = (a*d - c*b) / (b + d)

            if ch_transfer > a:
                ch_transfer = a

        elif uratio1 > uratio0:
            a = (1 - useful_mask0).sum()
            b = useful_mask0.numel()
            c = (1 - useful_mask1).sum()
            d = useful_mask1.numel()
            ch_transfer = (a*d - c*b) / (b + d)
            
            if ch_transfer > a:
                ch_transfer = a

        ch_transfer = int(ch_transfer)
        transfer_mask = torch.zeros_like(factors).byte()
        order_new = []

        if logger:
            logger.info("ch_transfer: %d" % ch_transfer)

        if ch_transfer <= 0:
            return factors0, factors1, False

        if uratio0 >= uratio1:
            # conv0 is more important than conv1
            # conv1 -> conv0
            conv_weight_old = torch.cat((self.conv2.conv0.weight.data,
                                         self.conv2.conv1.weight.data), dim=0)

            v1, idx1_sorted = torch.sort(factors1, descending=True)
            idx1_sorted = (idx1_sorted + ch0).tolist()
            idx1_removed = idx1_sorted[-ch_transfer::]

            order0 = list(range(ch0))
            order1 = list(range(ch0, ch0+ch1))

            for i in idx1_removed:
                order0.append(i)
                order1.remove(i)

            order_new = order0 + order1

            _, inplanes, kernel_size, _ = self.conv2.conv0.weight.data.shape

            conv_weight_new = conv_weight_old[order_new, :, :, :]

            padding = self.conv2.conv0.padding
            stride = self.conv2.conv0.stride

            self.conv2.conv0 = nn.Conv2d(inplanes, ch0+ch_transfer, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv2.conv0.weight.data = conv_weight_new[:ch0+ch_transfer, :, :, :]
            # reinitiate conv parameters
            torch.nn.init.kaiming_normal_(self.conv2.conv0.weight.data[-ch_transfer::], mode='fan_out', nonlinearity='relu')

            self.conv2.conv1 = nn.Conv2d(inplanes, ch1-ch_transfer, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv2.conv1.weight.data = conv_weight_new[ch0+ch_transfer::, :, :, :]

            transfer_mask[ch0:ch0+ch_transfer] = 1

            # update ch0 and ch1 at last
            self.conv2.ch0 = self.conv2.ch0 + ch_transfer
            self.conv2.ch1 = self.conv2.ch1 - ch_transfer

        elif uratio0 < uratio1:
            # conv1 is more important than conv1
            # conv0 -> conv1
            conv_weight_old = torch.cat((self.conv2.conv0.weight.data,
                                         self.conv2.conv1.weight.data), dim=0)

            v0, idx0_sorted = torch.sort(factors0, descending=True)
            idx0_sorted = idx0_sorted.tolist()
            idx0_removed = idx0_sorted[-ch_transfer::]

            order0 = list(range(ch0))
            order1 = list(range(ch0, ch0+ch1))

            for i in idx0_removed:
                order0.remove(i)
                order1.append(i)

            order_new = order0 + order1

            conv_weight_new = conv_weight_old[order_new]

            _, inplanes, kernel_size, _ = self.conv2.conv0.weight.data.shape
            padding = self.conv2.conv0.padding
            stride = self.conv2.conv0.stride

            self.conv2.conv0 = nn.Conv2d(inplanes, ch0-ch_transfer, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv2.conv0.weight.data = conv_weight_new[:ch0-ch_transfer, :, :, :]

            self.conv2.conv1 = nn.Conv2d(inplanes, ch1+ch_transfer, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False).cuda()
            self.conv2.conv1.weight.data = conv_weight_new[ch0-ch_transfer::, :, :, :]
            # reinitiate conv parameters
            torch.nn.init.kaiming_normal_(self.conv2.conv1.weight.data[-ch_transfer::], mode='fan_out', nonlinearity='relu')

            transfer_mask[-ch_transfer::] = 1

            # update ch0 and ch1 at last
            self.conv2.ch0 = self.conv2.ch0 - ch_transfer
            self.conv2.ch1 = self.conv2.ch1 + ch_transfer

        #################################################
        # NOTE: adjust other parameters order accordingly
        #################################################
        self.conv2.bn.weight.data = self.conv2.bn.weight.data[order_new]
        self.conv2.bn.bias.data = self.conv2.bn.bias.data[order_new]
        self.conv3.weight.data = self.conv3.weight.data[:, order_new, :, :]

        # reinitiate parameters
        self.conv2.bn.weight.data[transfer_mask] = 1
        self.conv2.bn.bias.data[transfer_mask] = 0
        self.conv3.weight.data[:, transfer_mask, :, :] = 0

        useful_mask = useful_mask[order_new]
        self.conv2.bn.weight.data[1-useful_mask] = 1
        self.conv2.bn.bias.data[1-useful_mask] = 0
        self.conv3.weight.data[:, 1-useful_mask, :, :] = 0

        # rescale bn factors to 1
        if True:
            factors = self.conv2.bn.weight.data.clone()
            sign = torch.sign(factors)
            factors = factors.abs()
            r = torch.ones_like(factors)
            idx = (factors < 1) * (factors > 0)
            r[idx] = factors[idx]
            r = r.view(1, r.numel(), 1, 1)
            r = r.expand_as(self.conv3.weight.data)
            factors[idx] = sign[idx]
            self.conv2.bn.weight.data = factors
            self.conv3.weight.data *= r

        factors = self.conv2.bn.weight.data.clone().abs()
        factors0 = factors[:self.conv2.ch0]
        factors1 = factors[self.conv2.ch0::]

        if logger:
            logger.info("factors0 (updated): %s" % str(factors0.tolist()))
            logger.info("factors1 (updated): %s" % str(factors1.tolist()))
            logger.info("uratio0 (updated): %f, uratio1 (updated): %f" % (
                (factors0 >= (factors0.max() * 0.01)).float().mean(),
                (factors1 >= (factors1.max() * 0.01)).float().mean()
            ))

        return factors0, factors1, True


class ResNet(nn.Module):

    def __init__(self, block, layers, logger=None, num_classes=1000, zero_init_residual=False):
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

        self.logger = logger

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
    
    def allocate(self, threshold=0.7):
        results = {}
        for name, m in self.named_modules():
            if isinstance(m, Bottleneck):
                if self.logger:
                    self.logger.info("=====> allocate layer %s. <=====" % name)
                factors0, factors1, triggered = m.allocate(threshold=threshold, logger=self.logger)
                results[name] = (factors0, factors1, triggered)

        return results
                


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
