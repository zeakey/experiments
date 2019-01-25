import torch
import torchvision
import torch.nn as nn

class VGG16(nn.Module):

    def __init__(self, num_classes):
        
        self.vgg = torchvision.models.vgg16_bn(pretrained=True)

        self.classifier = nn.Linear(1000, num_classes, bias=False)

        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):

        x = self.vgg(x)
        x = classifier(x)

        return torch.softmaxx(x, dim=1)
        