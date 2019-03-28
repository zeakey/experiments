import torch
import torch.nn as nn
import resnet

class AttrModel(nn.Module):
    
    def __init__(self, extract_feature=False):
        
        super(AttrModel, self).__init__()
        self.extract_feature = extract_feature

        m = resnet.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(m.children())[:-1])
        
        # gender
        self.fc0 = nn.Linear(512, 2, bias=False)
        # race
        self.fc1 = nn.Linear(512, 5, bias=False)

    def forward(self, x):
        
        feature = self.base(x)
        feature = feature.view(feature.size(0), feature.size(1))

        if self.extract_feature:
            return feature

        gender = self.fc0(feature)
        race = self.fc1(feature)

        return gender, race
