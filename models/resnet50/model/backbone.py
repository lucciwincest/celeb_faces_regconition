import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Backbone (nn.Module):
    def __init__ (self):
        super ().__init__ ()
        net = resnet50 (weights = ResNet50_Weights.DEFAULT)
        self.blocks = nn.ModuleList ([
            nn.Sequential (
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool
            ),
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4
        ])

    def forward (self, X):
        for block in self.blocks:
            X = block (X)
        return X