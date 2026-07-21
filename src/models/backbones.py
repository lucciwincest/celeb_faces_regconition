import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights

class SimpleBackbone (nn.Module):
    def __init__ (self, in_channels):
        super ().__init__ ()

        self.blocks = nn.ModuleList ([
            nn.Sequential (
                nn.Conv2d (in_channels, 32, 3, 1, 1),
                nn.ReLU (),
                nn.MaxPool2d (2)
            ),
            nn.Sequential (
                nn.Conv2d (32, 64, 3, 1, 1),
                nn.ReLU (),
                nn.MaxPool2d (2)
            )
        ])

    def forward (self, X):
        for block in self.blocks:
            X = block (X)
        return X


class ResNet50Backbone (nn.Module):
    def __init__ (self,  pretrained = True):
        super ().__init__ ()
        net = resnet50 (weights = ResNet50_Weights.DEFAULT if pretrained else None)
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