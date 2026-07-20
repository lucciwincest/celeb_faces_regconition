import torch.nn as nn
from config import *

class Backbone (nn.Module):
    def __init__ (self):
        super ().__init__ ()

        self.blocks = nn.ModuleList ([
            nn.Sequential (
                nn.Conv2d (IN_CHANNELS, 32, 3, 1, 1),
                nn.ReLU (),
                nn.MaxPool2d (2)
            ),
            nn.Sequential (
                nn.Conv2d (32, 64, 3, 1, 1),
                nn.ReLU (),
                nn.MaxPool2d (2)
            )
        ])



    