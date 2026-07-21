import torch.nn as nn

class GAPNeck (nn.Module):
    def __init__ (self):
        super ().__init__ ()
        self.net = nn.Sequential (
            nn.AdaptiveAvgPool2d (1),
            nn.Flatten ()
        )
    def forward (self, X):
        return self.net (X)