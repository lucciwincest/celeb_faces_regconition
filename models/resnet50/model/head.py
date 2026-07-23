import torch.nn as nn

class Head (nn.Module):
    def __init__  (self, num_classes):
        super ().__init__ ()
        self.net = nn.LazyLinear (num_classes)

    def forward (self, X):
        return self.net (X)