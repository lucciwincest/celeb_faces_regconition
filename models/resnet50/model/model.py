from .backbone import Backbone
from .head import Head
from .neck import Neck
import torch.nn as nn

class Model (nn.Module):
    def __init__ (self, num_classes):
        super ().__init__ ()
        self.backbone  = Backbone ()
        self.neck = Neck ()
        self.head = Head (num_classes)

    def forward (self, X):
        X = self.backbone (X)
        X = self.neck (X)
        X = self.head (X)
        return X 