import torch.nn as nn
from .backbone import Backbone
from .neck import Neck
from .head import Head
from config import *

class Model (nn.Module):
    def __init__ (self, num_classes = NUM_CLASSES):
        super ().__init__ ()
        self.backbone = Backbone ()
        self.neck = Neck ()
        self.head = Head (num_classes)

    def forward (self, X):
        X = self.backbone (X)
        X = self.neck (X)
        X = self.head (X)
        return X