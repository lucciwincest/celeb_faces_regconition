import torch.nn as nn

class Model (nn.Module):
    def __init__ (self, backbone, neck, head):
        super ().__init__ ()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward (self, X):
        X = self.backbone (X)
        X = self.neck (X)
        X = self.head (X)
        return X