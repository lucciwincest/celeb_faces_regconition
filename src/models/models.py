from .backbones import ResNet50Backbone
from .heads import LinearHead
from .necks import GAPNeck
from .model import Model

class ResNet50Model (Model):
    def __init__ (self, pretrained, num_classes):
        super ().__init__ (
            backbone = ResNet50Backbone (pretrained),
            neck = GAPNeck (),
            head = LinearHead (num_classes)
        )