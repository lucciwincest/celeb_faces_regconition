from torchvision import transforms
from torchvision.models import ResNet50_Weights
from config import *

def get_transforms ():
    train_transform = transforms.Compose ([
        transforms.Resize (IMAGE_SIZE),
        transforms.RandomHorizontalFlip (),
        transforms.RandomRotation (10),
        transforms.ToTensor (),
        transforms.Normalize (
            mean = ResNet50_Weights.IMAGENET1K_V2.transforms ().mean,
            std = ResNet50_Weights.IMAGENET1K_V2.transforms ().std
        )
    ])

    val_transform = transforms.Compose ([
        transforms.Resize (IMAGE_SIZE),
        transforms.ToTensor (),
        transforms.Normalize (
            mean = ResNet50_Weights.IMAGENET1K_V2.transforms ().mean,
            std = ResNet50_Weights.IMAGENET1K_V2.transforms ().std
        )
    ])   

    test_transform = val_transform

    return train_transform, val_transform, test_transform


