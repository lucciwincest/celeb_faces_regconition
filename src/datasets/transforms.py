from torchvision import transforms
from torchvision.models import ResNet50_Weights

def get_transforms (image_size):
    train_transform = transforms.Compose ([
        transforms.Resize (image_size),
        transforms.RandomHorizontalFlip (),
        transforms.RandomRotation (10),
        transforms.ToTensor (),
        transforms.Normalize (
            mean = ResNet50_Weights.IMAGENET1K_V2.mean,
            std = ResNet50_Weights.IMAGENET1K_V2.std
        )
    ])

    val_transform = transforms.Compose ([
        transforms.Resize (image_size),
        transforms.ToTensor (),
        transforms.Normalize (
            mean = ResNet50_Weights.IMAGENET1K_V2.mean,
            std = ResNet50_Weights.IMAGENET1K_V2.std
        )
    ])   

    test_transform = val_transform

    return train_transform, val_transform, test_transform


