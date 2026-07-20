from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .transforms import get_transforms
from config import *

def get_data_loader (dataset_root = DATASET_ROOT, batch_size = BATCH_SIZE, shuffle = False):
    dataset_root = Path (dataset_root)
    train_transform, val_transform, test_transform = get_transforms ()

    train_dataset = ImageFolder (dataset_root / "train", train_transform)
    val_dataset = ImageFolder (dataset_root / "val", val_transform)
    test_dataset = ImageFolder (dataset_root / "test", test_transform)

    train_loader = DataLoader (train_dataset, batch_size, shuffle, drop_last = True)
    val_loader = DataLoader (val_dataset, batch_size)
    test_loader = DataLoader (test_dataset, batch_size)

    return train_loader, val_loader, test_loader