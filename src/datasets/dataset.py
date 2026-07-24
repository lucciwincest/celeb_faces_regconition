from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from .transforms import get_transforms


def get_data_loader (dataset_root, batch_size, image_size):
    dataset_root = Path (dataset_root)
    num_workers = min (4, os.cpu_count () // 2)
    persistent_workers = (num_workers > 0)
    pin_memory =  True
    train_transform, val_transform, test_transform = get_transforms (image_size)

    train_dataset = ImageFolder (dataset_root / "train", train_transform)
    val_dataset = ImageFolder (dataset_root / "val", val_transform)
    test_dataset = ImageFolder (dataset_root / "test", test_transform)

    train_loader = DataLoader (train_dataset, batch_size = batch_size, num_workers = num_workers, persistent_workers = persistent_workers, pin_memory = pin_memory, drop_last = True)
    val_loader = DataLoader (val_dataset, batch_size = batch_size, num_workers = 4, pin_memory = True, persistent_workers = persistent_workers, pin_memory = pin_memory)
    test_loader = DataLoader (test_dataset, batch_size  = batch_size, num_workers = num_workers, persistent_workers = persistent_workers, pin_memory = pin_memory)

    return train_loader, val_loader, test_loader