"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T
from pathlib import Path

def create_dataloaders(train_dir: str | Path,
                       test_dir: str | Path,
                       train_transform: T.Compose,
                       test_transform: T.Compose,
                       batch_size: int=32,
                       num_workers: int=0,
                       pin_memory: bool=False):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """

    # Use ImageFolder to create dataset(s)
    train_data = ImageFolder(train_dir, transform=train_transform)
    test_data = ImageFolder(test_dir, transform=test_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_data, batch_size, False, num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, test_dataloader, class_names
