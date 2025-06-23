# src/dataloader.py

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class DatasetLoader:
    """Handles dataset loading and transformations."""

    def __init__(self, data_dir, batch_size):
        """
        Initializes the data loader with the specified data directory and
        batch size.

        Args:
            data_dir (str): Path to the directory containing the dataset.
            batch_size (int): Number of samples per batch to load.

        Attributes:
            mean (list of float): Mean values for normalization (per channel).
            std (list of float): Standard deviation values for normalization
                (per channel).
            transform (torchvision.transforms.Compose): Composed image
                transformations including resizing, tensor conversion, and
                normalization.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def load(self):
        """
        Loads image datasets for training, validation, and testing from the
        specified data directory.

        Returns:
            dict: A dictionary containing:
                - 'train': DataLoader for the training set with shuffling
                           enabled.
                - 'valid': DataLoader for the validation set without shuffling.
                - 'test': DataLoader for the test set without shuffling.
                - 'classes': List of class names inferred from the training
                    set.

        The datasets are expected to be organized in 'train', 'valid', and
        'test' subdirectories within the data directory. Each subdirectory
        should contain one folder per class.
        """
        train_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'), transform=self.transform)
        val_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'valid'), transform=self.transform)
        test_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'), transform=self.transform)

        return {
            'train': DataLoader(train_set, batch_size=self.batch_size,
                                shuffle=True),
            'valid': DataLoader(val_set, batch_size=self.batch_size,
                                shuffle=False),
            'test': DataLoader(test_set, batch_size=self.batch_size,
                               shuffle=False),
            'classes': train_set.classes
        }
