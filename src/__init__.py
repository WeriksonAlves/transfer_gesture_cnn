"""
Initialization module for the SIBGRAPI2025 Classifier package.

This module centralizes and re-exports core components, configuration
constants, and utilities for simplified and standardized access.

Exports:
    - Classes:
        DatasetLoader: Handles image dataset loading and preprocessing.
        Tester: Evaluation logic for trained models.
        Trainer: Training loop and logging logic.
        YOLOTrainer: Specialized trainer for YOLO models.

    - Functions:
        prepare_model: Builds and initializes a ResNet-based classifier.
        print_device_info: Displays CUDA/CPU hardware info.
"""


from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer, YOLOTrainer
from src.tester import Tester
from src.utils import print_device_info

__all__ = [
    "DatasetLoader",
    "prepare_model",
    "Trainer",
    "YOLOTrainer",
    "Tester",
    "print_device_info"
]
