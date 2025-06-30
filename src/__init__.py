"""
Initialization module for the SIBGRAPI2025 Classifier package.

This module centralizes and re-exports core components, configuration
constants, and utilities for simplified and standardized access.

Exports:
    - Classes:
        DatasetLoader: Handles image dataset loading and preprocessing.
        Tester: Evaluation logic for trained models.
        Trainer: Training loop and logging logic.

    - Functions:
        prepare_model: Builds and initializes a ResNet-based classifier.
        print_device_info: Displays CUDA/CPU hardware info.

    - Constants (from src.config):
        BATCH_SIZE: Batch size for training/inference.
        EPOCHS: Number of training epochs.
        LEARNING_RATE: Optimizer learning rate.
        OPTIMIZER: Optimizer identifier (e.g., "SGD", "Adam").
        NAME_PATH: String identifier for model/run.
        DATASET_PATH: Input dataset root path.
        FREEZE_BACKBONE: Integer indicating how much of the model is frozen.
        MODEL_TRAINED_PATH: Path to pre-trained or fine-tuned model.
        TENSORBOARD_DIR: Directory for TensorBoard logs.
        MODEL_FILE: Filename for saving final model weights.
        OUTPUT_PATH: Directory to save metrics, plots, and CSV results.
"""

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OPTIMIZER,
    NAME_PATH,
    DATASET_PATH,
    FREEZE_BACKBONE,
    MODEL_TRAINED_PATH,
    TENSORBOARD_DIR,
    MODEL_FILE,
    OUTPUT_PATH
)

from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer
from src.tester import Tester
from src.utils import print_device_info

__all__ = [
    "DatasetLoader",
    "prepare_model",
    "Trainer",
    "Tester",
    "print_device_info",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "OPTIMIZER",
    "NAME_PATH",
    "DATASET_PATH",
    "FREEZE_BACKBONE",
    "MODEL_TRAINED_PATH",
    "TENSORBOARD_DIR",
    "MODEL_FILE",
    "OUTPUT_PATH"
]
