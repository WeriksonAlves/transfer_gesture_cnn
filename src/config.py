# src/config.py
"""
Configuration module for training and fine-tuning ResNet18-based image
classifiers.

This module defines experiment modes and their corresponding hyperparameters,
dataset paths, model checkpoints, and training options for various transfer
learning and fine-tuning scenarios.

Attributes:
    TRAIN_MODE (int): Selects the experiment mode (0 to 5).
    BATCH_SIZE (int): Number of samples per training batch.
    EPOCHS (int): Number of training epochs.
    LEARNING_RATE (float): Learning rate for the optimizer.
    OPTIMIZER (str): Optimizer type (e.g., "SGD").
    NAME_PATH (str): Identifier for the experiment, used in output paths.
    DATASET_PATH (str): Path to the dataset for the selected mode.
    FREEZE_BACKBONE (int): Number of layers to freeze in the backbone
        (model-specific).
    MODEL_TRAINED_PATH (str or ResNet18_Weights): Path or identifier for the
        pretrained model weights.
    PREFIX (str): Prefix for output and logging paths, encoding experiment
        parameters.
    TENSORBOARD_DIR (str): Directory for TensorBoard logs.
    MODEL_PATH (str): Path to save trained model checkpoints.

Experiment Modes:
    0: Fine-tuning from ImageNet to Generic dataset.
    1: Fine-tuning from ImageNet to Personalized dataset.
    2: Transfer learning from Generic to Personalized dataset.
    3: Transfer learning from Generic to Generic+Personalized dataset.
    4: Transfer learning from ImageNet to Personalized dataset.
    5: Transfer learning from ImageNet to Generic+Personalized dataset.

Raises:
    ValueError: If TRAIN_MODE is not in the range 0 to 5.
"""

from torchvision.models import ResNet18_Weights
from datetime import datetime

# Select experiment mode (0 to 5)
TRAIN_MODE = 0

# Default hyperparameters (can be overridden per mode)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-5

# Mode-specific configuration
if TRAIN_MODE == 0:  # Fine-tuning: ImageNet → Generic
    NAME_PATH = "ft_ImageNet_to_generic"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE.v3i.folder/"
    FREEZE_BACKBONE = 3
    OPTIMIZER = "SGD"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 1:  # Fine-tuning: ImageNet → Personalized
    NAME_PATH = "ft_ImageNet_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_MY.v3i.folder/"
    FREEZE_BACKBONE = 3
    OPTIMIZER = "SGD"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 2:  # Transfer learning: Generic → Personalized
    NAME_PATH = "tl_generic_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = (
        "models/resnet18/ft_ImageNet_to_generic-b-32-e-100-lr-1e-05-SGD-20250627_232559-97.69.pkl"
    )

elif TRAIN_MODE == 3:  # Transfer learning: Generic → Generic+Personalized
    NAME_PATH = "tl_generic_to_generic-personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = (
        "models/resnet18/ft_ImageNet_to_generic-b-32-e-100-lr-1e-05-SGD-20250627_232559-97.69.pkl"
    )

elif TRAIN_MODE == 4:  # Transfer learning: ImageNet → Personalized
    NAME_PATH = "tl_ImageNet_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 5:  # Transfer learning: ImageNet → Generic+Personalized
    NAME_PATH = "tl_ImageNet_to_generic-personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 6:  # Fine-tuning: yolov8n-pose → Generic
    NAME_PATH = "ft_yolov8n_pose_to_generic"
    DATASET_PATH = "data/annotated/INF692_GEST_POSE_GE.v1i.yolov8/data.yaml"
    FREEZE_BACKBONE = 2
    MODEL_TRAINED_PATH = "yolov8n-pose.pt"

elif TRAIN_MODE == 7:  # Fine-tuning: yolov8n-pose → Personalized
    NAME_PATH = "ft_yolov8n_pose_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_POSE_MY.v5i.yolov8/data.yaml"
    FREEZE_BACKBONE = 2
    MODEL_TRAINED_PATH = "yolov8n-pose.pt"

elif TRAIN_MODE == 8:  # Transfer learning: yolov8n-pose (generic) → Personalized
    NAME_PATH = "tl_yolov8n_pose_generic_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_POSE_MY.v5i.yolov8/data.yaml"
    FREEZE_BACKBONE = 1
    MODEL_TRAINED_PATH = "models/yolov8n-pose/ft_yolov8n_pose_to_generic-b-32-e-100-lr-1e-05-SGD-"

else:
    raise ValueError("Invalid TRAIN_MODE. Choose a value from 0 to 5.")

# Output and logging paths
PREFIX = (
    f"{NAME_PATH}-b-{BATCH_SIZE}-e-{EPOCHS}-lr-{LEARNING_RATE}-o-{OPTIMIZER}-f-{FREEZE_BACKBONE}"
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
TENSORBOARD_DIR = f"tensorboard/{PREFIX}_{timestamp}/"
MODEL_FILE = f"models/{PREFIX}_{timestamp}"
OUTPUT_PATH = f"outputs/{PREFIX}_{timestamp}/"
