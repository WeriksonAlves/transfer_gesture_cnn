# main_trainer.py

import torch
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer, YOLOTrainer
from src.tester import Tester
from src.utils import print_device_info

from torchvision.models import ResNet18_Weights
from datetime import datetime

# Select experiment mode (0 to 9)
TRAIN_MODE = 9

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
        "models/resnet18/ft/ft_ImageNet_to_generic-b-32-e-50-lr-1e-05-o-SGD-f-3_20250701_154138-97.22.pkl"
    )

elif TRAIN_MODE == 3:  # Transfer learning: Generic → Generic+Personalized
    NAME_PATH = "tl_generic_to_generic-personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = (
        "models/resnet18/ft/ft_ImageNet_to_generic-b-32-e-50-lr-1e-05-o-SGD-f-3_20250701_154138-97.22.pkl"
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

elif TRAIN_MODE == 6:  # Transfer learning: ImageNet → Personalized (SGD)
    NAME_PATH = "tl_ImageNet_to_personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "SGD"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 7:  # Transfer learning: ImageNet → Generic+Personalized (SGD)
    NAME_PATH = "tl_ImageNet_to_generic-personalized"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 1
    OPTIMIZER = "SGD"
    MODEL_TRAINED_PATH = ResNet18_Weights.IMAGENET1K_V1

elif TRAIN_MODE == 8:  # From scratch: Generic+Personalized (Adam)
    NAME_PATH = "from_scratch"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 0
    OPTIMIZER = "Adam"
    MODEL_TRAINED_PATH = None

elif TRAIN_MODE == 9:  # From scratch: Generic+Personalized (SGD)
    NAME_PATH = "from_scratch"
    DATASET_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
    FREEZE_BACKBONE = 0
    OPTIMIZER = "SGD"
    MODEL_TRAINED_PATH = None

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


def resnet_trainer():
    # Display PyTorch version and device info
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # Load dataset
    loader = DatasetLoader(DATASET_PATH, BATCH_SIZE)
    data = loader.load()

    # Initialize model architecture
    model = prepare_model(num_classes=len(data["classes"]),
                          freeze_backbone=FREEZE_BACKBONE,
                          model_trained_path=MODEL_TRAINED_PATH,
                          device=device)

    # Train the model
    trainer = Trainer(
        model=model,
        data=data,
        device=device,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        tensorboard_dir=TENSORBOARD_DIR,
        model_file=MODEL_FILE,
        criterion=torch.nn.CrossEntropyLoss()
    )

    best_model = trainer.train(
        epochs=EPOCHS,
        log_layers=True,
        save_best=True,
        upper_bound=100.0,
        debug=False
    )

    # Evaluate one random test sample
    tester = Tester(
        model=best_model,
        data=data,
        device=device,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
    )

    predictions = tester.infer()
    tester.save_results(predictions)


def yolo_trainer():
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    IMAGE_SIZE = 640

    print("PyTorch version:", torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_device_info(device)

    trainer = YOLOTrainer(
        model_path=MODEL_TRAINED_PATH,
        dataset_yaml=DATASET_PATH,
        output_dir=MODEL_FILE,
        experiment_name=NAME_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        img_size=IMAGE_SIZE,
        freeze=FREEZE_BACKBONE,
        optimizer=OPTIMIZER,
        lr=LEARNING_RATE,
        device=device
    )

    model, _ = trainer.train()
    trainer.save_model(model, OUTPUT_PATH)


if __name__ == "__main__":
    resnet_trainer()
    # yolo_trainer()  # Uncomment to run YOLO training
