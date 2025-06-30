# yolo_trainer.py

import torch

from src.trainer import YOLOTrainer
from src.utils import print_device_info
from src.config import (
    BATCH_SIZE,
    OPTIMIZER,
    NAME_PATH,
    DATASET_PATH,
    FREEZE_BACKBONE,
    MODEL_TRAINED_PATH,
    MODEL_FILE,
    OUTPUT_PATH
)
EPOCHS = 3
LEARNING_RATE = 1e-4
IMAGE_SIZE = 640


def main():
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
    main()
