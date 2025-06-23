# main.py

from src.config import *
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader = DatasetLoader(DATASET_PATH, BATCH_SIZE)
    data = loader.load()

    # Build model
    model = prepare_model(num_classes=len(data['classes']))
    model.to(device)

    # Train model
    trainer = Trainer(model, data, device, TENSORBOARD_DIR, MODEL_PATH)
    best_model = trainer.train(
        epochs=EPOCHS,
        log_layers=True,
        save_best=True
    )


if __name__ == '__main__':
    main()
