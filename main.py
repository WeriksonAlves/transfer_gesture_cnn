# main.py

import torch
from src.config import (
    DATASET_PATH,
    BATCH_SIZE,
    EPOCHS
)
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer
from src.tester import Tester
from src.utils import print_device_info


def main():
    # Display PyTorch version and device info
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # Load dataset
    loader = DatasetLoader(DATASET_PATH, BATCH_SIZE)
    data = loader.load()

    # Initialize model architecture
    model = prepare_model(num_classes=len(data["classes"]),
                          device=device)

    # Train the model
    trainer = Trainer(
        model=model,
        data=data,
        device=device
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
        device=device
    )

    predictions = tester.infer()
    tester.save_results(predictions)


if __name__ == "__main__":
    main()
