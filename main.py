# main.py

import torch
from src.config import (
    DATASET_PATH,
    MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    PREFIX
)
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer
from src.tester import ModelTester
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
    model = prepare_model(num_classes=len(data["classes"]))

    # Train the model
    trainer = Trainer(
        model=model,
        data=data,
        device=device,
        prefix=PREFIX,
        model_path=MODEL_PATH
    )

    best_model = trainer.train(
        epochs=EPOCHS,
        log_layers=True,
        save_best=True,
        upper_bound=100.0,
        debug=False
    )

    # Evaluate one random test sample
    tester = ModelTester(
        model=best_model,
        test_data=data["test"].dataset,
        device=device,
        class_names=data["classes"]
    )

    tester.sample_and_predict(seed=42)  # Fixed seed for reproducibility


if __name__ == "__main__":
    main()
