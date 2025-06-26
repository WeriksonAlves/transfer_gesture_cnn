# main.py

from src.config import (
    DATASET_PATH,
    MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    PREFIX,
    MODEL_TL_PATH
)
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer
from src.tester import ModelTester
from src.utils import print_device_info
import torch


def main():
    print("Versão do PyTorch:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # Load data
    loader = DatasetLoader(DATASET_PATH, BATCH_SIZE)
    data = loader.load()

    # Build model
    model = prepare_model(num_classes=len(data['classes']))

    # Load weights from GE model (fine-tuning base)
    if MODEL_TL_PATH is not None:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Loaded base model trained on GE for fine-tuning.")

    # Move model to device
    model.to(device)

    # Train model
    trainer = Trainer(model, data, device, PREFIX, MODEL_PATH)
    best_model = trainer.train(
        epochs=EPOCHS,
        log_layers=True,
        save_best=True
    )

    # Run inference on a sample image
    tester = ModelTester(
        model=best_model,
        test_data=data['test'].dataset,
        device=device,
        class_names=data['classes']
    )
    tester.sample_and_predict(seed=42)  # fixed seed for reproducibility


if __name__ == '__main__':
    main()
