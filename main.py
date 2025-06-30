# main_trainer.py

import torch
import os

from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.trainer import Trainer, YOLOTrainer
from src.tester import Tester
from src.utils import print_device_info
from src.config import (
    DATASET_PATH,
    BATCH_SIZE,
    EPOCHS,
    FREEZE_BACKBONE,
    MODEL_TRAINED_PATH,
    MODEL_FILE,
    OUTPUT_PATH,
    OPTIMIZER,
    NAME_PATH
)


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


def resnet_tester():
    # Evaluation configuration
    DATASET_EVAL_PATH = "data/annotated/INF692_GEST_CLAS_MY.v3i.folder/"

    # Mapping of model keys to their checkpoint paths
    MODEL_PATHS = {
        "ft_ImageNet_to_generic":
            "models/checkpoints/resnet18/ft_ImageNet_to_generic-b-32-e-100-lr-1e-05-SGD-20250627_232559-97.69.pkl",
        "ft_ImageNet_to_personalized":
            "models/checkpoints/resnet18/ft_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250627_234356-100.00.pkl",
        "tl_generic_to_personalized":
            "models/checkpoints/resnet18/tl_generic_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_000503-95.17.pkl",
        "tl_generic_to_generic-personalized":
            "models/checkpoints/resnet18/tl_generic_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_003548-95.65.pkl",
        "tl_ImageNet_to_personalized":
            "models/checkpoints/resnet18/tl_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_220151-97.59.pkl",
        "tl_ImageNet_to_generic-personalized":
            "models/checkpoints/resnet18/tl_ImageNet_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_222757-89.53.pkl"
    }

    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    loader = DatasetLoader(DATASET_EVAL_PATH, BATCH_SIZE)
    data = loader.load()

    for key, path in MODEL_PATHS.items():
        print(f"\n🧪 Evaluating model: {key}")
        print(f"📍 Model path: {path}\n")

        model = prepare_model(
            num_classes=len(data["classes"]),
            freeze_backbone=1,
            model_trained_path=path,
            device=device
        )

        tester = Tester(model=model, data=data, device=device)

        results = tester.infer()

        output_dir = os.path.join("outputs", "eval_my", key)
        tester.save_results(results, output_dir=output_dir)

    print("\n✅ All models evaluated successfully!")
    print("📁 Results saved in 'outputs/eval/' directory.")


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


def main():
    print("Welcome to the SIBGRAPI 2025 Classifier!")
    print("Choose an option:")
    print("1. Train ResNet model")
    print("2. Test ResNet model")
    print("3. Train YOLO model")
    print("4. Test YOLO model")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        resnet_trainer()
    elif choice == "2":
        resnet_tester()
    elif choice == "3":
        yolo_trainer()
    elif choice == "4":
        yolo_tester()
    else:
        print("Invalid choice! Please select a valid option.")


if __name__ == "__main__":
    main()
