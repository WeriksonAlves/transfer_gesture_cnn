# main_infer.py

import torch
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.tester import Tester
from src.utils import print_device_info


# Default hyperparameters (can be overridden per mode)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-5
OPTIMIZER = "SGD"
DATASET_EVAL_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"

MODEL_PATH = {
    "ft_ImageNet_to_generic": "models/resnet18/ft_ImageNet_to_generic-b-32-e-100-lr-1e-05-SGD-20250627_232559-97.69.pkl",
    "ft_ImageNet_to_personalized": "models/resnet18/ft_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250627_234356-100.00.pkl",
    "tl_generic_to_personalized": "models/resnet18/tl_generic_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_000503-95.17.pkl",
    "tl_generic_to_generic-personalized": "models/resnet18/tl_generic_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_003548-95.65.pkl",
    "tl_ImageNet_to_personalized": "models/resnet18/tl_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_220151-97.59.pkl",
    "tl_ImageNet_to_generic-personalized": "models/resnet18/tl_ImageNet_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_222757-89.53.pkl"
}


def evaluate_model_individual(device: torch.device, data):
    """
    Evaluate a model in a specific mode.

    Args:
        device (torch.device): The device to run the evaluation on.
    """
    for key, path in MODEL_PATH.items():
        print(f"\n🧪 Evaluating model: {key}...\n"
              f"Model path: {path}\n")

        # Initialize model architecture
        model = prepare_model(num_classes=len(data["classes"]),
                              freeze_backbone=1,
                              model_trained_path=path,
                              device=device)
        # Initialize tester
        tester = Tester(model=model,
                        data=data,
                        device=device)

        # Run inference
        results = tester.infer()

        # Save results
        output_dir = f"outputs/eval/{key}"
        tester.save_results(results, output_dir=output_dir)


def main():
    """
    Main function to evaluate models in all modes.
    """
    # Display PyTorch version and device info
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # Load dataset
    loader = DatasetLoader(DATASET_EVAL_PATH, BATCH_SIZE)
    data = loader.load()

    # Evaluate models in all modes
    evaluate_model_individual(device, data)
    print("\n✅ All models evaluated successfully!")
    print("Results saved in 'outputs/eval/' directory.")


if __name__ == "__main__":
    main()