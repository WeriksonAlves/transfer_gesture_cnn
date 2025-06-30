# main_tester.py

import os
import torch

from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.tester import Tester
from src.utils import print_device_info

# Evaluation configuration
BATCH_SIZE = 32
DATASET_EVAL_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"

# Mapping of model keys to their checkpoint paths
MODEL_PATHS = {
    "ft_ImageNet_to_generic":
        "models/resnet18/ft_ImageNet_to_generic-b-32-e-100-lr-1e-05-SGD-20250627_232559-97.69.pkl",
    "ft_ImageNet_to_personalized":
        "models/resnet18/ft_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250627_234356-100.00.pkl",
    "tl_generic_to_personalized":
        "models/resnet18/tl_generic_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_000503-95.17.pkl",
    "tl_generic_to_generic-personalized":
        "models/resnet18/tl_generic_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_003548-95.65.pkl",
    "tl_ImageNet_to_personalized":
        "models/resnet18/tl_ImageNet_to_personalized-b-32-e-100-lr-1e-05-SGD-20250628_220151-97.59.pkl",
    "tl_ImageNet_to_generic-personalized":
        "models/resnet18/tl_ImageNet_to_generic-personalized-b-32-e-100-lr-1e-05-SGD-20250628_222757-89.53.pkl"
}


def evaluate_all_models(device: torch.device, data: dict) -> None:
    """
    Evaluates all trained models listed in MODEL_PATHS on a shared test set.

    Args:
        device (torch.device): The device to run inference on.
        data (dict): Dictionary with DataLoaders and class names.
    """
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

        output_dir = os.path.join("outputs", "eval", key)
        tester.save_results(results, output_dir=output_dir)


def main() -> None:
    """
    Main function for evaluating multiple ResNet18 models on a shared test set.
    """
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    loader = DatasetLoader(DATASET_EVAL_PATH, BATCH_SIZE)
    data = loader.load()

    evaluate_all_models(device=device, data=data)

    print("\n✅ All models evaluated successfully!")
    print("📁 Results saved in 'outputs/eval/' directory.")


if __name__ == "__main__":
    main()
