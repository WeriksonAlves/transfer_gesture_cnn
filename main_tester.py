# main_tester.py

import os
import torch

from src.dataloader import DatasetLoader
from src.model_builder import prepare_model
from src.tester import Tester
from src.utils import print_device_info

# Evaluation configuration
DATASET_EVAL_PATH = "data/annotated/INF692_GEST_CLAS_GE-MY.v3i.folder/"
model_ft_paths = {
    "ft_ImageNet_to_generic":
        "models/resnet18/ft/ft_ImageNet_to_generic-b-32-e-50-lr-1e-05-o-SGD-f-3_20250701_154138-97.22.pkl",
    "tl_generic_to_generic-personalized":
        "models/resnet18/ft/tl_generic_to_generic-personalized-b-32-e-50-lr-1e-05-o-Adam-f-1_20250702_033550-96.64.pkl",
    "tl_generic_to_personalized":
        "models/resnet18/ft/tl_generic_to_personalized-b-32-e-50-lr-1e-05-o-Adam-f-1_20250702_031925-95.17.pkl",
    "ft_ImageNet_to_personalized":
        "models/resnet18/ft/ft_ImageNet_to_personalized-b-32-e-50-lr-1e-05-o-SGD-f-3_20250701_114714-100.00.pkl",
}
model_tl_paths = {
    "tl_ImageNet_to_generic-personalized_Adam":
        "models/resnet18/tl/tl_ImageNet_to_generic-personalized-b-32-e-50-lr-1e-05-o-Adam-f-1_20250702_013942-91.70.pkl",
    "tl_ImageNet_to_generic-personalized_SGD":
        "models/resnet18/tl/tl_ImageNet_to_generic-personalized-b-32-e-50-lr-1e-05-o-SGD-f-1_20250702_021758-85.38.pkl",
    "tl_ImageNet_to_personalized_Adam":
        "models/resnet18/tl/tl_ImageNet_to_personalized-b-32-e-50-lr-1e-05-o-Adam-f-1_20250702_012747-90.69.pkl",
    "tl_ImageNet_to_personalized_SGD":
        "models/resnet18/tl/tl_ImageNet_to_personalized-b-32-e-50-lr-1e-05-o-SGD-f-1_20250702_020825-95.17.pkl"
}
BATCH_SIZE = 32
MODEL_PATHS = model_ft_paths
OUTPUT_DIR = "outputs/resnet18/ft/models_comparation/"


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

        tester = Tester(
            model=model,
            data=data,
            device=device,
            batch_size=BATCH_SIZE,
            output_path=OUTPUT_DIR)

        results = tester.infer()

        output_dir = os.path.join(OUTPUT_DIR, key)
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
