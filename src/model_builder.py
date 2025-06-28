# src/model_builder.py

import torch
import torch.nn as nn
from torchvision import models
from src.config import FREEZE_BACKBONE, MODEL_TRAINED_PATH


def freeze_resnet_layers(model: nn.Module, mode: int) -> None:
    """
    Freezes specific layers of the ResNet model according to the given mode.

    Args:
        model (nn.Module): The ResNet model to modify.
        mode (int): Freeze configuration.
            0 - No layers frozen (all trainable)
            1 - Freeze all except final fully connected layer (fc)
            2 - Freeze all except fc and layer4
            3 - Freeze all except fc, layer4, and layer3
    """
    for name, param in model.named_parameters():
        if mode == 1:
            param.requires_grad = name.startswith("fc")
        elif mode == 2:
            param.requires_grad = name.startswith("fc") or "layer4" in name
        elif mode == 3:
            param.requires_grad = (
                name.startswith("fc")
                or "layer4" in name
                or "layer3" in name
            )
        else:  # mode == 0 or invalid: all layers trainable
            param.requires_grad = True


def prepare_model(
    num_classes: int, device: torch.device, debug: bool = False
) -> nn.Module:
    """
    Prepares a ResNet-18 model with custom output layer, and loads weights
    from ImageNet or from a local .pkl file (if given).

    Args:
        num_classes (int): Number of output classes for classification.
        debug (bool): If True, prints the model architecture and layer freeze
            status.

    Returns:
        nn.Module: A ResNet-18 model ready for training.
    """
    # Load ResNet-18 base model
    if isinstance(
        MODEL_TRAINED_PATH, models.ResNet18_Weights
    ) or MODEL_TRAINED_PATH is None:
        model = models.resnet18(weights=MODEL_TRAINED_PATH)

        # Replace final classification layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.resnet18(weights=None)
        state_dict = torch.load(MODEL_TRAINED_PATH, map_location=device)

        # Replace final classification layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)

    # Apply layer freezing configuration
    freeze_resnet_layers(model, FREEZE_BACKBONE)

    # Optional: print layer status for debugging
    if debug:
        print("Model architecture with layer freeze status:")
        print(model)
        for name, param in model.named_parameters():
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")

    return model
