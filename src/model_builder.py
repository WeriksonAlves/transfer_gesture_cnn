# src/model_builder.py

from torchvision import models
import torch.nn as nn


def prepare_model(num_classes):
    """
    Creates a ResNet-18 model pre-trained on ImageNet, freezes all layers
    except the final fully connected layer, and replaces the final layer to
    match the specified number of output classes.

    Args:
        num_classes (int): The number of output classes for the classification
            task.

    Returns:
        torch.nn.Module: The modified ResNet-18 model ready for training on
            the specified number of classes.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Enable training only for the final layer
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    for name, param in model.named_parameters():
        param.requires_grad = (name.startswith("fc"))

    return model
