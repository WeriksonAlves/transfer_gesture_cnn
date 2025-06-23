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
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
