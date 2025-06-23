# File: training_pipeline.py

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class DatasetLoader:
    """Handles dataset loading and transformations."""

    def __init__(self, data_dir, batch_size):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.data_dir = data_dir

    def load(self):
        train_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'), transform=self.transform)
        val_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'valid'), transform=self.transform)
        test_set = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'), transform=self.transform)

        return {
            'train': DataLoader(train_set, batch_size=self.batch_size,
                                shuffle=True),
            'valid': DataLoader(val_set, batch_size=self.batch_size,
                                shuffle=False),
            'test': DataLoader(test_set, batch_size=self.batch_size,
                               shuffle=False),
            'classes': train_set.classes
        }


class Trainer:
    """Manages the training process, logging and model saving."""

    def __init__(self, model, dataloaders, device, log_dir, model_path):
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_path = model_path
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

    def _log_layer_weights(self, epoch):
        for i, layer in enumerate(self.model.modules()):
            if isinstance(layer, nn.Linear):
                self.writer.add_histogram(f'Weight/layer{i+1}',
                                          layer.weight, epoch)

    def validate(self):
        self.model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for x, labels in self.dataloaders['valid']:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total

    def train(self, epochs, log_layers=False, save_best=True):
        best_acc = 0.0
        best_model = None
        total_batches = len(self.dataloaders['train'])

        for epoch in tqdm(range(epochs), desc='Training'):
            self.model.train()
            for i, (x, labels) in enumerate(self.dataloaders['train']):
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step = epoch * total_batches + i
                self.writer.add_scalar('Loss/train', loss.item(), step)
                acc = (outputs.argmax(1) == labels).float().mean().item()
                self.writer.add_scalar('Accuracy/train', acc, step)

            if log_layers:
                self._log_layer_weights(epoch)

            val_acc = self.validate()
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            if val_acc > best_acc:
                best_model = copy.deepcopy(self.model)
                best_acc = val_acc
                if save_best:
                    torch.save(best_model.state_dict(), self.model_path)

        self.writer.close()
        return best_model


class ModelTester:
    """Handles model inference and visualization of predictions."""

    def __init__(self, model, test_data, device, class_names):
        self.model = model.to(device)
        self.test_data = test_data
        self.device = device
        self.class_names = class_names
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def sample_and_predict(self, seed=None):
        """
        Randomly selects one test sample, predicts, and displays the result.

        Args:
            seed (int, optional): Seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)

        index = np.random.randint(len(self.test_data))
        sample, true_label = self.test_data[index]

        # Visualize image
        plt.figure(figsize=(2, 2))
        plt.axis('off')

        image_np = sample.numpy().transpose(1, 2, 0)
        image_np = np.clip(self.std * image_np + self.mean, 0, 1)
        plt.imshow(image_np)
        plt.title("Sample Image")
        plt.show()

        # Prepare for inference
        input_tensor = sample.unsqueeze(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output.squeeze(0), dim=0)
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()

        correct = predicted_class == true_label

        print(f"Sample Index: {index}")
        print("Prediction:", "Correct" if correct else "Incorrect")
        print(f"Predicted: {self.class_names[predicted_class]} | "
              f"True: {self.class_names[true_label]} | "
              f"Confidence: {confidence * 100:.2f}%")

        return predicted_class, true_label, confidence


def show_tensor_image(tensor_img, label=None, mean=None, std=None):
    """Displays a normalized tensor image."""
    img = tensor_img.numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    if label is not None:
        plt.title(f'Label: {label}')
    plt.show()


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def prepare_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    dataset_path = 'datasets/annotated/INF692_GEST_CLAS_GE.v3i.folder/'
    tensorboard_dir = 'Tensorboard/YOLO-TL/'
    model_path = 'models/MyYolo-INF692_GEST_CLAS_GE.pkl'
    batch_size = 32
    epochs = 100

    device = get_device()

    data_loader = DatasetLoader(dataset_path, batch_size)
    loaders = data_loader.load()
    model = prepare_model(num_classes=len(loaders['classes']))
    model.to(device)

    trainer = Trainer(
        model, loaders, device, tensorboard_dir, model_path
    )
    best_model = trainer.train(epochs=epochs, log_layers=True)

    # Run inference on a sample image
    tester = ModelTester(
        model=best_model,
        test_data=loaders['test'].dataset,
        device=device,
        class_names=loaders['classes']
    )
    tester.sample_and_predict(seed=42)  # fixed seed for reproducibility


if __name__ == '__main__':
    main()
