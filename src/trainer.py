# src/trainer.py

import copy
import time
import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import (
    LEARNING_RATE,
    TENSORBOARD_DIR,
    OPTIMIZER,
    MODEL_FILE
)


class Trainer:
    """
    Trainer class for managing model training, validation, and logging.

    Attributes:
        model (nn.Module): Model to be trained.
        device (torch.device): Device used (CPU or CUDA).
        train_loader (DataLoader): Training data.
        valid_loader (DataLoader): Validation data.
        classes (list): List of class names.
        criterion (Loss): Loss function used (CrossEntropyLoss).
        optimizer (Optimizer): SGD or Adam optimizer.
        scheduler (LR Scheduler): StepLR scheduler.
        writer (SummaryWriter): TensorBoard writer.
    """

    def __init__(self, model, data, device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = data["train"]
        self.valid_loader = data["valid"]
        self.classes = data["classes"]

        self.writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
        self.criterion = nn.CrossEntropyLoss()

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(params, lr=LEARNING_RATE)
        elif OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

        self.scheduler = StepLR(self.optimizer, step_size=25, gamma=0.5)

    def _log_layers(self, epoch: int) -> None:
        """
        Logs the weights of linear layers for analysis in TensorBoard.
        """
        for i, layer in enumerate(self.model.modules()):
            if isinstance(layer, nn.Linear):
                self.writer.add_histogram(
                    tag=f"Weight/layer{i}",
                    values=layer.weight,
                    global_step=epoch
                )

    def validate(self) -> tuple:
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: Average validation loss and accuracy.
        """
        self.model.eval()
        total, correct, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for x, labels in self.valid_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                val_loss += loss.item() * x.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(
        self,
        epochs: int,
        log_layers: bool = False,
        save_best: bool = True,
        upper_bound: float = 100.0,
        debug: bool = False
    ) -> nn.Module:
        """
        Main training loop for the model.

        Args:
            epochs (int): Number of training epochs.
            log_layers (bool): Whether to log layer histograms.
            save_best (bool): Whether to save the best performing model.
            upper_bound (float): Early stopping threshold.
            debug (bool): If True, prints additional debug information.

        Returns:
            nn.Module: The best performing model.
        """
        start_time = time.time()
        best_acc = 0.0
        best_model = None
        steps_per_epoch = len(self.train_loader)

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            if debug:
                epoch_start = time.time()

            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step = epoch * steps_per_epoch + i
                acc = (outputs.argmax(1) == y).float().mean().item()

                self.writer.add_scalar("Loss/train", loss.item(), step)
                self.writer.add_scalar("Accuracy/train", acc, step)

            if log_layers:
                self._log_layers(epoch)

            val_loss, val_acc = self.validate()
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(self.model)
                print(
                    f"\nNew best model found at epoch {epoch+1} "
                    f"(accuracy: {val_acc:.2f}%)\n"
                )

            if val_acc >= upper_bound:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(accuracy: {val_acc:.2f}%)"
                )
                break

            if debug:
                epoch_duration = time.time() - epoch_start
                print(
                    f"Epoch {epoch+1}/{epochs} completed "
                    f"in {epoch_duration:.2f}s"
                    f" (Loss: {val_loss:.4f}, "
                    f"Accuracy: {val_acc:.2f}%)"
                )

        # Report and log total training time
        elapsed = time.time() - start_time
        minutes, seconds = divmod(elapsed, 60)
        print(f"⏱️ Total training time: {int(minutes)}m {int(seconds)}s")

        time_log_path = f"{TENSORBOARD_DIR}_training_time.txt"
        with open(time_log_path, "w") as f:
            f.write(
                f"{elapsed:.2f} seconds ({int(minutes)}m {int(seconds)}s)\n"
            )

        # Save model if required
        if save_best and best_model:
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            save_path = f"{MODEL_FILE}-{best_acc:.2f}.pkl"
            torch.save(best_model.state_dict(), save_path)

        self.writer.close()
        return best_model
