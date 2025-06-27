# src/trainer.py

import copy
import torch
import time
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from src.config import LEARNING_RATE, TENSORBOARD_DIR


class Trainer:
    def __init__(self, model, data, device, prefix, model_path):
        self.model = model
        self.device = device
        self.train_loader = data['train']
        self.valid_loader = data['valid']
        self.classes = data['classes']
        now = datetime.now()
        self.prefix = prefix + '-' + now.strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=TENSORBOARD_DIR + self.prefix)
        self.model_path = model_path
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        # self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def _log_layers(self, epoch):
        for i, layer in enumerate(self.model.modules()):
            if isinstance(layer, nn.Linear):
                self.writer.add_histogram(
                    f'Weight/layer{i}', layer.weight, epoch
                )

    def validate(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, labels in self.valid_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    def train(self, epochs, log_layers=False, save_best=True,
              upper_bound=100.0,
              ):
        start_time = time.time()
        best_acc = 0.0
        best_model = None
        steps_per_epoch = len(self.train_loader)

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step = epoch * steps_per_epoch + i
                acc = (outputs.argmax(1) == y).float().mean().item()

                self.writer.add_scalar('Loss/train', loss.item(), step)
                self.writer.add_scalar('Accuracy/train', acc, step)

            if log_layers:
                self._log_layers(epoch)

            val_acc = self.validate()
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            if val_acc > best_acc:
                best_model = copy.deepcopy(self.model)
                best_acc = val_acc

            if val_acc >= upper_bound:
                print(
                    f"Early stopping at epoch {epoch + 1} with"
                    f"accuracy {val_acc:.2f}%"
                )
                break

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Salva tempo em arquivo ou imprime:
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"⏱️ Training time: {int(minutes)}m {int(seconds)}s")

        with open(
            f"{TENSORBOARD_DIR + self.prefix}_training_time.txt", "w"
        ) as f:
            f.write(f"{elapsed_time:.2f} seconds "
                    f"({int(minutes)}m {int(seconds)}s)\n")

        if save_best:
            torch.save(best_model.state_dict(),
                       self.model_path + f'-{best_acc:.2f}.pkl')

        self.writer.close()
        return best_model
