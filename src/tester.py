# src/tester.py

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from src.config import OUTPUT_PATH, BATCH_SIZE


class Tester:
    """
    Tester class for evaluating a trained model on a test dataset.

    Attributes:
        model (nn.Module): Model to be evaluated.
        device (torch.device): Device used (CPU or CUDA).
        test_loader (DataLoader): Test set DataLoader.
        class_names (list): List of class names.
        mean (np.ndarray): Normalization mean for images.
        std (np.ndarray): Normalization standard deviation for images.
    """

    def __init__(self, model, data, device):
        self.model = model.to(device)
        self.test_loader = data["test"]
        self.class_names = data["classes"]
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def sample_and_predict(self, seed: Optional[int] = None):
        """
        Selects one test image, runs inference, and displays prediction.

        Args:
            seed (int, optional): Fix random selection for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        index = np.random.randint(len(self.test_loader))
        sample, label = self.test_loader.dataset[index]

        img = sample.numpy().transpose(1, 2, 0)
        img = np.clip(self.std * img + self.mean, 0, 1)

        plt.figure(figsize=(2, 2))
        plt.axis("off")
        plt.imshow(img)
        plt.title("Sample Image")
        plt.show()

        self.model.eval()
        x = sample.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output.squeeze(0), dim=0)
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

        print(f"Sample Index: {index}")
        print("Prediction:", "Correct" if pred_class == label else "Incorrect")
        print(f"Predicted: {self.class_names[pred_class]} | "
              f"True: {self.class_names[label]} | "
              f"Confidence: {confidence * 100:.2f}%")

        return pred_class, label, confidence

    def infer(self):
        """
        Runs inference on the entire test set and collects predictions.

        Args:
            test_loader (DataLoader): Test set DataLoader.

        Returns:
            List[dict]: List of prediction entries.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(
                self.test_loader, desc="Inferencing"
            )):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                pred_classes = torch.argmax(probs, dim=1)

                for idx in range(images.size(0)):
                    predictions.append({
                        "sample_id": i * BATCH_SIZE + idx,
                        "true_label": self.class_names[labels[idx].item()],
                        "predicted_label": self.class_names[
                            pred_classes[idx].item()],
                        "confidence": probs[idx][
                            pred_classes[idx]
                        ].item()
                    })

        return predictions

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names,
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plots a confusion matrix and optionally saves it.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.
            class_names (list): Class names for axes.
            normalize (bool): Normalize matrix values.
            save_path (str, optional): File path to save the image.
        """
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            cbar=False
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"🖼️ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, results: list, output_dir: str = None) -> None:
        """
        Saves the inference results and evaluation report.

        Args:
            results (list): Output from `infer()`.
            output_dir (str, optional): Directory to save results.
                                       Defaults to OUTPUT_PATH.
        """
        if output_dir is None:
            output_dir = OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_dir, "predictions.csv")
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Predictions saved to: {csv_path}")

        # Metrics
        y_true = df["true_label"]
        y_pred = df["predicted_label"]
        acc = accuracy_score(y_true, y_pred)

        print(f"\n🎯 Accuracy: {acc:.4f}\n")
        print("📋 Classification Report:\n")
        print(classification_report(
            y_true, y_pred, target_names=self.class_names
        ))

        # Confusion matrices
        self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            normalize=False,
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )

        self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            normalize=True,
            save_path=os.path.join(output_dir,
                                   "confusion_matrix_normalized.png")
        )
