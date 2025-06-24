# infer.py

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from src.config import DATASET_PATH, MODEL_TEST_PATH
from src.dataloader import DatasetLoader
from src.model_builder import prepare_model


def run_inference(model, test_loader, device, class_names):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader,
                                                  desc="Inferencing")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

            for idx in range(images.size(0)):
                predictions.append({
                    "sample_id": i * test_loader.batch_size + idx,
                    "true_label": class_names[labels[idx].item()],
                    "predicted_label": class_names[pred_classes[idx].item()],
                    "confidence": probs[idx][pred_classes[idx]].item()
                })

    return predictions


def plot_confusion_matrix(y_true, y_pred, class_names,
                          normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', cbar=False)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Confusion Matrix" + (" (Normalized)" if normalize else "")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader = DatasetLoader(DATASET_PATH, batch_size=32)
    data = loader.load()
    test_loader = data["test"]
    class_names = data["classes"]

    # Load model
    model = prepare_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_TEST_PATH, map_location=device))
    model.to(device)

    # Run inference
    results = run_inference(model, test_loader, device, class_names)

    # Save CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/predictions.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Predictions saved to: {csv_path}")

    # Evaluation
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    print("📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrices
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=False,
        save_path="outputs/confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=True,
        save_path="outputs/confusion_matrix_normalized.png"
    )


if __name__ == "__main__":
    main()
