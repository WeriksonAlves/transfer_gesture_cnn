# analyze_predictions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()


def main():
    csv_path = "outputs/predictions.csv"
    df = pd.read_csv(csv_path)

    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    class_names = sorted(y_true.unique())

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}\n")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("📋 Classification Report:\n")
    print(report)

    # Confusion Matrix (Raw)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False)

    # Confusion Matrix (Normalized)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)


if __name__ == "__main__":
    main()
