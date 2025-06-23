# infer.py

import torch
import pandas as pd
from tqdm import tqdm
from src.config import DATASET_PATH, MODEL_PATH
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    loader = DatasetLoader(DATASET_PATH, batch_size=32)
    data = loader.load()
    test_loader = data["test"]
    class_names = data["classes"]

    # Load model
    model = prepare_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # Run inference
    results = run_inference(model, test_loader, device, class_names)

    # Save to CSV
    df = pd.DataFrame(results)
    output_path = "outputs/predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
