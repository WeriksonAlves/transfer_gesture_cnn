# src/tester.py

import torch
import numpy as np
import matplotlib.pyplot as plt


class ModelTester:
    def __init__(self, model, test_data, device, class_names):
        self.model = model.to(device)
        self.test_data = test_data
        self.device = device
        self.class_names = class_names
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def sample_and_predict(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        index = np.random.randint(len(self.test_data))
        sample, label = self.test_data[index]

        plt.figure(figsize=(2, 2))
        plt.axis('off')
        img = sample.numpy().transpose(1, 2, 0)
        img = np.clip(self.std * img + self.mean, 0, 1)
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
