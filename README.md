# 🤖 Human Gesture Recognition using CNNs for Human-Robot Interaction

This repository contains the final project for the **INF692 – Convolutional Neural Networks for Computer Vision** course. The project aims to recognize human gestures in images using both **standard CNN classifiers (ResNet18)** and **pose-based representations via YOLOv8-Pose**.

---

## 📌 Problem

Recognizing human gestures is a fundamental task for **hands-free control of mobile robots (UGVs/UAVs)**. The goal is to replace traditional handcrafted methods with **modern, robust CNN-based solutions**, suitable for deployment in real-world environments.

---

## 🎯 Objectives

- Train a **CNN classifier** (ResNet18) on generic and personalized gesture datasets.
- Apply **transfer learning** and **fine-tuning** strategies.
- Train a **YOLOv8-Pose** model and use pose keypoints for gesture recognition.
- Evaluate and compare performance across approaches.

---

## 📁 Project Structure

```

├── data/                     # Datasets (YOLOv8-format + ImageFolder)
├── models/                   # Trained models (.pt, .pkl)
├── outputs/                  # Evaluation metrics, confusion matrices, predictions
├── tensorboard/              # TensorBoard logs
├── src/
│   ├── config.py             # Experiment settings
│   ├── dataloader.py         # ImageFolder loader
│   ├── model\_builder.py      # ResNet18 model setup
│   ├── trainer.py            # PyTorch training pipeline
│   ├── tester.py             # Evaluation, metrics, confusion matrix
│   └── utils.py              # Utilities (device info, image viewers)
├── main\_trainer.py           # Train ResNet18 (PyTorch)
├── main\_tester.py             # Batch evaluation of trained models
└── README.md                 # This file

```

---

## 🧠 Dataset

We used two dataset splits:

- **Generic (GE)**: standard hand gestures from multiple users.
- **Personalized (MY)**: user-specific gesture samples.

Additionally, YOLOv8-Pose annotations were created via **Roboflow** and exported in pose estimation format.

---

## 🧪 Experiments

### ✅ CNN Classifier (ResNet18)

| Strategy                                   | Notes                          |
|--------------------------------------------|--------------------------------|
| ImageNet → Generic (fine-tuning)           | `train_mode = 0`               |
| ImageNet → Personalized (fine-tuning)      | `train_mode = 1`               |
| Generic → Personalized (transfer)          | `train_mode = 2`               |
| Generic → GE+MY (transfer)                 | `train_mode = 3`               |
| ImageNet → Personalized (transfer) Adam    | `train_mode = 4`               |
| ImageNet → GE+MY (transfer) Adam           | `train_mode = 5`               |
| ImageNet → Personalized (transfer) SGD     | `train_mode = 6`               |
| ImageNet → GE+MY (transfer) SGD            | `train_mode = 7`               |

### ✅ YOLOv8-Pose + Keypoint Classifier (ongoing)

- Trained using `yolov8n-pose.pt` with `freeze=2`
- Images annotated with 11 keypoints (Roboflow format)

---

## 📊 Results

- CNN classifier (ResNet18) achieved up to **100% accuracy** on validation.
- YOLOv8-Pose model also showed **high keypoint precision** (`mAP50-95 ≈ 84%`).
- Confusion matrices and classification reports are available in `/outputs/`.

---

## 🚀 How to Run

### 🔧 Setup (Optional)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

### 🏋️‍♂️ Train ResNet18:

Edit the experiment parameters in `main_trainer.py`, setting:

* `TRAIN_MODE` to desired strategy (e.g., `0` to `7`)
* `EPOCHS`, `BATCH_SIZE`, `OPTIMIZER`, etc.

Then, run:

```bash
python main_trainer.py
```

This will:

* Load the dataset
* Build the ResNet18 model
* Train and save the best model checkpoint
* Log training metrics to TensorBoard

---

### 🧪 Evaluate All Models:

To evaluate trained models on the combined dataset (GEN+PER), update and run:

```bash
python main_tester.py
```

This script will:

* Load each `.pkl` model found in `models/`
* Evaluate it on a fixed test split
* Generate confusion matrices and classification reports
* Save predictions to `/outputs/eval/`

---

### 🧍 YOLOv8-Pose Inference (Optional):


---

### 📈 Launch TensorBoard:

```bash
tensorboard --logdir=tensorboard/
```

This will allow visualizing:

* Training/validation loss
* Accuracy curves
* Weights and predictions histograms

---

## 👨‍💻 Author

**Wérikson Alves**
PhD Candidate in Computer Science
[NERo - Núcleo de Especialização em Robótica](https://github.com/NERo-UFV)

---

## 📄 License

This project is part of an academic evaluation and is shared for educational purposes.
