# src/config.py
dataset = 'INF692_GEST_CLAS_GE'
DATASET_PATH = f'data/annotated/{dataset}.v3i.folder/'
TENSORBOARD_DIR = 'tensorboard/ResNet18/'
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_PATH = f'models/resnet18-{dataset}-{BATCH_SIZE}-{EPOCHS}-{LEARNING_RATE}'
MODEL_TEST_PATH = 'models/resnet18-INF692_GEST_CLAS_GE-32-100-0.0001-97.22.pkl'
