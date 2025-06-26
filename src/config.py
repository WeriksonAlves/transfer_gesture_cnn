# src/config.py

# dataset = 'INF692_GEST_CLAS_GE-MY'
dataset = 'INF692_GEST_CLAS_MY'
# dataset = 'INF692_GEST_CLAS_GE-MY'

DATASET_PATH = f'data/annotated/{dataset}.v3i.folder/'

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

PREFIX = 'ResNet-{}-b-{}-e-{}-lr-{}'.format(
    dataset, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
TENSORBOARD_DIR = f'tensorboard/{PREFIX}/'
MODEL_PATH = f'models/resnet18-{PREFIX}'
MODEL_TL_PATH = None
