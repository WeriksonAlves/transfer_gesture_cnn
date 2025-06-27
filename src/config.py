# src/config.py

mode = 3

if mode == 1:  # ResNet18 do ImageNet → Personalizado
    dataset = 'INF692_GEST_CLAS_MY'
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    MODEL_TL_PATH = None

elif mode == 2:  # ResNet18 do ImageNet → Genérico + Personalizado
    dataset = 'INF692_GEST_CLAS_GE-MY'
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    MODEL_TL_PATH = None

elif mode == 3:  # ResNet18 do ImageNet → Genérico
    dataset = 'INF692_GEST_CLAS_GE'
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    MODEL_TL_PATH = None
else:
    raise ValueError("Invalid mode selected. Choose 1, 2, or 3.")


DATASET_PATH = f'data/annotated/{dataset}.v3i.folder/'
PREFIX = 'ResNet-{}-b-{}-e-{}-lr-{}'.format(
    dataset, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
TENSORBOARD_DIR = f'tensorboard/{PREFIX}/'
MODEL_PATH = f'models/resnet18-{PREFIX}'
