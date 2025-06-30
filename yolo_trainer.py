'''
Treinamento do modelo YOLOv8 para detecção de gestos

Este script treina um modelo YOLOv8 para detecção de gestos usando um dataset
genérico. O modelo é treinado com base em um arquivo de configuração YAML que
contém informações sobre o dataset. O modelo é salvo após o treinamento e
avaliado com métricas de desempenho. O script também detecta automaticamente o
dispositivo disponível (CUDA, MPS ou CPU) para treinamento. O treinamento é
realizado com parâmetros configuráveis, como número de épocas, tamanho da
imagem e tamanho do lote. O modelo é salvo em um diretório específico após o
treinamento. O script é executado a partir da função principal, que orquestra
todo o processo de treinamento e avaliação.
'''

# Importa as bibliotecas necessárias
import os
import torch
from ultralytics import YOLO

from src.config import (
    BATCH_SIZE,
    # EPOCHS,
    # LEARNING_RATE,
    OPTIMIZER,
    NAME_PATH,
    DATASET_PATH,
    FREEZE_BACKBONE,
    MODEL_TRAINED_PATH,
    TENSORBOARD_DIR,
    MODEL_FILE,
    OUTPUT_PATH
)
EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224


def print_device_info(device: str):
    """
    Exibe informações sobre o dispositivo utilizado.
    """
    print(f"Dispositivo selecionado: {device}")
    if device == 'cuda':
        print("CUDA disponível:", torch.cuda.is_available())
        print("Total de GPUs:", torch.cuda.device_count())
        print("GPU atual:", torch.cuda.current_device())
        print("Nome da GPU:", torch.cuda.get_device_name(
            torch.cuda.current_device()))
    else:
        print("Nenhuma GPU CUDA disponível.")


def train_model(model_path: str, dataset_path: str, device: str):
    """
    Executa o treinamento do modelo YOLO.
    """
    print("Inicializando modelo:", model_path)
    model = YOLO(model_path)

    if model is None:
        raise RuntimeError("Erro ao carregar o modelo YOLO.")

    results = model.train(
        data=dataset_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=device,
        workers=4,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
        freeze=FREEZE_BACKBONE,  # Freezing the backbone layers
        project=OUTPUT_PATH,
        name=NAME_PATH,
        pretrained=True
    )

    return model, results


def evaluate_model(model: YOLO):
    """
    Executa a avaliação do modelo treinado.
    """
    metrics = model.val()
    print("Métricas de avaliação:", metrics)


def save_model(model: YOLO, path: str):
    """
    Salva os pesos do modelo em disco.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Modelo salvo em: {path}")


def main():
    """
    Função principal do pipeline de treinamento.
    """
    # Display PyTorch version and device info
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # # Load dataset
    # loader = DatasetLoader(DATASET_PATH, BATCH_SIZE)
    # data = loader.load()

    model, _ = train_model(MODEL_TRAINED_PATH, DATASET_PATH, device)

    # evaluate_model(model)
    save_model(model, MODEL_FILE)


if __name__ == '__main__':
    main()
