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


DATASET_CONFIG_PATH = 'SIBGRAPI2025_classifier/datasets/generic_dataset_v1/data.yaml'
MODEL_TYPE = 'yolov8n-pose.pt'
PROJECT_NAME = 'SIBGRAPI2025_classifier/results'
EXPERIMENT_NAME = 'generalist_v1_myolov8n_e100_s640_b16'
OUTPUT_MODEL_PATH = f'SIBGRAPI2025_classifier/runs/{EXPERIMENT_NAME}/weights/best.pt'
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16


def get_device():
    """
    Detecta e retorna o melhor dispositivo disponível (CUDA, MPS ou CPU).
    """
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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


def validate_dataset_path(dataset_path: str):
    """
    Valida a existência e a extensão do arquivo de configuração do dataset.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")
    if not dataset_path.endswith('.yaml'):
        raise ValueError(f"Formato inválido: {dataset_path}")
    print(f"Arquivo de configuração do dataset validado: {dataset_path}")


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
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
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
    print("Versão do PyTorch:", torch.__version__)

    device = get_device()
    print_device_info(device)

    # validate_dataset_path(DATASET_CONFIG_PATH)

    # model, _ = train_model(MODEL_TYPE, DATASET_CONFIG_PATH, device)

    # # evaluate_model(model)
    # save_model(model, OUTPUT_MODEL_PATH)


if __name__ == '__main__':
    main()
