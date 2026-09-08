from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionCNN

MODEL_PATH = Path("models/fashion_cnn.pth")
BATCH_SIZE = 64


def main():
    # Fuerza la evaluación en CPU.
    device = torch.device("cpu")

    # Comprueba que existe el archivo de pesos antes de intentar cargarlo.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo: {MODEL_PATH}"
        )

    # Convierte cada imagen de Fashion-MNIST a tensor float32 y escala sus valores de píxel al rango [0, 1].
    transform = transforms.ToTensor()

    # Carga únicamente la partición oficial de test.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,   # No descarga nada: espera que el dataset ya exista localmente.
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,    # En evaluación no necesitamos alterar el orden de las muestras.
        num_workers=0,    # La carga de datos se realiza en el proceso principal.
    )

    # Crea la arquitectura del modelo y la mueve al dispositivo seleccionado.
    model = FashionCNN().to(device)

    # Carga los pesos guardados asegurando que se mapean al dispositivo actual.
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    # Introduce los pesos cargados dentro de la arquitectura FashionCNN.
    model.load_state_dict(state_dict)

    # Cambia el modelo a modo evaluación.
    model.eval()

    correct = 0
    total = 0

    # Desactiva el cálculo de gradientes durante la inferencia.
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Selecciona, para cada imagen, el índice del logit más alto.
            predictions = outputs.argmax(dim=1)

            # Cuenta cuántas predicciones coinciden con la etiqueta real.
            correct += (predictions == labels).sum().item()

            # Acumula el número total de muestras evaluadas.
            total += labels.size(0)

    # Accuracy = número de predicciones correctas / número total de muestras.
    accuracy = correct / total

    print("Modelo:", MODEL_PATH)
    print("Muestras evaluadas:", total)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()