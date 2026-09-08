from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionCNN

SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

MODEL_PATH = Path("models/fashion_cnn.pth")


def evaluate(model, data_loader, criterion):
    # Cambia el modelo a modo evaluación.
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Desactiva el cálculo de gradientes durante la evaluación.
    with torch.inference_mode():
        for images, labels in data_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Multiplica la loss media del batch por el número de muestras para poder calcular después la loss media global del dataset.
            total_loss += loss.item() * images.size(0)

            # Selecciona la clase con el logit más alto para cada imagen.
            predictions = outputs.argmax(dim=1)

            # Acumula el número de predicciones correctas.
            correct += (predictions == labels).sum().item()

            # Acumula el número total de muestras evaluadas.
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    # Fija la semilla de PyTorch para reducir la variabilidad entre ejecuciones.
    torch.manual_seed(SEED)

    # Fuerza tanto el entrenamiento como la evaluación en CPU.
    device = torch.device("cpu")

    print("Dispositivo:", device)

    # Convierte las imágenes a tensor float32 y escala los píxeles al rango [0, 1].
    transform = transforms.ToTensor()

    # Carga la partición oficial de entrenamiento.
    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=transform,
    )

    # Carga la partición oficial de test.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    # Crea un generador aleatorio independiente para controlar de forma reproducible el orden aleatorio del DataLoader.
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,      # Mezcla las muestras en cada época durante entrenamiento.
        num_workers=0,     # La carga de datos se realiza en el proceso principal.
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,     # En test no es necesario alterar el orden de las muestras.
        num_workers=0,
    )

    # Crea el modelo y lo mueve al dispositivo seleccionado.
    model = FashionCNN().to(device)

    # CrossEntropyLoss es adecuada para clasificación multiclase y trabaja directamente con los logits de salida del modelo.
    criterion = nn.CrossEntropyLoss()

    # Adam actualizará todos los parámetros entrenables del modelo.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        # Cambia el modelo a modo entrenamiento.
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Elimina los gradientes acumulados de la iteración anterior.
            optimizer.zero_grad()

            # Forward pass: obtiene los logits del modelo.
            outputs = model(images)

            # Calcula el error entre las predicciones y las etiquetas reales.
            loss = criterion(outputs, labels)

            # Backward pass: calcula los gradientes de la loss respecto a los parámetros entrenables.
            loss.backward()

            # Actualiza los parámetros del modelo usando los gradientes calculados.
            optimizer.step()

            # Acumula la loss ponderada por el número de muestras del batch.
            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Calcula las métricas globales de entrenamiento de la época.
        train_loss = running_loss / total
        train_accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy * 100:.2f}%"
        )

    # Evalúa el modelo entrenado sobre la partición oficial de test.
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nResultado en test:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")

    # Crea la carpeta models si todavía no existe.
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guarda únicamente los pesos del modelo, no el objeto completo.
    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()