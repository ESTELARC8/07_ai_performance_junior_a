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
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print("Dispositivo:", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FashionCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy * 100:.2f}%"
        )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nResultado en test:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()