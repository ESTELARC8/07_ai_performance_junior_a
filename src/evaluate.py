from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionCNN


MODEL_PATH = Path("models/fashion_cnn.pth")
BATCH_SIZE = 64


def main():
    device = torch.device("cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo: {MODEL_PATH}"
        )

    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FashionCNN().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print("Modelo:", MODEL_PATH)
    print("Muestras evaluadas:", total)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()