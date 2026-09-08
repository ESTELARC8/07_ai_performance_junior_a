from torchvision import datasets


def main():
    datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
    )

    datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
    )

    print("Fashion-MNIST descargado correctamente.")


if __name__ == "__main__":
    main()