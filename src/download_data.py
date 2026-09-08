from torchvision import datasets


def main():
    # Descarga/carga la partición oficial de entrenamiento de Fashion-MNIST.
    datasets.FashionMNIST(
        root="data",       # Carpeta raíz donde se almacenará el dataset.
        train=True,        # True selecciona las 60.000 imágenes de entrenamiento.
        download=True,     # Si el dataset no existe localmente, lo descarga.
    )

    # Descarga/carga la partición oficial de test de Fashion-MNIST.
    datasets.FashionMNIST(
        root="data",
        train=False,       # False selecciona las 10.000 imágenes oficiales de test.
        download=True,
    )

    print("Fashion-MNIST descargado correctamente.")


if __name__ == "__main__":
    main()