import torch.nn as nn


# Define una red neuronal personalizada heredando de nn.Module, que es la clase base de los modelos en PyTorch.
class FashionCNN(nn.Module):
    def __init__(self):
        # Inicializa correctamente la parte heredada de nn.Module.
        super().__init__()

        # Bloque encargado de extraer características de las imágenes mediante convoluciones y reducir progresivamente su resolución.
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,   # Fashion-MNIST tiene un único canal: escala de grises.
                out_channels=16, # La convolución genera 16 mapas de características.
                kernel_size=3,
                padding=1,       # Mantiene el tamaño espacial 28x28 tras la convolución.
            ),
            nn.ReLU(),

            # Reduce cada dimensión espacial a la mitad: 28x28 -> 14x14.
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,  # Recibe los 16 mapas generados por la convolución anterior.
                out_channels=32, # Genera 32 mapas de características.
                kernel_size=3,
                padding=1,       # Mantiene el tamaño espacial 14x14.
            ),
            nn.ReLU(),

            # Reduce de nuevo las dimensiones espaciales: 14x14 -> 7x7.
            nn.MaxPool2d(kernel_size=2),
        )

        # Bloque encargado de convertir las características extraídas en una predicción sobre las 10 clases.
        self.classifier = nn.Sequential(
            # Convierte [batch, 32, 7, 7] en [batch, 1568].
            nn.Flatten(),

            # 32 * 7 * 7 = 1568 características de entrada.
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),

            # Produce 10 logits, uno por cada clase de Fashion-MNIST.
            nn.Linear(64, 10),
        )

    # Define el recorrido que sigue una entrada cuando se ejecuta model(x).
    def forward(self, x):
        # Primero extrae características mediante convoluciones y pooling.
        x = self.features(x)

        # Después utiliza esas características para realizar la clasificación.
        x = self.classifier(x)

        return x