from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionCNN

PYTORCH_MODEL_PATH = Path("models/fashion_cnn.pth")
ONNX_MODEL_PATH = Path("models/fashion_cnn.onnx")

BATCH_SIZE = 32


def main():
    # Comprueba que existen ambos artefactos antes de hacer la comparación.
    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo PyTorch: {PYTORCH_MODEL_PATH}"
        )

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo ONNX: {ONNX_MODEL_PATH}"
        )

    device = torch.device("cpu")

    # -------------------------
    # Dataset
    # -------------------------

    # Convierte las imágenes a tensor float32 con valores en [0, 1].
    transform = transforms.ToTensor()

    # Carga únicamente la partición oficial de test.
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

    # Obtiene un único batch del DataLoader para usar exactamente las mismas imágenes en PyTorch y ONNX Runtime.
    images, labels = next(iter(test_loader))

    images = images.to(device)

    # -------------------------
    # PyTorch
    # -------------------------

    # Recrea la arquitectura del modelo PyTorch.
    pytorch_model = FashionCNN().to(device)

    # Carga los pesos entrenados en CPU.
    state_dict = torch.load(
        PYTORCH_MODEL_PATH,
        map_location=device,
    )

    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    # Ejecuta inferencia sin calcular gradientes.
    with torch.inference_mode():
        pytorch_output = pytorch_model(images)

    # Convierte la salida de PyTorch a NumPy para poder compararla directamente con la salida de ONNX Runtime.
    pytorch_output = pytorch_output.cpu().numpy()

    # -------------------------
    # ONNX Runtime
    # -------------------------

    # Crea una sesión de inferencia ONNX forzando CPUExecutionProvider.
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    # ONNX Runtime trabaja aquí con arrays NumPy.
    onnx_input = images.cpu().numpy()

    # Ejecuta el modelo ONNX:
    # ["logits"] indica la salida solicitada y
    # {"input": onnx_input} asocia el nombre de entrada del grafo con los datos.
    onnx_output = session.run(
        ["logits"],
        {
            "input": onnx_input,
        },
    )[0]

    # -------------------------
    # Comparación numérica
    # -------------------------

    # Calcula la diferencia absoluta elemento a elemento entre los logits de PyTorch y ONNX Runtime.
    absolute_difference = np.abs(
        pytorch_output - onnx_output
    )

    max_difference = absolute_difference.max()
    mean_difference = absolute_difference.mean()

    # -------------------------
    # Comparación de predicciones
    # -------------------------

    # Selecciona para cada imagen la clase con el logit más alto.
    # axis=1 recorre la dimensión correspondiente a las 10 clases.
    pytorch_predictions = np.argmax(
        pytorch_output,
        axis=1,
    )

    onnx_predictions = np.argmax(
        onnx_output,
        axis=1,
    )

    # Comprueba si ambos arrays de predicciones son exactamente iguales.
    same_predictions = np.array_equal(
        pytorch_predictions,
        onnx_predictions,
    )

    # Cuenta cuántas predicciones coinciden entre ambos runtimes.
    matching_predictions = np.sum(
        pytorch_predictions == onnx_predictions
    )

    # -------------------------
    # Resultados
    # -------------------------

    print("Batch evaluado:", BATCH_SIZE)

    print("\nShapes:")
    print("PyTorch:", pytorch_output.shape)
    print("ONNX Runtime:", onnx_output.shape)

    print("\nDiferencia numérica:")
    print(f"Máxima diferencia absoluta: {max_difference:.10f}")
    print(f"Diferencia absoluta media: {mean_difference:.10f}")

    print("\nPredicciones:")
    print(
        f"Coinciden: {matching_predictions}/{BATCH_SIZE}"
    )
    print(
        "Todas las predicciones coinciden:",
        same_predictions,
    )

    # Comprueba automáticamente que ambas salidas son numéricamente suficientemente próximas dentro de las tolerancias definidas.
    # Si no se cumplen, lanza una excepción.
    np.testing.assert_allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-4,
        atol=1e-5,
    )

    print("\nValidación PyTorch vs ONNX Runtime: OK")


if __name__ == "__main__":
    main()