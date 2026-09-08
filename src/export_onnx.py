from pathlib import Path

import torch
import onnx

from model import FashionCNN

PYTORCH_MODEL_PATH = Path("models/fashion_cnn.pth")
ONNX_MODEL_PATH = Path("models/fashion_cnn.onnx")


def main():
    # Fuerza la carga y exportación del modelo desde CPU.
    device = torch.device("cpu")

    # Comprueba que existe el modelo PyTorch antes de intentar exportarlo.
    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo PyTorch: {PYTORCH_MODEL_PATH}"
        )

    # Crea la arquitectura del modelo.
    model = FashionCNN().to(device)

    # Carga los pesos entrenados asegurando que se mapean a CPU.
    state_dict = torch.load(
        PYTORCH_MODEL_PATH,
        map_location=device,
    )

    # Introduce los pesos guardados dentro de la arquitectura.
    model.load_state_dict(state_dict)

    # Cambia el modelo a modo evaluación antes de exportarlo.
    model.eval()

    # Crea una entrada de ejemplo con la forma esperada por la CNN: [batch, canales, alto, ancho] = [1, 1, 28, 28].
    dummy_input = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
    )

    # Crea la carpeta de destino si no existe.
    ONNX_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,

        # Define los nombres visibles de entrada y salida dentro del grafo ONNX.
        input_names=["input"],
        output_names=["logits"],

        # Declara la dimensión 0 de la entrada como dinámica para permitir distintos tamaños de batch sin volver a exportar el modelo.
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch_size"),
            }
        },

        # Utiliza el exportador moderno basado en torch.export.
        dynamo=True,

        # Guarda los pesos dentro del propio archivo ONNX en lugar de crear un archivo externo adicional de datos.
        external_data=False,
    )

    # Carga de nuevo el archivo ONNX generado para comprobar su estructura.
    onnx_model = onnx.load(ONNX_MODEL_PATH)

    # Verifica que el modelo ONNX cumple las reglas estructurales del formato.
    onnx.checker.check_model(onnx_model)

    print(f"Modelo ONNX exportado en: {ONNX_MODEL_PATH}")
    print("Validación estructural ONNX: OK")


if __name__ == "__main__":
    main()