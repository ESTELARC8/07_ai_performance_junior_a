from pathlib import Path

import torch
import onnx

from model import FashionCNN


PYTORCH_MODEL_PATH = Path("models/fashion_cnn.pth")
ONNX_MODEL_PATH = Path("models/fashion_cnn.onnx")


def main():
    device = torch.device("cpu")

    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo PyTorch: {PYTORCH_MODEL_PATH}"
        )

    model = FashionCNN().to(device)

    state_dict = torch.load(
        PYTORCH_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
    )

    ONNX_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch_size"),
            }
        },
        dynamo=True,
        external_data=False,
    )

    onnx_model = onnx.load(ONNX_MODEL_PATH)

    onnx.checker.check_model(onnx_model)

    print(f"Modelo ONNX exportado en: {ONNX_MODEL_PATH}")
    print("Validación estructural ONNX: OK")


if __name__ == "__main__":
    main()