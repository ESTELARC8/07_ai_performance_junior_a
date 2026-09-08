from pathlib import Path
import json
import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
import torchvision


RESULTS_PATH = Path("results/system_info.json")


def main():
    system_info = {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },

        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },

        "memory": {
            "total_ram_gb": round(
                psutil.virtual_memory().total
                / (1024 ** 3),
                2,
            ),
        },

        "software": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },

        "onnxruntime": {
            "available_providers": ort.get_available_providers(),
            "benchmark_provider": "CPUExecutionProvider",
        },
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            system_info,
            file,
            indent=4,
        )

    print(json.dumps(system_info, indent=4))

    print(
        f"\nInformación del sistema guardada en: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()