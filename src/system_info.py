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
    # Agrupa en un único diccionario la información relevante del entorno para poder guardar junto al benchmark el contexto hardware/software.
    system_info = {
        "system": {
            # Información general del sistema operativo y arquitectura.
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },

        "cpu": {
            # logical=False devuelve núcleos físicos.
            "physical_cores": psutil.cpu_count(logical=False),

            # logical=True incluye también los hilos lógicos del procesador.
            "logical_cores": psutil.cpu_count(logical=True),
        },

        "memory": {
            # Convierte la RAM total del sistema de bytes a GiB y redondea el resultado a dos decimales.
            "total_ram_gb": round(
                psutil.virtual_memory().total
                / (1024 ** 3),
                2,
            ),
        },

        "software": {
            # Guarda las versiones exactas de las librerías principales para poder reproducir e interpretar correctamente los resultados.
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },

        "onnxruntime": {
            # Lista los Execution Providers disponibles en ONNX Runtime.
            "available_providers": ort.get_available_providers(),

            # Deja registrado explícitamente qué provider utiliza el benchmark.
            "benchmark_provider": "CPUExecutionProvider",
        },
    }

    # Crea la carpeta results si todavía no existe.
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guarda la información del entorno en formato JSON.
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

    # Muestra por consola exactamente la misma información que se guarda en el JSON.
    print(json.dumps(system_info, indent=4))

    print(
        f"\nInformación del sistema guardada en: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()