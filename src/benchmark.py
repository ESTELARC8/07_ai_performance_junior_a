from pathlib import Path
import csv
import time

import numpy as np
import onnxruntime as ort
import psutil
import yaml
from torchvision import datasets, transforms


ONNX_MODEL_PATH = Path("models/fashion_cnn.onnx")
CONFIG_PATH = Path("configs/benchmark.yaml")

TRIALS_RESULTS_PATH = Path("results/benchmark_trials.csv")
SUMMARY_RESULTS_PATH = Path("results/benchmark_summary.csv")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado la configuración: {CONFIG_PATH}"
        )

    # Carga el archivo YAML y lo convierte en estructuras Python como diccionarios y listas.
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Define los parámetros mínimos que debe tener la configuración para poder ejecutar el benchmark.
    required_fields = [
        "batch_sizes",
        "thread_counts",
        "trials",
        "warmup_runs",
        "repetitions",
    ]

    # Comprueba que no falte ningún parámetro obligatorio.
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Falta el campo obligatorio '{field}' en {CONFIG_PATH}"
            )

    return config


def load_input_batch(batch_size):
    # Convierte cada imagen a tensor float32 con valores en [0, 1].
    transform = transforms.ToTensor()

    # Utiliza imágenes reales de la partición oficial de test como entrada del benchmark.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    # Evita intentar construir un batch mayor que el dataset disponible.
    if batch_size > len(test_dataset):
        raise ValueError(
            f"batch_size={batch_size} es mayor que el número "
            f"de muestras disponibles ({len(test_dataset)})."
        )

    # Extrae las primeras batch_size imágenes y las convierte a NumPy.
    # Cada imagen individual tiene forma [1, 28, 28].
    images = [
        test_dataset[index][0].numpy()
        for index in range(batch_size)
    ]

    # Apila las imágenes creando un único array con forma: [batch_size, 1, 28, 28].
    batch = np.stack(images, axis=0)

    # Garantiza que ONNX Runtime recibe entradas float32.
    return batch.astype(np.float32)


def create_session(num_threads):
    # Permite configurar explícitamente cómo ONNX Runtime utiliza la CPU.
    session_options = ort.SessionOptions()

    # Número de hilos que ONNX Runtime puede utilizar para paralelizar el trabajo dentro de una operación.
    session_options.intra_op_num_threads = num_threads

    # Mantiene fijo el paralelismo entre operaciones para que la variable comparada en el benchmark sea principalmente intra_op_num_threads.
    session_options.inter_op_num_threads = 1

    # Crea la sesión forzando explícitamente la ejecución en CPU.
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    return session


def measure_latency(
    session,
    input_batch,
    warmup_runs,
    repetitions,
):
    # Asocia el nombre de entrada definido en el grafo ONNX con el batch que se utilizará para inferencia.
    input_feed = {
        "input": input_batch,
    }

    # Primera inferencia
    # Se mide aparte porque puede tener costes iniciales diferentes al comportamiento estable posterior.
    start = time.perf_counter_ns()

    session.run(
        ["logits"],
        input_feed,
    )

    end = time.perf_counter_ns()

    # perf_counter_ns devuelve nanosegundos.
    # Dividir entre 1.000.000 para convertirlo a milisegundos.
    first_run_ms = (end - start) / 1_000_000

    # Warm-up
    # Estas inferencias se ejecutan pero no forman parte de las estadísticas.
    for _ in range(warmup_runs):
        session.run(
            ["logits"],
            input_feed,
        )

    # Régimen estable
    latencies_ms = []

    # Mide individualmente cada inferencia una vez terminado el warm-up.
    for _ in range(repetitions):
        start = time.perf_counter_ns()

        session.run(
            ["logits"],
            input_feed,
        )

        end = time.perf_counter_ns()

        latency_ms = (end - start) / 1_000_000

        latencies_ms.append(latency_ms)

    # Devuelve por separado la primera inferencia y la distribución de latencias del régimen estable.
    return first_run_ms, np.array(latencies_ms)


def get_process_memory_mb():
    # Obtiene información del proceso Python que está ejecutando el benchmark.
    process = psutil.Process()

    # RSS representa la memoria RAM residente utilizada por el proceso.
    memory_bytes = process.memory_info().rss

    # Convierte bytes a MiB.
    return memory_bytes / (1024 * 1024)


def get_model_size_mb():
    # Obtiene el tamaño físico del archivo ONNX almacenado en disco.
    model_size_bytes = ONNX_MODEL_PATH.stat().st_size

    return model_size_bytes / (1024 * 1024)


def run_benchmark(
    batch_size,
    num_threads,
    warmup_runs,
    repetitions,
    trial,
):
    # Prepara el batch antes de comenzar la medición para que la carga y el preprocesado de datos no formen parte de la latencia de inferencia.
    input_batch = load_input_batch(batch_size)

    # Crea una sesión nueva para esta configuración de hilos.
    session = create_session(num_threads)

    first_run_ms, latencies_ms = measure_latency(
        session=session,
        input_batch=input_batch,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )

    # Calcula distintas estadísticas porque la media por sí sola no describe completamente la distribución de latencias.
    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))

    # Convierte la latencia media de milisegundos a segundos y calcula cuántas imágenes pueden procesarse teóricamente por segundo.
    throughput = (
        batch_size
        / (mean_ms / 1000.0)
    )

    # Estas medidas se realizan fuera de la región temporizada, por lo que no contaminan la latencia de session.run().
    memory_rss_mb = get_process_memory_mb()
    model_size_mb = get_model_size_mb()

    # Devuelve todas las métricas de este trial en una estructura fácil de almacenar posteriormente como CSV.
    return {
        "batch_size": batch_size,
        "threads": num_threads,
        "trial": trial,
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "first_run_ms": first_run_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "throughput_images_s": throughput,
        "memory_rss_mb": memory_rss_mb,
        "model_size_mb": model_size_mb,
    }


def build_summary(results):
    summary = []

    # Obtiene todas las combinaciones únicas de batch size y número de hilos que aparecen en los resultados.
    configurations = sorted(
        {
            (result["batch_size"], result["threads"])
            for result in results
        }
    )

    for batch_size, threads in configurations:
        # Selecciona únicamente los trials pertenecientes a la configuración actual.
        config_results = [
            result
            for result in results
            if result["batch_size"] == batch_size
            and result["threads"] == threads
        ]

        # Agrupa la misma métrica de todos los trials para poder calcular posteriormente estadísticas entre ejecuciones.
        first_runs = np.array(
            [result["first_run_ms"] for result in config_results]
        )

        means = np.array(
            [result["mean_ms"] for result in config_results]
        )

        medians = np.array(
            [result["median_ms"] for result in config_results]
        )

        p95_values = np.array(
            [result["p95_ms"] for result in config_results]
        )

        throughputs = np.array(
            [
                result["throughput_images_s"]
                for result in config_results
            ]
        )

        memory_values = np.array(
            [result["memory_rss_mb"] for result in config_results]
        )

        # Resume los distintos trials de una misma configuración en una única fila.
        summary.append(
            {
                "batch_size": batch_size,
                "threads": threads,
                "trials": len(config_results),

                # Media del first run obtenido en los distintos trials.
                "first_run_mean_ms": float(np.mean(first_runs)),

                # Media de las latencias medias de los distintos trials.
                "latency_mean_ms": float(np.mean(means)),

                # Desviación estándar de las latencias medias entre trials: da una idea de cuánto varían unas ejecuciones respecto a otras.
                "latency_mean_std_ms": float(np.std(means)),

                # Promedio de las medianas obtenidas en los trials.
                "median_mean_ms": float(np.mean(medians)),

                # Promedio de los p95 obtenidos en los trials.
                "p95_mean_ms": float(np.mean(p95_values)),

                "throughput_mean_images_s": float(
                    np.mean(throughputs)
                ),

                "memory_rss_mean_mb": float(
                    np.mean(memory_values)
                ),

                # El tamaño del mismo modelo no cambia entre trials, por lo que basta con tomar el valor del primero.
                "model_size_mb": config_results[0]["model_size_mb"],
            }
        )

    return summary


def save_csv(path, rows):
    # Crea la carpeta de destino si todavía no existe.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        # Utiliza las claves del primer diccionario como columnas del CSV.
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo ONNX: {ONNX_MODEL_PATH}"
        )

    config = load_config()

    # Extrae los parámetros experimentales del archivo YAML.
    batch_sizes = config["batch_sizes"]
    thread_counts = config["thread_counts"]
    trials = config["trials"]
    warmup_runs = config["warmup_runs"]
    repetitions = config["repetitions"]

    # Valida la configuración antes de empezar una ejecución potencialmente larga.
    if not batch_sizes:
        raise ValueError("batch_sizes no puede estar vacío.")

    if not thread_counts:
        raise ValueError("thread_counts no puede estar vacío.")

    if trials <= 0:
        raise ValueError("trials debe ser mayor que 0.")

    if warmup_runs < 0:
        raise ValueError("warmup_runs no puede ser negativo.")

    if repetitions <= 0:
        raise ValueError("repetitions debe ser mayor que 0.")

    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(
                "Todos los valores de batch_sizes deben ser mayores que 0."
            )

    for num_threads in thread_counts:
        if num_threads <= 0:
            raise ValueError(
                "Todos los valores de thread_counts deben ser mayores que 0."
            )

    print("Configuración global:")
    print("Batch sizes:", batch_sizes)
    print("Threads:", thread_counts)
    print("Trials:", trials)
    print("Warm-up runs:", warmup_runs)
    print("Repeticiones:", repetitions)

    results = []

    # Recorre todas las combinaciones: batch size × número de hilos × número de trial.
    for batch_size in batch_sizes:
        for num_threads in thread_counts:
            for trial in range(1, trials + 1):

                print("\n" + "=" * 60)

                print(
                    f"Batch size: {batch_size} | "
                    f"Threads: {num_threads} | "
                    f"Trial: {trial}/{trials}"
                )

                result = run_benchmark(
                    batch_size=batch_size,
                    num_threads=num_threads,
                    warmup_runs=warmup_runs,
                    repetitions=repetitions,
                    trial=trial,
                )

                # Guarda el resultado individual para generar después tanto el CSV de trials como el resumen agregado.
                results.append(result)

                print(
                    f"Primera ejecución: "
                    f"{result['first_run_ms']:.4f} ms"
                )

                print(
                    f"Latencia media: "
                    f"{result['mean_ms']:.4f} ms"
                )

                print(
                    f"Mediana: "
                    f"{result['median_ms']:.4f} ms"
                )

                print(
                    f"p95: "
                    f"{result['p95_ms']:.4f} ms"
                )

                print(
                    f"Throughput: "
                    f"{result['throughput_images_s']:.2f} imágenes/s"
                )

                print(
                    f"Memoria RSS: "
                    f"{result['memory_rss_mb']:.2f} MB"
                )

                print(
                    f"Tamaño modelo ONNX: "
                    f"{result['model_size_mb']:.4f} MB"
                )

    # Agrupa los trials correspondientes a cada configuración.
    summary = build_summary(results)

    # Guarda por separado los resultados individuales y el resumen agregado.
    save_csv(
        TRIALS_RESULTS_PATH,
        results,
    )

    save_csv(
        SUMMARY_RESULTS_PATH,
        summary,
    )

    print("\n" + "=" * 60)
    print(
        f"Trials guardados en: {TRIALS_RESULTS_PATH}"
    )
    print(
        f"Resumen guardado en: {SUMMARY_RESULTS_PATH}"
    )

    print("\nResumen final:")

    for result in summary:
        print("\n" + "-" * 50)

        print(
            f"Batch: {result['batch_size']} | "
            f"Threads: {result['threads']}"
        )

        print(
            f"Latencia media entre trials: "
            f"{result['latency_mean_ms']:.4f} ms"
        )

        print(
            f"Desviación entre trials: "
            f"{result['latency_mean_std_ms']:.4f} ms"
        )

        print(
            f"p95 medio: "
            f"{result['p95_mean_ms']:.4f} ms"
        )

        print(
            f"Throughput medio: "
            f"{result['throughput_mean_images_s']:.2f} imágenes/s"
        )


if __name__ == "__main__":
    main()