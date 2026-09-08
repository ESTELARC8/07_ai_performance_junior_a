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

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_fields = [
        "batch_sizes",
        "thread_counts",
        "trials",
        "warmup_runs",
        "repetitions",
    ]

    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Falta el campo obligatorio '{field}' en {CONFIG_PATH}"
            )

    return config


def load_input_batch(batch_size):
    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=transform,
    )

    if batch_size > len(test_dataset):
        raise ValueError(
            f"batch_size={batch_size} es mayor que el número "
            f"de muestras disponibles ({len(test_dataset)})."
        )

    images = [
        test_dataset[index][0].numpy()
        for index in range(batch_size)
    ]

    batch = np.stack(images, axis=0)

    return batch.astype(np.float32)


def create_session(num_threads):
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = 1

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
    input_feed = {
        "input": input_batch,
    }

    # Primera inferencia
    start = time.perf_counter_ns()

    session.run(
        ["logits"],
        input_feed,
    )

    end = time.perf_counter_ns()

    first_run_ms = (end - start) / 1_000_000

    # Warm-up
    for _ in range(warmup_runs):
        session.run(
            ["logits"],
            input_feed,
        )

    # Régimen estable
    latencies_ms = []

    for _ in range(repetitions):
        start = time.perf_counter_ns()

        session.run(
            ["logits"],
            input_feed,
        )

        end = time.perf_counter_ns()

        latency_ms = (end - start) / 1_000_000

        latencies_ms.append(latency_ms)

    return first_run_ms, np.array(latencies_ms)


def get_process_memory_mb():
    process = psutil.Process()

    memory_bytes = process.memory_info().rss

    return memory_bytes / (1024 * 1024)


def get_model_size_mb():
    model_size_bytes = ONNX_MODEL_PATH.stat().st_size

    return model_size_bytes / (1024 * 1024)


def run_benchmark(
    batch_size,
    num_threads,
    warmup_runs,
    repetitions,
    trial,
):
    input_batch = load_input_batch(batch_size)

    session = create_session(num_threads)

    first_run_ms, latencies_ms = measure_latency(
        session=session,
        input_batch=input_batch,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))

    throughput = (
        batch_size
        / (mean_ms / 1000.0)
    )

    memory_rss_mb = get_process_memory_mb()
    model_size_mb = get_model_size_mb()

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

    configurations = sorted(
        {
            (result["batch_size"], result["threads"])
            for result in results
        }
    )

    for batch_size, threads in configurations:
        config_results = [
            result
            for result in results
            if result["batch_size"] == batch_size
            and result["threads"] == threads
        ]

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

        summary.append(
            {
                "batch_size": batch_size,
                "threads": threads,
                "trials": len(config_results),
                "first_run_mean_ms": float(np.mean(first_runs)),
                "latency_mean_ms": float(np.mean(means)),
                "latency_mean_std_ms": float(np.std(means)),
                "median_mean_ms": float(np.mean(medians)),
                "p95_mean_ms": float(np.mean(p95_values)),
                "throughput_mean_images_s": float(
                    np.mean(throughputs)
                ),
                "memory_rss_mean_mb": float(
                    np.mean(memory_values)
                ),
                "model_size_mb": config_results[0]["model_size_mb"],
            }
        )

    return summary


def save_csv(path, rows):
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

    batch_sizes = config["batch_sizes"]
    thread_counts = config["thread_counts"]
    trials = config["trials"]
    warmup_runs = config["warmup_runs"]
    repetitions = config["repetitions"]

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

    summary = build_summary(results)

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