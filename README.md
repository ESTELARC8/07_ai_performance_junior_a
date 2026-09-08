# AI Performance Benchmark — Fashion-MNIST

Benchmark reproducible de una CNN ligera ejecutada con ONNX Runtime sobre CPU.

El proyecto entrena desde cero un clasificador sobre Fashion-MNIST, lo exporta a ONNX, valida la coherencia de las salidas frente a PyTorch y mide su rendimiento bajo diferentes configuraciones de batch size y número de hilos.

## Estructura

```text
07-ai-performance-junior-a/
│
├── configs/
│   └── benchmark.yaml
│
├── models/
│   ├── fashion_cnn.onnx
│   └── fashion_cnn.pth
│
├── results/
│   ├── benchmark_summary.csv
│   ├── benchmark_trials.csv
│   └── system_info.json
│
├── src/
│   ├── benchmark.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── model.py
│   ├── system_info.py
│   ├── train.py
│   └── validate_onnx.py
│
├── requirements.txt
├── requirements-lock.txt
├── report.md
└── README.md
```

## Requisitos

Entorno utilizado para generar los resultados incluidos:

- Python `3.11.15`
- PyTorch `2.14.0+cpu`
- Torchvision `0.29.0+cpu`
- NumPy `2.4.6`
- ONNX `1.22.0`
- ONNX Runtime `1.29.0`
- ONNX Script `0.7.1`
- psutil `7.2.2`
- PyYAML `6.0.3`

Las dependencias directas están definidas en:

`requirements.txt`

El archivo:

`requirements-lock.txt`

contiene el snapshot completo del entorno utilizado.

## Preparación del entorno

Se recomienda utilizar un entorno virtual independiente.

Con Conda:

```bash
conda create -n 07_ai_performance python=3.11 -y
conda activate 07_ai_performance
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

## Reproducción

Los comandos deben ejecutarse desde la raíz del proyecto.

### 1. Descargar Fashion-MNIST

```bash
python src/download_data.py
```

El dataset se descarga en:

```text
data/FashionMNIST/
```

Se utiliza la partición oficial:

- 60.000 imágenes de entrenamiento;
- 10.000 imágenes de test.

### 2. Entrenar el modelo

```bash
python src/train.py
```

Configuración utilizada:

```text
seed          = 42
batch size    = 64
epochs        = 3
learning rate = 0.001
optimizer     = Adam
device        = CPU
```

El modelo se guarda en:

```text
models/fashion_cnn.pth
```

### 3. Evaluar el modelo

```bash
python src/evaluate.py
```

Resultado obtenido en el entorno de referencia:

```text
Accuracy: 88.38%
```

### 4. Exportar a ONNX

```bash
python src/export_onnx.py
```

El modelo se exporta a:

```text
models/fashion_cnn.onnx
```

Características principales:

```text
Opset: 20
Input:  [batch_size, 1, 28, 28]
Output: [batch_size, 10]
```

El batch se declara dinámico y los parámetros se integran en un único archivo ONNX mediante:

```text
external_data=False
```

### 5. Validar PyTorch frente a ONNX Runtime

```bash
python src/validate_onnx.py
```

Resultado obtenido sobre un batch de 32 imágenes:

```text
Máxima diferencia absoluta: 0.0000019073
Diferencia absoluta media:  0.0000004335
Predicciones coincidentes:  32/32
```

La comparación utiliza:

```text
rtol = 1e-4
atol = 1e-5
```

### 6. Configurar el benchmark

La configuración se encuentra en:

```text
configs/benchmark.yaml
```

Contenido utilizado para los resultados incluidos:

```yaml
batch_sizes:
  - 1
  - 32

thread_counts:
  - 1
  - 4

trials: 3

warmup_runs: 200
repetitions: 2000
```

Se evalúan cuatro configuraciones:

```text
batch 1  / 1 hilo
batch 1  / 4 hilos
batch 32 / 1 hilo
batch 32 / 4 hilos
```

Cada trial contiene:

```text
1 primera inferencia medida por separado
200 inferencias de warm-up descartadas
2.000 inferencias utilizadas para estadísticas
```

### 7. Ejecutar el benchmark

```bash
python src/benchmark.py
```

El benchmark utiliza explícitamente:

```text
CPUExecutionProvider
```

y mide únicamente:

```text
session.run()
```

La lectura del dataset, el preprocesado y la preparación de la entrada quedan fuera de la región temporizada.

Los resultados se guardan en:

```text
results/benchmark_trials.csv
results/benchmark_summary.csv
```

### 8. Registrar información del sistema

```bash
python src/system_info.py
```

La información se guarda en:

```text
results/system_info.json
```

## Resultados de referencia

Hardware utilizado:

```text
CPU: Intel Core i7-1065G7 @ 1.30 GHz
Núcleos físicos: 4
Núcleos lógicos: 8
RAM: ~7.79 GiB
ONNX Runtime: 1.29.0
Provider: CPUExecutionProvider
```

Resultados agregados:

| Batch | Hilos | Latencia media | Desv. entre trials | p95 medio | Throughput medio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0,1008 ms | 0,0136 ms | 0,1924 ms | 10.122,69 img/s |
| 1 | 4 | **0,0778 ms** | **0,0078 ms** | **0,1241 ms** | **12.986,57 img/s** |
| 32 | 1 | 0,9752 ms | 0,2107 ms | 1,5925 ms | 34.222,86 img/s |
| 32 | 4 | **0,3827 ms** | **0,0209 ms** | **0,4872 ms** | **83.851,26 img/s** |

Tamaño del modelo ONNX:

```text
~0,4200 MiB
```

Memoria RSS observada:

```text
~303–308 MiB
```

La memoria corresponde al proceso Python completo y no al modelo ONNX de forma aislada.

## Recomendaciones

Para uso interactivo, entre las configuraciones evaluadas:

```text
batch_size = 1
intra_op_num_threads = 4
```

Resultado:

```text
Latencia media: 0,0778 ms
p95:            0,1241 ms
```

Para procesamiento por lotes:

```text
batch_size = 32
intra_op_num_threads = 4
```

Resultado:

```text
Latencia media: 0,3827 ms
Throughput:     83.851,26 img/s
```

Estas recomendaciones corresponden exclusivamente al hardware y configuraciones evaluados.

## Notas sobre reproducibilidad

Los resultados de rendimiento dependen del hardware y del estado del sistema.

Para reducir la variabilidad:

- la primera inferencia se mide por separado;
- se utilizan 200 ejecuciones de warm-up;
- se realizan 2.000 repeticiones por trial;
- cada configuración se ejecuta tres veces;
- las versiones de las dependencias quedan fijadas;
- el entorno hardware/software queda registrado.

Las cifras presentadas deben interpretarse como resultados del entorno descrito y no como valores universales del modelo.

Para el análisis técnico, las limitaciones y la interpretación de los resultados consultar:

`report.md`