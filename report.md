# Informe técnico — AI Performance Engineer

## 1. Objetivo y decisiones

El objetivo de la prueba es establecer una línea base reproducible del rendimiento de inferencia de una CNN ligera sobre CPU.

El modelo se entrena desde cero sobre Fashion-MNIST utilizando PyTorch, se exporta posteriormente a ONNX y se ejecuta mediante ONNX Runtime forzando `CPUExecutionProvider`.

Las principales decisiones tomadas fueron:

- utilizar una CNN compacta de **105.866 parámetros**;
- mantener la partición oficial de Fashion-MNIST: 60.000 imágenes de entrenamiento y 10.000 de test;
- utilizar el conjunto oficial de test únicamente para la evaluación final;
- exportar el modelo con batch dinámico para poder reutilizar el mismo artefacto ONNX;
- comparar batch sizes `1` y `32`;
- comparar `1` y `4` hilos mediante `intra_op_num_threads`;
- medir la primera inferencia por separado;
- ejecutar **200 warm-ups** antes de las mediciones estables;
- realizar **2.000 repeticiones por trial**;
- repetir cada configuración durante **3 trials**.

El benchmark mide únicamente la llamada `session.run()` con la entrada ya preparada en memoria, por lo que los resultados representan latencia de inferencia y no latencia end-to-end.

---

## 2. Calidad y validación ONNX

Tras tres épocas de entrenamiento, el modelo obtuvo sobre las 10.000 imágenes del conjunto oficial de test:

- **Accuracy: 88,38 %**
- Test loss: **0,3210**

La diferencia entre la accuracy final de entrenamiento y test fue de aproximadamente **0,40 puntos porcentuales**, sin observarse una brecha elevada entre ambos valores en este entrenamiento.

El modelo se exportó correctamente a ONNX con:

- opset `20`;
- entrada: `[batch_size, 1, 28, 28]`;
- salida: `[batch_size, 10]`;
- tamaño del artefacto: aproximadamente **0,4200 MiB**.

La coherencia PyTorch ↔ ONNX Runtime se comprobó sobre un batch de 32 imágenes reales:

- máxima diferencia absoluta entre logits: **0,0000019073**;
- diferencia absoluta media: **0,0000004335**;
- predicciones coincidentes: **32/32**.

La validación mediante `numpy.testing.assert_allclose()` finalizó correctamente con `rtol=1e-4` y `atol=1e-5`.

---

## 3. Resultados de rendimiento

Entorno utilizado:

- CPU: **Intel Core i7-1065G7 @ 1.30 GHz**
- 4 núcleos físicos / 8 hilos lógicos
- RAM: aproximadamente **7,79 GiB**
- Python `3.11.15`
- ONNX Runtime `1.29.0`
- Provider: `CPUExecutionProvider`

Resultados agregados de los tres trials:

| Batch | Hilos | Latencia media | Desv. entre trials | p95 medio | Throughput medio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0,1008 ms | 0,0136 ms | 0,1924 ms | 10.122,69 img/s |
| 1 | 4 | **0,0778 ms** | **0,0078 ms** | **0,1241 ms** | **12.986,57 img/s** |
| 32 | 1 | 0,9752 ms | 0,2107 ms | 1,5925 ms | 34.222,86 img/s |
| 32 | 4 | **0,3827 ms** | **0,0209 ms** | **0,4872 ms** | **83.851,26 img/s** |

La memoria RSS observada durante los trials se situó aproximadamente entre **303 y 308 MiB**. Esta cifra corresponde al proceso Python completo y no al modelo de forma aislada.

Las medias de la primera inferencia fueron aproximadamente:

| Batch | Hilos | First run media |
|---:|---:|---:|
| 1 | 1 | 0,7084 ms |
| 1 | 4 | 0,2952 ms |
| 32 | 1 | 2,3046 ms |
| 32 | 4 | 0,8294 ms |

La primera ejecución presenta un coste claramente superior al régimen estable, lo que justifica medirla por separado y utilizar warm-up antes de calcular las estadísticas finales.

---

## 4. Análisis

### Efecto del número de hilos

Para batch 1, pasar de uno a cuatro hilos produjo aproximadamente:

- **22,8 % menos latencia media**;
- **35,5 % menos p95**;
- **28,3 % más throughput**.

Para batch 32, la mejora fue mayor:

- **60,8 % menos latencia media**;
- **69,4 % menos p95**;
- **145,0 % más throughput**.

En este hardware, cuatro hilos resultaron mejores que uno en todas las métricas principales evaluadas.

El beneficio observado fue especialmente elevado con batch 32. Una hipótesis razonable es que el batch mayor proporciona suficiente trabajo computacional para aprovechar mejor el paralelismo y amortizar su overhead. No obstante, el benchmark no incluye profiling interno suficiente para demostrar el mecanismo exacto.

### Efecto del tamaño de lote

Con cuatro hilos:

- batch 1: `0,0778 ms` y `12.986,57 img/s`;
- batch 32: `0,3827 ms` y `83.851,26 img/s`.

Batch 32 presenta aproximadamente **4,92 veces más latencia por llamada**, pero obtiene aproximadamente **6,46 veces más throughput**.

Esto refleja el trade-off principal del benchmark:

- batches pequeños favorecen la respuesta de entradas individuales;
- batches mayores permiten procesar muchas más muestras por segundo.

### Variabilidad

Los resultados muestran variabilidad entre trials, especialmente con batch 32 y un único hilo.

El modelo presenta tiempos de inferencia extremadamente bajos, por lo que interrupciones del sistema operativo, scheduler, frecuencia dinámica de CPU y tareas en segundo plano pueden tener un peso relativo importante.

Por este motivo se utilizan múltiples warm-ups, 2.000 repeticiones y tres trials por configuración.

---

## 5. Limitaciones

Las principales limitaciones del benchmark son:

- todas las mediciones se realizaron sobre una única CPU;
- se utilizó Windows sin aislamiento específico de núcleos ni control de frecuencia;
- únicamente se evaluaron batch sizes `1` y `32`;
- únicamente se compararon `1` y `4` hilos;
- la temporización mide inferencia aislada, no el pipeline completo;
- la memoria RSS representa el proceso Python completo;
- los trials se ejecutan dentro del mismo proceso;
- no se realizaron cuantización, pruning, distillation ni optimizaciones adicionales;
- la validación PyTorch ↔ ONNX se realizó sobre un batch de 32 imágenes y no sobre todo el conjunto de test.

Los resultados deben interpretarse como una línea base del entorno evaluado y no como propiedades universales del modelo.

---

## 6. Recomendación final

### Uso interactivo

Para un escenario donde las entradas llegan individualmente y se prioriza el tiempo de respuesta, la mejor configuración entre las evaluadas es:

`batch_size = 1`

`intra_op_num_threads = 4`

Resultados:

- latencia media: **0,0778 ms**;
- p95 medio: **0,1241 ms**;
- throughput: **12.986,57 img/s**.

### Procesamiento por lotes

Para un escenario donde existen múltiples entradas disponibles y se prioriza throughput, la mejor configuración entre las evaluadas es:

`batch_size = 32`

`intra_op_num_threads = 4`

Resultados:

- latencia media por llamada: **0,3827 ms**;
- p95 medio: **0,4872 ms**;
- throughput: **83.851,26 img/s**.

Por tanto, no existe una única configuración óptima independientemente del caso de uso. La elección debe realizarse en función de si el sistema prioriza latencia individual o capacidad total de procesamiento, y debe validarse siempre sobre el hardware objetivo.