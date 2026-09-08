# Prueba técnica — AI Performance Engineer

**Nivel:** Junior  
**Duración estimada:** 4 h  
**Entorno mínimo:** ordenador convencional con CPU moderna y memoria suficiente para desarrollo local  
**GPU:** opcional; no necesaria para completar la prueba  
**Hardware edge específico:** no requerido

---

## 1. Reto

### Benchmark reproducible de un clasificador ligero en CPU

Se dispone de un modelo de clasificación que deberá ejecutarse en un equipo edge de propósito general. Antes de plantear optimizaciones es necesario establecer una línea base fiable que permita saber cuánto tarda realmente la inferencia, qué variabilidad existe y cómo influyen el tamaño de lote y el número de hilos.

La prueba es autocontenida: La solución deberá quedar preparada para que otra persona pueda reproducirla a partir de la entrega.

---

## 2. Dataset obligatorio

**Fashion-MNIST** — 70.000 imágenes en escala de grises de 28×28 píxeles, 10 clases, con partición oficial de entrenamiento y prueba.

- Fuente oficial: https://github.com/zalandoresearch/fashion-mnist

---

## 3. Modelo de trabajo

Se entrenará desde cero una CNN pequeña para Fashion-MNIST. La red deberá ser lo bastante compacta como para entrenarse rápidamente en CPU. La evaluación de rendimiento se realizará sobre una versión exportada a ONNX y ejecutada con ONNX Runtime CPU.

### Herramientas y licencias admitidas

La solución deberá apoyarse únicamente en herramientas de código abierto y dependencias cuya licencia permita su uso y redistribución en un contexto técnico ordinario. Como referencia son válidos PyTorch (proyecto principal BSD-3-Clause), scikit-learn (BSD-3-Clause), ONNX y ONNX Runtime (MIT), NumPy, SciPy, pandas, psutil y herramientas equivalentes de licencia permisiva.

Cuando la prueba requiera un modelo, deberá entrenarse desde cero durante la propia prueba con el dataset indicado. No se utilizarán pesos preentrenados descargados de terceros. El objetivo no es maximizar la precisión sino disponer de un artefacto controlado, reproducible y sencillo de desplegar.

La ejecución principal y todas las métricas deberán poder obtenerse en CPU. Una GPU podrá utilizarse de forma adicional, pero los resultados acelerados se presentarán siempre por separado y nunca sustituirán a la validación en CPU.

---

## 4. Trabajo a realizar

1. Preparar una línea base funcional con una métrica de calidad sobre el conjunto oficial de prueba.
2. Exportar el modelo y comprobar que las salidas son coherentes con el framework de entrenamiento.
3. Diseñar un benchmark reproducible que separe primera ejecución y régimen estable y que utilice calentamiento y un número suficiente de repeticiones.
4. Medir latencia media, mediana, p95 y throughput para al menos dos tamaños de lote razonables.
5. Comparar al menos dos configuraciones de hilos de CPU manteniendo el resto de condiciones.
6. Registrar memoria aproximada del proceso y tamaño del artefacto.
7. Entregar una recomendación sencilla para el caso de uso interactivo y otra para procesamiento por lotes, basadas en los datos obtenidos.

---

## 5. Requisitos y restricciones

- No se compararán configuraciones distintas sin declarar todos los cambios relevantes.
- La medición no incluirá entrenamiento.
- Los resultados GPU, si se aportan, serán únicamente anexos.
- El benchmark deberá poder repetirse sin editar manualmente el código entre configuraciones.

Además:

- la ejecución principal deberá completarse en CPU;
- las decisiones relevantes deberán quedar justificadas con evidencia o con una hipótesis explícita;
- los fallos esperables deberán producir mensajes útiles y no excepciones silenciosas;
- cualquier resultado no reproducible en la máquina de evaluación deberá identificarse como limitación y no presentarse como hecho medido.

---

## 6. Entregables

La entrega deberá incluir, como mínimo:

- un `README.md` con requisitos, preparación del entorno, ejecución y reproducción de los resultados;
- dependencias y versiones fijadas;
- código fuente organizado y ejecutable sin depender de un notebook interactivo;
- configuración separada del código cuando existan parámetros relevantes;
- resultados medidos en CSV, JSON o formato equivalente legible por máquina;
- un informe técnico breve en Markdown, con las decisiones, resultados, limitaciones y recomendación final;
- los artefactos de modelo generados durante la prueba cuando su tamaño lo permita.

Se permite utilizar notebooks para exploración, pero la entrega final deberá poder ejecutarse de forma reproducible sin intervención manual sobre celdas.

---

## 7. Fuera de alcance

Quedan fuera de alcance el entrenamiento prolongado, la búsqueda exhaustiva de hiperparámetros, el uso de hardware edge específico, la creación de una interfaz gráfica, la integración con servicios cloud y cualquier optimización que dependa obligatoriamente de CUDA. No se espera una solución de producción completa; sí una implementación suficientemente sólida para comprobar el criterio técnico evaluado.
