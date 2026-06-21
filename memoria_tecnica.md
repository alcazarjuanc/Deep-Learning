# Memoria técnica — Práctica Final Deep Learning

## Predicción de Engagement en POIs Turísticos

**Autor:** Juan Carlos San Martin Alcazar

---

## 1. Objetivo del proyecto

El objetivo de esta práctica fue desarrollar un modelo de Deep Learning capaz de predecir si un punto de interés turístico, también llamado POI, tendría un nivel de engagement bajo o alto.

Para resolver el problema se trabajó con dos fuentes de información:

* Imágenes principales de cada POI.
* Metadatos estructurados asociados a cada POI.

El proyecto se abordó como un problema de clasificación binaria. La clase `0` representa engagement bajo y la clase `1` representa engagement alto.

---

## 2. Descripción del dataset

El dataset contiene información de puntos de interés turísticos. Cada fila incluye metadatos del POI, métricas de interacción y una ruta hacia su imagen principal.

Las variables principales utilizadas fueron:

* `id`: identificador del POI.
* `name`: nombre del POI.
* `shortDescription`: descripción breve.
* `locationLon` y `locationLat`: coordenadas geográficas.
* `categories`: categorías asociadas al POI.
* `tags`: etiquetas descriptivas.
* `xps`: experiencia asociada al POI.
* `Visits`: número de visitas.
* `Likes`: número de valoraciones positivas.
* `Dislikes`: número de valoraciones negativas.
* `Bookmarks`: número de veces añadido a favoritos.
* `main_image_path`: ruta de la imagen principal.

El dataset original tenía 1569 filas y 1492 POIs únicos. Se identificaron 77 filas duplicadas por `id`, por lo que se realizó una agregación para trabajar a nivel de POI único.

---

## 3. Preparación y análisis de datos

Primero se cargó el archivo `poi_dataset.csv` y se verificó la existencia de las imágenes en la carpeta `data_main/`.

Posteriormente, se revisaron:

* Tipos de datos.
* Valores faltantes.
* IDs duplicados.
* Distribución de las métricas originales de engagement.
* Existencia de imágenes asociadas a cada POI.

Para evitar duplicados, el dataset se agrupó por `id`. En esta agregación se conservaron los primeros valores descriptivos y se promediaron las métricas numéricas de interacción.

También se procesaron las columnas `categories` y `tags`, convirtiéndolas de texto a listas reales de Python. Después se crearon variables derivadas:

* `num_categories`
* `num_tags`

Estas variables permiten representar de forma numérica parte de la información categórica del POI.

---

## 4. Construcción de la variable objetivo

El dataset no incluía una etiqueta directa de éxito o engagement alto/bajo. Por eso se construyó una métrica propia de engagement a partir de las variables disponibles.

La fórmula utilizada fue:

```text
engagement_score =
0.25 * log1p(Visits)
+ 2.00 * log1p(Likes)
+ 2.00 * log1p(Bookmarks)
- 1.00 * log1p(Dislikes)
```

Esta fórmula asigna mayor peso a `Likes` y `Bookmarks`, ya que representan interacciones positivas más explícitas. `Visits` tiene un peso menor porque visitar un POI no necesariamente implica una valoración positiva. `Dislikes` se resta porque representa una señal negativa.

Después se usó la mediana de `engagement_score` como umbral para crear la variable binaria:

* `0`: engagement bajo.
* `1`: engagement alto.

El resultado fue un dataset balanceado:

* Clase 0: 746 POIs.
* Clase 1: 746 POIs.

---

## 5. División train / validation / test

El dataset se dividió en tres subconjuntos:

* Entrenamiento: 1044 muestras.
* Validación: 224 muestras.
* Test: 224 muestras.

La división se realizó de forma estratificada para mantener la proporción entre las clases de engagement bajo y alto.

También se construyeron matrices de metadatos usando:

* Variables numéricas: `locationLon`, `locationLat`, `xps`, `num_categories`, `num_tags`.
* Variables categóricas codificadas a partir de `categories`.

No se usaron como variables predictoras `Visits`, `Likes`, `Dislikes`, `Bookmarks` ni `engagement_score`, porque forman parte de la construcción de la etiqueta objetivo. Esto evita data leakage.

---

## 6. Modelos desarrollados

Se entrenaron y compararon cuatro enfoques.

### 6.1 Baseline

El baseline predice siempre la clase mayoritaria. Como las clases están balanceadas, sirve como punto mínimo de comparación.

### 6.2 Modelo solo metadatos

Se implementó una red fully connected usando únicamente las variables estructuradas del dataset.

Este modelo permite evaluar cuánto poder predictivo tienen los metadatos sin utilizar imágenes.

### 6.3 Modelo solo imágenes CNN

Se implementó una red neuronal convolucional para clasificar los POIs usando únicamente la imagen principal.

Este modelo permite evaluar si las imágenes contienen información suficiente para diferenciar entre engagement bajo y alto.

### 6.4 Modelo híbrido CNN + metadatos

Finalmente, se desarrolló un modelo híbrido con dos ramas:

* Una rama CNN para procesar imágenes.
* Una rama fully connected para procesar metadatos.

Después, ambas representaciones se fusionan y pasan por capas densas finales para realizar la clasificación binaria.

Este fue el modelo principal de la práctica, ya que combina información visual y estructurada.

---

## 7. Entrenamiento

El entrenamiento se realizó con PyTorch.

Se utilizó:

* Función de pérdida: `CrossEntropyLoss`.
* Optimizador: `Adam`.
* Semilla aleatoria fija para reproducibilidad.
* Validación durante el entrenamiento.
* Dropout como técnica básica para reducir sobreajuste.
* Selección del mejor modelo según el rendimiento en validación.

El notebook detecta automáticamente si existe GPU disponible:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

En este proyecto el entrenamiento se realizó usando una GPU NVIDIA GeForce RTX 4080 SUPER. Si no hay GPU disponible, el código puede ejecutarse en CPU.

---

## 8. Evaluación de resultados

Los modelos se evaluaron en el conjunto de test usando accuracy, precision, recall y F1-score.

| Modelo                     | Accuracy | Precision | Recall | F1-score |
| -------------------------- | -------: | --------: | -----: | -------: |
| Baseline clase mayoritaria |   0.5000 |    0.0000 | 0.0000 |   0.0000 |
| Solo metadatos             |   0.7991 |    0.7190 | 0.9821 |   0.8302 |
| Solo imágenes CNN          |   0.7589 |    0.7544 | 0.7679 |   0.7611 |
| Híbrido CNN + metadatos    |   0.9196 |    1.0000 | 0.8393 |   0.9126 |

El baseline obtuvo un accuracy de 0.5000, lo esperado en un problema balanceado. No es un modelo útil, pero funciona como referencia mínima.

El modelo solo metadatos obtuvo un buen rendimiento, con un F1-score de 0.8302. Su recall fue muy alto, lo que indica que detectó casi todos los POIs de engagement alto. Sin embargo, su precision fue menor, por lo que también generó algunos falsos positivos.

El modelo solo imágenes obtuvo un F1-score de 0.7611. Esto demuestra que las imágenes sí aportan información relevante, aunque no fueron suficientes para superar al modelo de metadatos.

El modelo híbrido obtuvo el mejor resultado, con un accuracy de 0.9196 y un F1-score de 0.9126. Además, logró una precision de 1.0000, lo que indica que todos los POIs clasificados como engagement alto en test fueron correctamente identificados.

---

## 9. Modelo final seleccionado

El modelo final seleccionado fue el modelo híbrido CNN + metadatos.

Se seleccionó porque obtuvo el mejor rendimiento general y superó tanto al modelo solo imágenes como al modelo solo metadatos.

El archivo del modelo final se guardó en:

```text
models/hybrid_cnn_metadata_model.pt
```

Este modelo representa la mejor solución desarrollada en la práctica porque integra dos tipos de información complementaria:

* La imagen del POI.
* Los metadatos estructurados del POI.

---

## 10. Limitaciones y mejoras futuras

La principal limitación del proyecto es que la variable objetivo fue construida de forma sintética. Es decir, no existía una etiqueta directa de engagement alto o bajo, por lo que el modelo aprende a predecir la definición de engagement propuesta en esta práctica.

Otra limitación es que no se utilizó el texto de `shortDescription` como entrada del modelo. Esta información podría aportar contexto semántico adicional.

Como mejoras futuras se podrían considerar:

* Probar otras fórmulas de engagement.
* Ajustar hiperparámetros del modelo.
* Incorporar modelos preentrenados para imágenes.
* Usar data augmentation más amplio.
* Incluir información textual de las descripciones.
* Realizar análisis de errores con falsos positivos y falsos negativos.
* Probar técnicas adicionales de regularización.

---

## 11. Conclusión

La práctica permitió desarrollar un pipeline completo de Deep Learning para clasificación binaria de engagement en POIs turísticos.

Los resultados muestran que los metadatos tienen un alto poder predictivo y que las imágenes también aportan información útil. Sin embargo, el mejor rendimiento se obtuvo al combinar ambas fuentes de información en un modelo híbrido.

El modelo híbrido CNN + metadatos fue la solución más efectiva, alcanzando un F1-score de 0.9126 en test. Esto confirma que la integración de información visual y estructurada mejora la capacidad predictiva frente a modelos unimodales.
