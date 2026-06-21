# Práctica Final: Deep Learning

**Autor:** Juan Carlos San Martin Alcazar

## Predicción de Engagement en POIs Turísticos

Este repositorio contiene la práctica final del módulo de Deep Learning.  
El objetivo del proyecto es desarrollar un modelo capaz de clasificar puntos de interés turísticos como POIs de **engagement bajo** o **engagement alto**, utilizando imágenes y metadatos estructurados.

## Archivos principales

```text
Deep-Learning/
│
├── README.md
├── requirements.txt
├── Practica_DL_Prediccion_Engagement_POIs.ipynb
│
├── poi_dataset.csv
├── data_main/
│
├── models/
│   └── hybrid_cnn_metadata_model.pt
│
├── outputs/
│   ├── poi_dataset_model_ready.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── results_summary.csv
│
└── src/
    ├── preprocessing.py
    ├── dataset.py
    └── models.py
```

## Entorno de ejecución

El proyecto fue desarrollado con Python 3.11.15 y PyTorch con soporte CUDA.

Versiones principales utilizadas:

- Python: 3.11.15
- NumPy: 2.4.6
- Pandas: 3.0.3
- Matplotlib: 3.11.0
- Scikit-learn: 1.9.0
- Pillow: 12.2.0
- Jupyter: 1.1.1
- IPython kernel: 7.3.0
- PyTorch: 2.12.1+cu126
- Torchvision: 0.27.1+cu126

El entrenamiento se realizó usando una GPU NVIDIA GeForce RTX 4080 SUPER.

Sin embargo, el notebook verifica automáticamente si hay GPU disponible:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Si CUDA está disponible, el modelo se entrena en GPU.  
Si no, el código se ejecuta en CPU.

## Instalación

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Ejecución

Abrir el notebook principal:

```text
Practica_DL_Prediccion_Engagement_POIs.ipynb
```

Ejecutar las celdas en orden desde el inicio hasta el final.

## Dataset

El proyecto utiliza:

```text
poi_dataset.csv
data_main/
```

El archivo `poi_dataset.csv` contiene los metadatos de los POIs.  
La carpeta `data_main/` contiene las imágenes principales asociadas a cada POI.

## Modelo final

El modelo final entrenado se encuentra en:

```text
models/hybrid_cnn_metadata_model.pt
```

El modelo seleccionado fue un modelo híbrido que combina:

- Rama CNN para procesamiento de imágenes.
- Rama fully connected para metadatos.
- Fusión de ambas representaciones para clasificación binaria.

## Resultados finales

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline clase mayoritaria | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Solo metadatos | 0.7991 | 0.7190 | 0.9821 | 0.8302 |
| Solo imágenes CNN | 0.7589 | 0.7544 | 0.7679 | 0.7611 |
| Híbrido CNN + metadatos | 0.9196 | 1.0000 | 0.8393 | 0.9126 |

El mejor rendimiento se obtuvo con el modelo híbrido CNN + metadatos.

## Conclusión

La combinación de imágenes y metadatos fue la estrategia más efectiva para predecir el engagement de los POIs turísticos.  
Los resultados muestran que ambas fuentes de información aportan señales complementarias y que el modelo híbrido supera claramente al baseline y a los modelos unimodales.