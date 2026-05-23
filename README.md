# Proyecto de Modelado de Deserción y Rendimiento Estudiantil (Kedro)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Descripción General

Este proyecto utiliza la arquitectura avanzada **Kedro** para la orquestación y desarrollo de modelos de Machine Learning. Dado que el curso requiere una estructura de carpetas específica para la evaluación, se ha realizado un mapeo ("wrapper") para cumplir con los requerimientos de entrega sin comprometer la reproducibilidad técnica de Kedro.

## Mapeo de la Rúbrica (Estructura Solicitada)

A continuación, se detalla dónde encontrar los entregables solicitados:

### 1. Notebooks (`notebooks/`)
Los notebooks principales solicitados se encuentran numerados del 01 al 05 en la raíz de la carpeta `notebooks/`:
- `01_exploratory_analysis.ipynb`: Análisis exploratorio, visualizaciones y patrones.
- `02_supervised_modeling.ipynb`: Implementación de modelos base.
- `03_model_evaluation.ipynb`: Evaluación comparativa y métricas.
- `04_hyperparameter_optimization.ipynb`: Optimización de hiperparámetros.
- `05_final_analysis.ipynb`: Análisis final e integración.

> **Nota:** Los notebooks extra que detallan pruebas avanzadas y experimentación (`06_regression_modeling.ipynb` y `07_model_shootout.ipynb`) también se encuentran en esta carpeta como complemento a los requisitos. Archivos de pruebas y borradores se han movido a `notebooks/extras/`.

### 2. Código Fuente (`src/`)
El código real de los pipelines se ejecuta mediante Kedro, pero para facilitar la revisión y cumplir con la estructura requerida, se han expuesto las lógicas en los siguientes archivos (que actúan como origen de funciones para los nodos de Kedro):
- `src/data_preprocessing.py`: Funciones de limpieza, transformación y tabla maestra.
- `src/model_training.py`: Definición, partición (split) y entrenamiento de modelos.
- `src/model_evaluation.py`: Funciones para evaluación experta y reportes métricos.
- `src/hyperparameter_tuning.py`: Lógica para búsqueda en grilla y validación cruzada.

*El paquete completo de Kedro se encuentra dentro de `src/ev_parcial1_uribe/`.*

### 3. Modelos y Resultados
Por diseño, Kedro gestiona las salidas a través de su Data Catalog (`conf/base/catalog.yml`). Se han reconfigurado las rutas para apuntar a:
- **Modelos:** `models/trained_models/` (contiene los `.pkl` finales).
- **Resultados y Reportes:** `results/reports/`, `results/metrics/` y `results/plots/`.

---

## Cómo Ejecutar el Proyecto

Este proyecto mantiene la compatibilidad total con Kedro.

### 1. Instalar Dependencias
Asegúrate de contar con el entorno activado e instalar los requerimientos:
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline Completo
Para correr todo el flujo de datos, limpieza, entrenamiento y reporte, utiliza:
```bash
kedro run
```
Todos los modelos y reportes se guardarán automáticamente en sus carpetas respectivas configuradas.

### 3. Ejecutar Jupyter
Para abrir la interfaz clásica con las variables de Kedro cargadas (`catalog`, `context`, etc.):
```bash
kedro jupyter lab
```
