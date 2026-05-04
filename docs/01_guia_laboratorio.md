# Guia de resolucion del laboratorio

Este proyecto esta organizado para responder, uno por uno, los puntos del PDF.

## Ejercicio 1. Descarga y exploracion del dataset

### Objetivo

- Descargar el dataset.
- Organizarlo en `data/male/` y `data/female/`.
- Contar imagenes.
- Revisar tamanos y formatos.
- Generar un mosaico representativo.

### Archivos que resuelven este punto

- `src/cnn_xai_lab/data.py`
- `scripts/train.py`

### Artefactos generados

- `outputs/dataset_summary.json`
- `outputs/dataset_mosaic.png`

## Ejercicio 2. Preprocesamiento y particion

### Objetivo

- Mantener RGB.
- Redimensionar a `224x224`.
- Normalizar a `[0, 1]`.
- Dividir en `70/15/15`.

### Archivos que resuelven este punto

- `src/cnn_xai_lab/data.py`

### Artefactos generados

- `outputs/split_summary.json`

## Ejercicio 3. Construccion y entrenamiento de la CNN

### Objetivo

- Construir una CNN secuencial desde cero.
- Usar `Conv2D + MaxPooling2D`.
- Usar `ReLU` y salida `sigmoid`.
- Compilar con `binary_crossentropy` y `Adam`.
- Guardar el modelo final.

### Archivos que resuelven este punto

- `src/cnn_xai_lab/modeling.py`
- `src/cnn_xai_lab/training.py`
- `scripts/train.py`

### Artefactos generados

- `models/model.keras`
- `outputs/experiments/baseline/history.csv`
- `outputs/experiments/baseline/training_curves.png`

## Ejercicio 4. Ajuste de hiperparametros

### Objetivo

- Probar al menos dos configuraciones.
- Comparar metricas finales.

### Archivos que resuelven este punto

- `src/cnn_xai_lab/training.py`
- `scripts/train.py`

### Artefactos generados

- `outputs/experiments/comparison.csv`
- `outputs/experiments/best_experiment.json`

## Ejercicio 5. Interpretabilidad visual

### Objetivo

- Aplicar `Saliency Map`.
- Aplicar `Grad-CAM`.
- Superponer los mapas sobre la imagen.
- Interpretar las regiones relevantes.

### Archivos que resuelven este punto

- `src/cnn_xai_lab/xai.py`
- `scripts/train.py`
- `scripts/predict.py`

### Artefactos generados

- `outputs/interpretability/correct_prediction_panel.png`
- `outputs/interpretability/correct_prediction_metadata.json`

## Ejercicio 6. Despliegue con Streamlit

### Objetivo

- Permitir carga de imagen.
- Mostrar imagen, prediccion y probabilidades.
- Visualizar `Saliency Map` y `Grad-CAM`.
- Desplegar en `Streamlit Community Cloud`.

### Archivos que resuelven este punto

- `streamlit_app.py`
- `.streamlit/config.toml`
- `requirements.txt`

## Ejercicio 7. Presentacion y reflexion final

### Objetivo

- Mostrar la app funcionando.
- Explicar la prediccion.
- Interpretar los mapas visuales.

### Apoyo disponible en este repo

- `docs/03_entregable_manuscrito.md`
- `docs/04_despliegue_streamlit_cloud.md`
- `docs/05_guion_presentacion.md`
- `docs/06_checklist_entrega.md`
