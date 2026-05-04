# Entrega final del laboratorio

Este archivo resume todo lo entregado y lo que falta para tener la version publica final.

## 1. Codigo fuente

### Exploracion y preprocesamiento

- [data.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab/data.py)
- [face_detection.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab/face_detection.py)
- [train.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/scripts/train.py)

### Modelado y entrenamiento

- [modeling.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab/modeling.py)
- [training.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab/training.py)

### Interpretabilidad

- [xai.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab/xai.py)
- [predict.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/scripts/predict.py)

### Aplicacion

- [streamlit_app.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/streamlit_app.py)

## 2. Documentacion

- [README.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/README.md)
- [01_guia_laboratorio.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/01_guia_laboratorio.md)
- [03_entregable_manuscrito.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/03_entregable_manuscrito.md)
- [04_despliegue_streamlit_cloud.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/04_despliegue_streamlit_cloud.md)
- [05_guion_presentacion.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/05_guion_presentacion.md)
- [06_checklist_entrega.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/06_checklist_entrega.md)
- [10_auditoria_modelo_y_datos.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/10_auditoria_modelo_y_datos.md)

## 3. Artefactos de resultados

- [dataset_summary_raw.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_summary_raw.json)
- [dataset_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_summary.json)
- [split_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/split_summary.json)
- [deduplication_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/deduplication_summary.json)
- [prepared_face_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/prepared_face_summary.json)
- [dataset_mosaic.png](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_mosaic.png)
- [comparison.csv](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/comparison.csv)
- [best_experiment.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/best_experiment.json)
- [correct_prediction_panel.png](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/interpretability/correct_prediction_panel.png)
- [cnn_architecture_manim.mp4](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/animations/cnn_architecture_manim.mp4)

## 4. Estado tecnico actual

- La app funciona localmente.
- La documentacion ya esta completa y alineada con el proyecto actual.
- El repo ya esta preparado para publicarse con `Git LFS`.
- La demo usa por defecto un modo `Robusto (ensemble)` para reducir errores extremos en imagenes externas.

## 5. Lo unico que falta para tener el link publico

1. Crear o elegir un repositorio de GitHub solo para `cnn_xai_gender_lab`.
2. Subir `models/model.keras` con `Git LFS`.
3. Conectar el repo a `Streamlit Community Cloud`.
4. Desplegar `streamlit_app.py`.
5. Copiar la URL publica final.

## 6. Donde registrar el enlace final

Cuando la app quede publicada, pega el link en:

- [03_entregable_manuscrito.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/03_entregable_manuscrito.md)
- este mismo archivo

Campos por completar:

- URL del repositorio: `Pendiente`
- URL de la app: `Pendiente`
- Estudiante: `Pendiente`
- Curso: `Pendiente`
