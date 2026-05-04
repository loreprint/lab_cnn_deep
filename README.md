# Laboratorio CNNs-XAI

Resolucion completa del laboratorio `Laboratorio_XAI_CNNs.pdf` con:

- `TensorFlow/Keras` para entrenamiento de CNNs.
- `Streamlit` para la aplicacion interactiva.
- `Saliency Map` y `Grad-CAM` para interpretabilidad.
- `Streamlit Community Cloud` como destino de despliegue publico.

## Que incluye el proyecto

- Exploracion y auditoria del dataset.
- Preprocesamiento reproducible.
- Entrenamiento de varios experimentos CNN.
- Analisis de hiperparametros.
- App web con seccion de informe y demo interactiva.
- Documentacion lista para entrega.

## Estado actual del laboratorio

- Dataset crudo detectado: `5418` imagenes.
- Imagenes unicas tras deduplicacion visual: `1672`.
- Split actual:
  - `train`: `3788`
  - `val`: `251`
  - `test`: `251`
- Mejor checkpoint individual guardado para la demo: `wider_k5_dropout`.
- Modo recomendado en la app: `Robusto (ensemble)`.

Importante:

- Las metricas muy altas del checkpoint `wider_k5_dropout` provienen de una corrida legada anterior a la auditoria completa.
- Las metricas metodologicamente mas honestas son las de los experimentos corregidos sin fuga entre grupos duplicados.
- La app publica conserva ambos niveles:
  - informe tecnico del laboratorio;
  - demo robusta para que el comportamiento frente a imagenes externas sea menos fragil.

## Estructura

```text
cnn_xai_gender_lab/
|-- .streamlit/config.toml
|-- data/
|-- docs/
|-- models/
|-- notebooks/
|-- outputs/
|-- scripts/
|-- src/
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- requirements-dev.txt
`-- streamlit_app.py
```

## Documentos principales

- Entregable manuscrito: [03_entregable_manuscrito.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/03_entregable_manuscrito.md)
- Guia de despliegue: [04_despliegue_streamlit_cloud.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/04_despliegue_streamlit_cloud.md)
- Checklist final: [06_checklist_entrega.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/06_checklist_entrega.md)
- Resumen de entrega: [07_entrega_final.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/07_entrega_final.md)
- Auditoria tecnica: [10_auditoria_modelo_y_datos.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/10_auditoria_modelo_y_datos.md)

## Instalacion local

```powershell
cd C:\Users\lore7\Documents\Playground\cnn_xai_gender_lab
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si necesitas regenerar la animacion de arquitectura con `Manim`:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Dataset soportado

Se soportan dos estructuras:

```text
data/
|-- female/
`-- male/
```

o la estructura original de Kaggle:

```text
data/
`-- Male and Female face dataset/
    |-- Female Faces/
    `-- Male Faces/
```

## Entrenamiento

Corrida base:

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --data-dir data --epochs 10 --batch-size 32
```

Corrida enfocada solo en rostros:

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --data-dir data --epochs 4 --batch-size 64 --experiment face_crop_k5
```

## Ejecutar la app

```powershell
& .\.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

Recomendacion de prueba:

- `Modo de inferencia`: `Robusto (ensemble)`
- `Region para inferencia`: `Auto: detectar rostro`

## Archivos que usa la app publicada

- `models/model.keras`
- `outputs/experiments/best_experiment.json`
- `outputs/experiments/comparison.csv`
- `outputs/experiments/*/metrics.json`
- `outputs/experiments/*/training_curves.png`
- `outputs/dataset_summary.json`
- `outputs/dataset_summary_raw.json`
- `outputs/split_summary.json`
- `outputs/dataset_mosaic.png`
- `outputs/interpretability/correct_prediction_panel.png`
- `outputs/animations/cnn_architecture_manim.mp4`

## Despliegue publico

Tu modelo `models/model.keras` pesa mas de `100 MB`, asi que GitHub normal lo bloquea. Por eso este proyecto ya quedo preparado para:

- subir `models/model.keras` con `Git LFS`;
- mantener fuera del repo los checkpoints pesados de experimentos;
- publicar la app en `Streamlit Community Cloud`.

Sigue la guia paso a paso en [04_despliegue_streamlit_cloud.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/04_despliegue_streamlit_cloud.md).

## Referencias oficiales usadas

- [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization for Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [App dependencies for Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [GitHub file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github?platform=windows)
