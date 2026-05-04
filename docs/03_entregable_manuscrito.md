# Entregable manuscrito

Este documento deja redactado el contenido principal que pide el laboratorio. Puedes copiarlo a tu informe final y solo completar portada, nombre del curso, nombre del estudiante y enlaces publicos.

## Portada

- Titulo: `Laboratorio CNNs-XAI`
- Estudiante: `Completar`
- Curso: `Completar`
- Repositorio GitHub: `https://github.com/loreprint/lab_cnn_deep`
- App publica: `https://labcnndeep-loreuec.streamlit.app/`

## 1. Contexto del problema

El objetivo del laboratorio es construir una red convolucional desde cero para clasificar rostros en dos clases binarias: `female` y `male`, y despues interpretar sus decisiones con tecnicas XAI. El trabajo no se limita a entrenar un modelo: tambien busca entender si la red realmente aprende rasgos faciales o si depende de patrones espurios como el fondo, el encuadre o la composicion de la imagen.

La solucion final del proyecto integra:

- exploracion y auditoria del dataset;
- preprocesamiento reproducible;
- varios experimentos CNN;
- interpretabilidad con `Saliency Map` y `Grad-CAM`;
- una aplicacion en `Streamlit` con seccion explicativa y demo interactiva.

## 2. Descripcion del dataset

Se utilizo el conjunto `Male and Female Faces Dataset`, organizado en dos carpetas de clase:

- `female`
- `male`

En bruto se detectaron `5418` imagenes:

- `2698` de la clase `female`
- `2720` de la clase `male`

Esto indica un balance inicial relativamente cercano entre clases. Sin embargo, durante la auditoria del dataset se encontro un hallazgo importante: muchas imagenes eran duplicadas o casi duplicadas. Usando una huella visual compacta, solo `1672` imagenes resultaron efectivamente unicas, por lo que existian `3746` filas redundantes.

Despues de la deduplicacion visual, el conjunto unico quedo asi:

- `723` imagenes `female`
- `949` imagenes `male`

Ademas:

- formato dominante: `JPEG`
- formatos minoritarios: `PNG`
- ancho minimo: `194`
- ancho maximo: `8675`
- ancho promedio: `955.55`
- alto minimo: `232`
- alto maximo: `7680`
- alto promedio: `1199.92`

Estos datos justifican el uso de un pipeline de redimensionamiento y normalizacion antes del entrenamiento.

Material de apoyo:

- [dataset_summary_raw.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_summary_raw.json)
- [dataset_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_summary.json)
- [dataset_mosaic.png](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_mosaic.png)
- [dataset_audit.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/dataset_audit.json)

## 3. Objetivos del taller

### Objetivo general

Entrenar e interpretar una CNN para clasificacion binaria de rostros, desarrollando una app web que permita visualizar tanto la prediccion como sus explicaciones visuales.

### Objetivos especificos

1. Explorar la estructura y el contenido del dataset.
2. Construir un pipeline de preprocesamiento consistente para entrenamiento e inferencia.
3. Implementar una CNN desde cero en `TensorFlow/Keras`.
4. Comparar experimentos con distintos hiperparametros.
5. Aplicar `Saliency Map` y `Grad-CAM`.
6. Integrar el modelo en una aplicacion `Streamlit`.
7. Publicar la app para consulta mediante enlace.

## 4. Preprocesamiento y particion

El pipeline base de cada imagen fue:

```text
Lectura RGB -> Resize a 224x224 -> Normalizacion [0,1] -> Tensor de entrada
```

Adicionalmente, en la version corregida del proyecto se incorporaron pasos que no estaban presentes en la corrida inicial:

- deduplicacion visual por `compact_hash`;
- particion estratificada evitando fuga entre grupos duplicados;
- cache de recortes faciales para experimentos centrados en rostro.

La particion actual usada en el pipeline corregido quedo asi:

- `train`: `3788` imagenes (`1920 female`, `1868 male`)
- `val`: `251` imagenes (`108 female`, `143 male`)
- `test`: `251` imagenes (`109 female`, `142 male`)

Es importante interpretar bien estos numeros:

- `1672` corresponde al numero de imagenes unicas tras deduplicacion;
- el split de entrenamiento vuelve a expandir duplicados solo dentro de grupos asignados a `train`, con el fin de no mezclar informacion entre `train`, `val` y `test`, pero aprovechar mas muestras internas en el ajuste.

Este cambio fue metodologicamente clave, porque la version inicial inflaba metricas al permitir que imagenes iguales o muy similares aparecieran en distintos splits.

Material de apoyo:

- [split_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/split_summary.json)
- [deduplication_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/deduplication_summary.json)
- [prepared_face_summary.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/prepared_face_summary.json)
- [02_diagrama_flujo.mmd](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/02_diagrama_flujo.mmd)

## 5. Arquitectura de la CNN

El proyecto incluyo varias configuraciones CNN. La arquitectura mejor posicionada como checkpoint individual para la demo fue `wider_k5_dropout`, con la siguiente estructura general:

| Capa | Configuracion |
|---|---|
| Conv2D 1 | 48 filtros, kernel 5x5, ReLU |
| BatchNormalization | estabilizacion |
| MaxPooling2D 1 | 2x2 |
| Conv2D 2 | 96 filtros, kernel 5x5, ReLU |
| BatchNormalization | estabilizacion |
| MaxPooling2D 2 | 2x2 |
| Conv2D 3 | 192 filtros, kernel 5x5, ReLU |
| BatchNormalization | estabilizacion |
| MaxPooling2D 3 | 2x2 |
| Conv2D 4 | 288 filtros, kernel 5x5, ReLU |
| BatchNormalization | estabilizacion |
| GlobalAveragePooling2D | reduccion espacial |
| Dense | 128 neuronas, ReLU |
| Dropout | 0.40 |
| Dense salida | 1 neurona, `sigmoid` |

El compilado se realizo con:

- optimizador `Adam`
- funcion de perdida `binary_crossentropy`
- metricas `accuracy` y `AUC`

En versiones corregidas del pipeline tambien se probaron arquitecturas mas compactas y variantes centradas en rostro. La app publicada no depende de una sola de ellas: usa un modo `Robusto (ensemble)` que combina varios checkpoints para reducir errores extremos en imagenes externas.

Material de apoyo:

- [08_arquitectura_cnn.mmd](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/08_arquitectura_cnn.mmd)
- [cnn_architecture_manim.mp4](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/animations/cnn_architecture_manim.mp4)

## 6. Resultados de entrenamiento

Aqui hay que distinguir dos lecturas:

### 6.1. Lectura metodologica corregida

Cuando se corrigio la fuga entre splits y se entreno con un protocolo mas honesto, el rendimiento disminuyo de forma importante respecto a la corrida inicial. Eso confirma que las metricas antiguas estaban infladas por duplicados.

Resultados representativos del pipeline corregido:

- `compact_k5_dropout`
  - `test_accuracy = 0.6614`
  - `test_auc = 0.7665`
  - `test_loss = 0.6220`
- `face_crop_k5`
  - `test_accuracy = 0.6175`
  - `test_auc = 0.6951`
  - `test_loss = 0.6493`
- `baseline`
  - `test_accuracy = 0.4382`
  - `test_auc = 0.5330`
  - `test_loss = 0.7131`

La conclusion tecnica de esta fase es que el problema real es mas dificil de lo que sugerian las metricas iniciales.

### 6.2. Checkpoint operativo para la demo

Se conserva ademas el checkpoint `wider_k5_dropout`, que obtuvo:

- `test_accuracy = 0.9742`
- `test_auc = 0.9860`
- `test_loss = 0.1337`

Este modelo se mantiene como referencia operativa para la app porque se comporta mejor en varios casos de demostracion. Sin embargo, esas metricas no deben presentarse como la evidencia metodologica principal del laboratorio, ya que provienen de una corrida legada anterior a la auditoria completa.

En otras palabras:

- para el informe tecnico, la lectura importante es la del pipeline corregido;
- para la experiencia publica de la app, se priorizo robustez perceptiva en la demo.

Material de apoyo:

- [comparison.csv](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/comparison.csv)
- [best_experiment.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/best_experiment.json)
- [wider_k5_dropout metrics.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/wider_k5_dropout/metrics.json)
- [compact_k5_dropout metrics.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/compact_k5_dropout/metrics.json)
- [face_crop_k5 metrics.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/experiments/face_crop_k5/metrics.json)

## 7. Ajuste de hiperparametros

La comparacion principal entre configuraciones fue la siguiente:

| Experimento | Filtros base | Kernel | Dense | Dropout | Learning rate | Test accuracy | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| wider_k5_dropout | 48 | 5 | 128 | 0.40 | 0.0005 | 0.9742 | 0.9860 |
| compact_k5_dropout | 32 | 5 | 96 | 0.50 | 0.0005 | 0.6614 | 0.7665 |
| face_crop_k5 | 32 | 5 | 128 | 0.35 | 0.0005 | 0.6175 | 0.6951 |
| baseline | 24 | 3 | 96 | 0.45 | 0.00035 | 0.4382 | 0.5330 |

Interpretacion:

- `baseline` funciona como linea base minima.
- `compact_k5_dropout` mejora claramente frente a `baseline` en el protocolo corregido.
- `face_crop_k5` explora entrenamiento sobre recortes faciales, pero todavia no supera a `compact_k5_dropout`.
- `wider_k5_dropout` es el checkpoint que mejor luce en la demo, aunque debe interpretarse como modelo legado operativo y no como la evidencia metodologica mas limpia.

## 8. Interpretabilidad visual

Se aplicaron dos tecnicas XAI:

- `Saliency Map`: destaca pixeles sensibles para la salida.
- `Grad-CAM`: resalta regiones semanticamente influyentes en capas profundas.

Estas tecnicas fueron utiles para dos tareas:

1. explicar predicciones correctas;
2. detectar errores de generalizacion.

En pruebas externas se vio que:

- cuando el modelo recibe imagenes con mucho fondo, puede apoyarse en contexto espurio;
- cuando el rostro queda mejor centrado, la prediccion suele mejorar;
- en imagenes editoriales o muy estilizadas, algunos modelos se vuelven excesivamente confiados.

Por eso la app final incorpora:

- `Imagen completa`
- `Auto: detectar rostro`
- `Recorte retrato`
- `Robusto (ensemble)` como modo de inferencia recomendado

Material de apoyo:

- [correct_prediction_panel.png](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/interpretability/correct_prediction_panel.png)
- [correct_prediction_metadata.json](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/interpretability/correct_prediction_metadata.json)

## 9. Descripcion de la app en Streamlit

La aplicacion desarrollada en `Streamlit` contiene dos espacios:

### Informe del laboratorio

- contexto del dataset;
- objetivos;
- flujo de preprocesamiento;
- arquitectura CNN;
- resultados e hiperparametros;
- hallazgos de auditoria y XAI.

### Demo interactiva

- carga de imagen;
- seleccion de modelo o modo robusto;
- seleccion de region de inferencia;
- visualizacion de probabilidades;
- `Saliency Map` y `Grad-CAM`;
- resumen de votos cuando se usa el ensemble.

Esto permite que la pagina sirva al mismo tiempo como entregable tecnico y como demostracion visual.

## 10. Mejoras futuras

1. Crear un conjunto externo curado de retratos reales y editoriales para validar generalizacion.
2. Reentrenar un modelo final solo sobre protocolo corregido, pero con mas variacion y augmentation fuerte.
3. Incorporar deteccion facial mas robusta que Haar Cascade.
4. Explorar calibracion de probabilidades para reducir salidas sobreconfiadas.

## 11. Conclusiones

El laboratorio deja dos aprendizajes centrales.

El primero es metodologico: un modelo puede parecer excelente cuando existe fuga entre entrenamiento y prueba, pero esa conclusion cambia radicalmente cuando se audita el dataset y se corrige la particion. En este proyecto, la revision de duplicados transformo por completo la lectura del sistema.

El segundo es practico: las tecnicas XAI no solo sirven para "decorar" una prediccion, sino para descubrir que la red todavia depende de factores espurios como fondo, encuadre o estilo visual. A partir de eso se justifico una app final mas robusta, con varios modos de recorte y un ensemble para la demostracion publica.

En conjunto, el proyecto cumple con lo pedido por el laboratorio y, ademas, aporta una lectura critica sobre la calidad del dataset, la validez de la evaluacion y las limitaciones reales del modelo.

