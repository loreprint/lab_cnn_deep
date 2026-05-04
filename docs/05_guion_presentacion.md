# Guion corto para la presentacion

## 1. Apertura

`En este laboratorio construimos una CNN desde cero para clasificar rostros en las clases female y male, y despues usamos tecnicas de interpretabilidad para entender en que regiones de la imagen se apoya el modelo.`

## 2. Dataset

`El dataset fue organizado en dos carpetas, male y female. Antes de entrenar revisamos el numero de imagenes, sus formatos y su variabilidad de tamanos.`

## 3. Preprocesamiento

`Todas las imagenes se mantuvieron en RGB, se redimensionaron a 224x224 y se normalizaron al rango de 0 a 1. Luego se dividieron en entrenamiento, validacion y prueba con una semilla fija.`

## 4. Modelo

`La arquitectura es una CNN secuencial con tres bloques Conv2D + MaxPooling2D, seguida por capas densas y una salida sigmoide para clasificacion binaria.`

## 5. Hiperparametros

`Comparamos al menos dos configuraciones para observar el impacto de filtros, kernel, dropout y tasa de aprendizaje sobre las metricas finales.`

## 6. Interpretabilidad

`El Saliency Map resalta pixeles sensibles a pequenos cambios, mientras que Grad-CAM muestra regiones mas semanticas a partir de una capa convolucional profunda.`

## 7. Demo

`Ahora voy a subir una imagen, mostrar la prediccion y explicar si el modelo esta mirando regiones faciales relevantes o elementos espurios del fondo.`

## 8. Cierre

`La interpretabilidad nos permitio validar si la CNN estaba tomando decisiones razonables y nos dio evidencia visual para discutir fortalezas y limitaciones del modelo.`
