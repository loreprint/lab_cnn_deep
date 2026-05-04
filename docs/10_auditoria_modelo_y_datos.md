# Auditoria del modelo y de los datos

Esta auditoria resume la revision tecnica realizada sobre el dataset, el protocolo de evaluacion y el modelo despues de detectar resultados sospechosamente altos en la primera version del laboratorio.

## 1. Balance de clases

El dataset crudo no presenta un desbalance importante:

- `female`: `2698`
- `male`: `2720`

Conclusion:

- El balanceo de clases no era el problema principal.
- Aplicar `class_weight` puede ayudar ligeramente, pero no explica por si solo la diferencia entre resultados antiguos y corregidos.

## 2. Preprocesamiento basico

El pipeline si estaba haciendo correctamente estas etapas:

- Lectura de imagenes en `RGB`
- Redimensionamiento a `224x224`
- Normalizacion de pixeles al rango `[0,1]`
- Particion estratificada por clase

Conclusion:

- El problema principal no estaba en RGB, resize o normalizacion.
- La debilidad venia de la limpieza de datos y del protocolo de evaluacion.

## 3. Limpieza de datos y problema encontrado

La auditoria encontro una gran cantidad de duplicados o casi duplicados:

- Total de imagenes procesadas: `5418`
- Huellas visuales compactas unicas: `1672`
- Filas duplicadas aproximadas: `3746`

Ademas, la version antigua del pipeline permitia fuga entre splits:

- `train` vs `val`: `593` huellas compartidas
- `train` vs `test`: `587`
- `val` vs `test`: `250`

Conclusion:

- Las metricas antiguas estaban contaminadas por fuga de informacion.
- El modelo probablemente estaba viendo imagenes iguales o muy parecidas en entrenamiento y luego en validacion o prueba.

## 4. Cambios aplicados tras la auditoria

Para corregir el problema se implementaron estos cambios:

1. Deduplicacion por `compact_hash` antes de entrenar.
2. Split estricto sobre el dataset limpio.
3. Arquitectura mas compacta con `GlobalAveragePooling2D` en lugar de `Flatten`.
4. `BatchNormalization`, `Dropout` y regularizacion mas conservadora.
5. Comparacion de dos configuraciones bajo el mismo protocolo corregido.

El dataset final usado para entrenar quedo en:

- `1170` imagenes de entrenamiento
- `251` imagenes de validacion
- `251` imagenes de prueba

## 5. Efecto real sobre las metricas

La corrida antigua reportaba resultados cercanos a `0.98` en prueba. Despues de corregir la fuga, el mejor experimento (`compact_k5_dropout`) obtuvo:

- `test_accuracy = 0.6614`
- `test_auc = 0.7665`
- `test_loss = 0.6220`

Conclusion:

- La caida de rendimiento confirma que las metricas antiguas estaban infladas.
- El nuevo resultado es metodologicamente mas confiable, aunque menos espectacular.

## 6. Comportamiento observado en la app

Las pruebas manuales con imagenes externas mostraron que:

- la prediccion mejora cuando se recorta el rostro;
- la prediccion empeora cuando la imagen tiene mucho fondo o la cara ocupa una region pequena.

Esto indica que el modelo todavia depende parcialmente del contexto visual y no solo de rasgos faciales.

## 7. Veredicto de la auditoria

### Lo que estaba bien

- El balance entre clases era razonable.
- El preprocesamiento basico RGB + resize + normalizacion era correcto.
- La app con XAI permitio detectar un problema real de generalizacion.

### Lo que estaba mal o incompleto

- No se habia deduplicado antes de separar el dataset.
- El split permitia fuga entre `train`, `val` y `test`.
- La primera arquitectura era demasiado grande y propensa a sobreajuste.
- Las metricas iniciales no eran una estimacion robusta del desempeno real.

## 8. Recomendaciones tecnicas

1. Entrenar sobre recortes de rostro desde la fase de entrenamiento.
2. Mantener la deduplicacion y el split estricto en futuras corridas.
3. Reforzar `data augmentation`.
4. Evaluar con un conjunto externo adicional.
5. Seguir usando XAI para validar si la red atiende realmente al rostro.

## 9. Conclusion corta para el informe

`El problema principal no fue el desbalance de clases, sino la presencia de duplicados y la fuga entre entrenamiento, validacion y prueba. Tras corregir ese problema, el rendimiento del modelo bajo a un nivel mas realista, lo que confirma que la evaluacion inicial estaba inflada.`
