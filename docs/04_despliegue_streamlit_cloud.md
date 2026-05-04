# Despliegue en Streamlit Community Cloud

Esta es la ruta recomendada para publicar la app y obtener un link compartible.

## 1. Punto clave antes de empezar

El archivo [model.keras](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/models/model.keras) pesa aproximadamente `238 MB`. GitHub bloquea archivos mayores a `100 MB` en un push normal, por lo que este proyecto debe subirse usando `Git LFS`.

Esto ya quedo preparado en:

- [.gitattributes](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/.gitattributes)
- [.gitignore](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/.gitignore)

Segun la documentacion oficial de GitHub, los archivos mayores de `100 MB` deben gestionarse con `Git LFS`. Streamlit Community Cloud es compatible con repositorios que usan `Git LFS`.

## 2. Requisitos previos

- Cuenta de GitHub.
- Cuenta de Streamlit Community Cloud.
- `git` instalado.
- `git lfs` instalado.
- Proyecto funcionando localmente.

Verificacion local sugerida:

```powershell
cd C:\Users\lore7\Documents\Playground\cnn_xai_gender_lab
git lfs version
& .\.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

## 3. Archivos que deben quedar en el repositorio

Minimos para la app:

- [streamlit_app.py](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/streamlit_app.py)
- [requirements.txt](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/requirements.txt)
- [.streamlit/config.toml](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/.streamlit/config.toml)
- [models/model.keras](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/models/model.keras)
- [src/cnn_xai_lab](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/src/cnn_xai_lab)
- [outputs/animations/cnn_architecture_manim.mp4](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/animations/cnn_architecture_manim.mp4)
- archivos `.json`, `.csv` e imagenes de `outputs/` que la app usa para el informe

No es necesario subir todos los checkpoints pesados de `outputs/experiments/*/model.keras`.

## 4. Crear el repositorio en GitHub

1. Crea un repositorio nuevo en GitHub.
2. Abre una terminal en la carpeta del proyecto.
3. Ejecuta:

```powershell
cd C:\Users\lore7\Documents\Playground\cnn_xai_gender_lab
git init
git lfs install
git lfs track "models/model.keras"
git add .gitattributes
git add .
git commit -m "Proyecto final del laboratorio CNN XAI"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
git push -u origin main
```

Si este directorio ya esta dentro de otro repo padre, tienes dos opciones:

- crear un repo independiente solo para `cnn_xai_gender_lab`;
- o subir esta carpeta manualmente a un repo nuevo desde GitHub Desktop.

La opcion mas limpia para este proyecto es un repo independiente.

## 5. Comprobaciones en GitHub

Antes de ir a Streamlit Cloud, verifica en GitHub que:

- `streamlit_app.py` sea visible;
- `requirements.txt` sea visible;
- `.streamlit/config.toml` sea visible;
- `models/model.keras` aparezca como archivo LFS;
- el directorio `outputs/` tenga los artefactos que usa el informe de la app.

## 6. Crear la app en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io).
2. Inicia sesion con GitHub.
3. Pulsa `Create app`.
4. Selecciona tu repositorio.
5. Escoge la rama `main`.
6. Como archivo principal selecciona `streamlit_app.py`.
7. Si quieres una URL mas limpia, define un subdominio personalizado en el campo `App URL`.

## 7. Advanced settings recomendados

- Python version: `3.12`
- Secrets: vacio, a menos que agregues claves externas

La documentacion oficial indica que Community Cloud usa Python soportado y actualmente `3.12` es la opcion por defecto. En este proyecto esa version es consistente con el entorno local.

## 8. Primera prueba despues del despliegue

Cuando la app abra, verifica:

1. que cargue sin errores;
2. que se vean las secciones `Informe del laboratorio` y `Demo interactiva`;
3. que permita subir imagenes;
4. que muestre probabilidades y mapas XAI;
5. que reproduzca el video de arquitectura;
6. que el modo `Robusto (ensemble)` funcione.

## 9. Problemas comunes

### GitHub rechaza el push

Causa probable:

- `models/model.keras` se intento subir sin `Git LFS`.

Solucion:

- confirma que `git lfs track "models/model.keras"` este aplicado;
- vuelve a hacer `git add .gitattributes`;
- luego `git add models/model.keras`.

### Streamlit Cloud no encuentra dependencias

Revisa:

- que [requirements.txt](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/requirements.txt) este en la raiz del repo;
- que no exista otro archivo de dependencias conflictivo;
- que no hayas borrado `src/cnn_xai_lab`.

### La app abre, pero faltan metricas o imagenes

Revisa:

- que los artefactos de `outputs/` si se hayan subido;
- que `best_experiment.json` y `comparison.csv` existan;
- que `dataset_mosaic.png` y `correct_prediction_panel.png` existan.

### El video de arquitectura no aparece

Revisa:

- que [cnn_architecture_manim.mp4](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/outputs/animations/cnn_architecture_manim.mp4) exista en el repo.

## 10. Donde pegar el link final

Cuando la app quede publicada, copia la URL y pegala en:

- [03_entregable_manuscrito.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/03_entregable_manuscrito.md)
- [07_entrega_final.md](/C:/Users/lore7/Documents/Playground/cnn_xai_gender_lab/docs/07_entrega_final.md)

## Referencias oficiales

- [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization for Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [App dependencies for Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github?platform=windows)
- [About Git Large File Storage](https://docs.github.com/es/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
