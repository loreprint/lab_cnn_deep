from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from cnn_xai_lab.modeling import find_last_conv_layer_name


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)


def batch_to_uint8(image_batch: np.ndarray) -> np.ndarray:
    image = np.clip(image_batch[0], 0.0, 1.0)
    return (image * 255).astype("uint8")


def predict_probabilities(model: keras.Model, image_batch: np.ndarray) -> dict:
    male_probability = float(model.predict(image_batch, verbose=0)[0][0])
    female_probability = 1.0 - male_probability
    predicted_index = 1 if male_probability >= 0.5 else 0

    return {
        "female_probability": female_probability,
        "male_probability": male_probability,
        "predicted_index": predicted_index,
        "predicted_label": "male" if predicted_index == 1 else "female",
        "confidence": male_probability if predicted_index == 1 else female_probability,
    }


def compute_saliency_map(model: keras.Model, image_batch: np.ndarray, target_class: int) -> np.ndarray:
    inputs = tf.convert_to_tensor(image_batch)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        score = predictions[:, 0] if target_class == 1 else 1.0 - predictions[:, 0]

    gradients = tape.gradient(score, inputs)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
    saliency = saliency.numpy()
    saliency = np.maximum(saliency, 0)
    max_value = saliency.max() or 1.0
    return saliency / max_value


def make_gradcam_heatmap(
    model: keras.Model,
    image_batch: np.ndarray,
    target_class: int,
    layer_name: str | None = None,
) -> np.ndarray:
    layer_name = layer_name or find_last_conv_layer_name(model)
    inputs = tf.convert_to_tensor(image_batch)

    _ = model(inputs, training=False)
    layer_names = [layer.name for layer in model.layers]
    layer_index = layer_names.index(layer_name)
    conv_layer = model.get_layer(layer_name)

    conv_model = keras.Model(inputs=model.inputs, outputs=conv_layer.output)

    classifier_input = keras.Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[layer_index + 1 :]:
        x = layer(x)
    classifier_model = keras.Model(inputs=classifier_input, outputs=x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(inputs, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        score = predictions[:, 0] if target_class == 1 else 1.0 - predictions[:, 0]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise ValueError(f"Gradients could not be computed for layer '{layer_name}'.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0.0:
        return np.zeros_like(heatmap.numpy())
    return (heatmap / max_value).numpy()


def overlay_map_on_image(
    image_batch: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> np.ndarray:
    base_image = batch_to_uint8(image_batch)
    cmap = cm.get_cmap(colormap)
    colored_map = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    colored_map = Image.fromarray((colored_map * 255).astype("uint8")).resize(
        (base_image.shape[1], base_image.shape[0]),
        Image.Resampling.BILINEAR,
    )
    colored_array = np.asarray(colored_map).astype("float32")

    overlay = ((1.0 - alpha) * base_image.astype("float32")) + (alpha * colored_array)
    return np.clip(overlay, 0, 255).astype("uint8")


def save_interpretability_panel(
    image_batch: np.ndarray,
    saliency: np.ndarray,
    gradcam: np.ndarray,
    output_path: Path,
    title: str = "Interpretabilidad visual",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = batch_to_uint8(image_batch)
    saliency_overlay = overlay_map_on_image(image_batch, saliency, colormap="magma")
    gradcam_overlay = overlay_map_on_image(image_batch, gradcam, colormap="viridis")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (original, "Imagen original"),
        (saliency_overlay, "Saliency Map"),
        (gradcam_overlay, "Grad-CAM"),
    ]

    for axis, (panel, panel_title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(panel_title)
        axis.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
