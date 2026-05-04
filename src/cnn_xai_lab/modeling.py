from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    base_filters: int = 32,
    kernel_size: int = 3,
    dense_units: int = 128,
    dropout: float = 0.30,
    learning_rate: float = 1e-3,
) -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(height_factor=(-0.10, 0.02), width_factor=(-0.10, 0.02)),
            layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            augmentation,
            layers.Conv2D(base_filters, kernel_size, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(base_filters * 2, kernel_size, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.08),
            layers.MaxPooling2D(),
            layers.Conv2D(base_filters * 4, kernel_size, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.10),
            layers.MaxPooling2D(),
            layers.Conv2D(base_filters * 6, kernel_size, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.12),
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="gender_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def find_last_conv_layer_name(model: keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("The model does not contain a Conv2D layer.")
