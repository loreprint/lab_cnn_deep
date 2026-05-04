from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from cnn_xai_lab.modeling import build_cnn_model


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    base_filters: int = 32
    kernel_size: int = 3
    dense_units: int = 128
    dropout: float = 0.30
    learning_rate: float = 1e-3


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["loss"], label="train")
    axes[0].plot(history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["accuracy"], label="train")
    axes[1].plot(history_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_experiment(
    config: ExperimentConfig,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: Path,
    image_size: tuple[int, int],
    epochs: int = 10,
    class_weight: dict[int, float] | None = None,
) -> dict:
    experiment_dir = output_dir / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model = build_cnn_model(
        input_shape=(image_size[0], image_size[1], 3),
        base_filters=config.base_filters,
        kernel_size=config.kernel_size,
        dense_units=config.dense_units,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.6,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(experiment_dir / "history.csv", index=False)
    plot_training_curves(history_df, experiment_dir / "training_curves.png")

    test_metrics = model.evaluate(test_dataset, return_dict=True, verbose=0)
    model_path = experiment_dir / "model.keras"
    model.save(model_path)

    result = {
        "name": config.name,
        "model_path": str(model_path),
        "epochs_trained": int(len(history_df)),
        **{key: float(value) for key, value in asdict(config).items() if key != "name"},
        **{key: float(value) for key, value in history_df.iloc[-1].to_dict().items()},
        **{f"test_{key}": float(value) for key, value in test_metrics.items()},
    }

    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def run_experiments(
    experiments: Iterable[ExperimentConfig],
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: Path,
    best_model_path: Path,
    image_size: tuple[int, int],
    epochs: int = 10,
    class_weight: dict[int, float] | None = None,
) -> Tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment in experiments:
        rows.append(
            train_experiment(
                config=experiment,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_dir=output_dir,
                image_size=image_size,
                epochs=epochs,
                class_weight=class_weight,
            )
        )

    comparison = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    best_row = comparison.iloc[0].to_dict()
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_row["model_path"], best_model_path)

    best_summary = {
        "best_experiment": best_row["name"],
        "best_model_path": str(best_model_path),
        "test_accuracy": float(best_row["test_accuracy"]),
        "test_loss": float(best_row["test_loss"]),
        "test_auc": float(best_row["test_auc"]),
    }

    with (output_dir / "best_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    return comparison, best_summary
