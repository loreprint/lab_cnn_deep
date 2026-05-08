from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cnn_xai_lab.config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, MODELS_DIR, OUTPUTS_DIR, SEED
from cnn_xai_lab.data import (
    build_face_crop_cache,
    deduplicate_dataframe,
    discover_images,
    enrich_with_compact_hash,
    make_tf_dataset,
    save_json,
    save_sample_mosaic,
    split_dataframe,
    summarize_dataset,
    summarize_splits,
)
from cnn_xai_lab.training import ExperimentConfig, run_experiments
from cnn_xai_lab.xai import (
    compute_saliency_map,
    make_gradcam_heatmap,
    predict_probabilities,
    prepare_image,
    save_interpretability_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN XAI laboratory model.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=["face_crop_k5", "face_crop_wider"],
        help="Run only the selected experiment. You can repeat this flag to run more than one.",
    )
    return parser.parse_args()


def find_correctly_classified_example(model: keras.Model, test_frame, image_size: tuple[int, int]):
    for _, row in test_frame.iterrows():
        image_path = row["model_path"] if "model_path" in row else row["path"]
        with Image.open(image_path) as image:
            image_batch = prepare_image(image, image_size=image_size)
        prediction = predict_probabilities(model, image_batch)
        if prediction["predicted_index"] == int(row["label"]):
            return row, image_batch, prediction
    return None, None, None


def main() -> None:
    args = parse_args()
    raw_dataset_frame = discover_images(args.data_dir)
    if raw_dataset_frame.empty:
        raise FileNotFoundError(
            f"No images were found in {args.data_dir}. Populate data/female and data/male first."
        )

    hashed_dataset_frame = enrich_with_compact_hash(raw_dataset_frame, hash_size=16)
    dataset_frame = deduplicate_dataframe(hashed_dataset_frame, group_column="compact_hash")
    prepared_dataset_frame = build_face_crop_cache(
        hashed_dataset_frame,
        cache_dir=OUTPUTS_DIR / "prepared_faces_v2",
    )

    outputs_dir = OUTPUTS_DIR
    experiments_dir = outputs_dir / "experiments"
    interpretability_dir = outputs_dir / "interpretability"

    raw_dataset_summary = summarize_dataset(hashed_dataset_frame)
    dataset_summary = summarize_dataset(dataset_frame)
    save_json(raw_dataset_summary, outputs_dir / "dataset_summary_raw.json")
    save_json(dataset_summary, outputs_dir / "dataset_summary.json")
    save_sample_mosaic(dataset_frame, outputs_dir / "dataset_mosaic.png")
    crop_method_counts = prepared_dataset_frame["crop_method"].value_counts().sort_index().to_dict()
    save_json(
        {
            "prepared_images": int(len(prepared_dataset_frame)),
            "crop_methods": {str(key): int(value) for key, value in crop_method_counts.items()},
        },
        outputs_dir / "prepared_face_summary.json",
    )

    splits = split_dataframe(
        prepared_dataset_frame,
        seed=args.seed,
        group_column="compact_hash",
        keep_group_duplicates_for_train=True,
    )
    split_summary = summarize_splits(splits)
    save_json(split_summary, outputs_dir / "split_summary.json")
    save_json(
        {
            "raw_images": int(len(raw_dataset_frame)),
            "unique_training_images": int(len(dataset_frame)),
            "removed_duplicates": int(len(raw_dataset_frame) - len(dataset_frame)),
            "mean_duplicate_group_size": float(dataset_frame["duplicate_count"].mean()),
        },
        outputs_dir / "deduplication_summary.json",
    )

    train_class_counts = splits["train"]["label"].value_counts().to_dict()
    total_train = sum(train_class_counts.values())
    minority_count = min(train_class_counts.values())
    majority_count = max(train_class_counts.values())
    imbalance_ratio = majority_count / minority_count if minority_count else 1.0
    class_weight = None
    if imbalance_ratio > 1.15:
        class_weight = {
            int(label): total_train / (len(train_class_counts) * count)
            for label, count in train_class_counts.items()
        }
    save_json(
        {
            "class_counts": {str(key): int(value) for key, value in train_class_counts.items()},
            "imbalance_ratio": float(imbalance_ratio),
            "class_weight": (
                {str(key): float(value) for key, value in class_weight.items()}
                if class_weight is not None
                else None
            ),
        },
        outputs_dir / "class_weight_summary.json",
    )

    train_dataset = make_tf_dataset(
        splits["train"],
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        image_column="model_path",
    )
    val_dataset = make_tf_dataset(
        splits["val"],
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
        image_column="model_path",
    )
    test_dataset = make_tf_dataset(
        splits["test"],
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
        image_column="model_path",
    )

    experiments = [
        ExperimentConfig(
            name="face_crop_k5",
            base_filters=40,
            kernel_size=5,
            dense_units=160,
            dropout=0.40,
            learning_rate=4e-4,
            l2_strength=1.5e-4,
            label_smoothing=0.04,
        ),
        ExperimentConfig(
            name="face_crop_wider",
            base_filters=56,
            kernel_size=3,
            dense_units=192,
            dropout=0.35,
            learning_rate=3e-4,
            l2_strength=1.0e-4,
            label_smoothing=0.03,
        ),
    ]
    if args.experiment:
        allowed = set(args.experiment)
        experiments = [experiment for experiment in experiments if experiment.name in allowed]

    comparison, best_info = run_experiments(
        experiments=experiments,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=experiments_dir,
        best_model_path=MODELS_DIR / "model.keras",
        image_size=IMAGE_SIZE,
        epochs=args.epochs,
        class_weight=class_weight,
    )

    best_model = keras.models.load_model(MODELS_DIR / "model.keras")
    selected_row, image_batch, prediction = find_correctly_classified_example(
        best_model,
        splits["test"],
        image_size=IMAGE_SIZE,
    )

    if selected_row is not None:
        saliency = compute_saliency_map(best_model, image_batch, prediction["predicted_index"])
        gradcam = make_gradcam_heatmap(best_model, image_batch, prediction["predicted_index"])
        save_interpretability_panel(
            image_batch=image_batch,
            saliency=saliency,
            gradcam=gradcam,
            output_path=interpretability_dir / "correct_prediction_panel.png",
            title=f'Interpretabilidad | {selected_row["class_name"]}',
        )
        save_json(
            {
                "image_path": selected_row["path"],
                "true_label": selected_row["class_name"],
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
            },
            interpretability_dir / "correct_prediction_metadata.json",
        )

    print("Training complete.")
    print(f"Raw images discovered: {len(raw_dataset_frame)}")
    print(f"Unique images after deduplication: {len(dataset_frame)}")
    print(f"Prepared face crops: {len(prepared_dataset_frame)}")
    print("Split summary:", split_summary)
    print("Best experiment:", best_info["best_experiment"])
    print("Comparison table:")
    print(comparison[["name", "test_accuracy", "test_loss", "test_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
