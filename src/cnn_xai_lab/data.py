from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from cnn_xai_lab.config import CLASS_NAMES
from cnn_xai_lab.face_detection import detect_primary_face, make_portrait_focus_crop


CLASS_DIR_ALIASES = {
    "female": {
        "female",
        "females",
        "femaleface",
        "femalefaces",
        "female_faces",
    },
    "male": {
        "male",
        "males",
        "maleface",
        "malefaces",
        "male_faces",
    },
}


def _normalize_dir_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_class_dir(data_dir: Path, class_name: str) -> Path | None:
    direct_dir = data_dir / class_name
    if direct_dir.exists():
        return direct_dir

    expected_names = {_normalize_dir_name(class_name)}
    expected_names.update(_normalize_dir_name(alias) for alias in CLASS_DIR_ALIASES.get(class_name, set()))

    candidates = []
    for directory in data_dir.rglob("*"):
        if not directory.is_dir():
            continue
        normalized = _normalize_dir_name(directory.name)
        if normalized in expected_names:
            candidates.append(directory)

    if not candidates:
        return None

    candidates.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return candidates[0]


def discover_images(data_dir: Path) -> pd.DataFrame:
    records = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = _find_class_dir(data_dir, class_name)
        if class_dir is None or not class_dir.exists():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            with Image.open(image_path) as image:
                width, height = image.size
                file_format = image.format or image_path.suffix.replace(".", "").upper()

            records.append(
                {
                    "path": str(image_path),
                    "class_name": class_name,
                    "label": label,
                    "width": width,
                    "height": height,
                    "extension": image_path.suffix.lower(),
                    "format": file_format,
                }
            )

    return pd.DataFrame(records)


def compute_compact_hash(image_path: str | Path, hash_size: int = 16) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    return hashlib.sha1(rgb.tobytes()).hexdigest()


def enrich_with_compact_hash(frame: pd.DataFrame, hash_size: int = 16) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["compact_hash"] = enriched["path"].map(lambda path: compute_compact_hash(path, hash_size=hash_size))
    return enriched


def deduplicate_dataframe(frame: pd.DataFrame, group_column: str = "compact_hash") -> pd.DataFrame:
    if group_column not in frame.columns:
        raise KeyError(f"'{group_column}' column not found in dataframe.")

    deduplicated = frame.drop_duplicates(subset=[group_column]).copy().reset_index(drop=True)
    group_sizes = frame.groupby(group_column).size().rename("duplicate_count")
    deduplicated = deduplicated.merge(group_sizes, on=group_column, how="left")
    deduplicated["duplicate_count"] = deduplicated["duplicate_count"].fillna(1).astype(int)
    return deduplicated


def build_face_crop_cache(
    frame: pd.DataFrame,
    cache_dir: Path,
    image_column: str = "path",
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = []

    for _, row in frame.iterrows():
        source_path = Path(str(row[image_column]))
        cache_name = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest() + ".jpg"
        class_dir = cache_dir / str(row["class_name"])
        class_dir.mkdir(parents=True, exist_ok=True)
        cached_path = class_dir / cache_name

        crop_method = "face_detection"
        if not cached_path.exists():
            with Image.open(source_path) as image:
                rgb_image = image.convert("RGB")
                face_result = detect_primary_face(rgb_image)
                if face_result is not None:
                    prepared_image = face_result.cropped_face
                else:
                    crop_method = "portrait_fallback"
                    prepared_image = make_portrait_focus_crop(rgb_image).cropped_face
                prepared_image.convert("RGB").save(cached_path, format="JPEG", quality=95)
        else:
            crop_method = "cached"

        prepared_rows.append(
            {
                **row.to_dict(),
                "model_path": str(cached_path),
                "crop_method": crop_method,
            }
        )

    return pd.DataFrame(prepared_rows)


def summarize_dataset(frame: pd.DataFrame) -> Dict[str, object]:
    if frame.empty:
        return {
            "num_images": 0,
            "classes": {},
            "formats": {},
            "width": {},
            "height": {},
        }

    class_counts = frame["class_name"].value_counts().sort_index().to_dict()
    format_counts = frame["format"].value_counts().sort_index().to_dict()

    summary = {
        "num_images": int(len(frame)),
        "classes": {key: int(value) for key, value in class_counts.items()},
        "formats": {key: int(value) for key, value in format_counts.items()},
        "width": {
            "min": int(frame["width"].min()),
            "max": int(frame["width"].max()),
            "mean": float(frame["width"].mean()),
        },
        "height": {
            "min": int(frame["height"].min()),
            "max": int(frame["height"].max()),
            "mean": float(frame["height"].mean()),
        },
    }

    if "compact_hash" in frame.columns:
        summary["unique_compact_hashes"] = int(frame["compact_hash"].nunique())
        summary["duplicate_rows"] = int(len(frame) - frame["compact_hash"].nunique())

    if "duplicate_count" in frame.columns:
        summary["duplicate_groups_retained"] = int(len(frame))
        summary["mean_duplicate_group_size"] = float(frame["duplicate_count"].mean())

    return summary


def save_json(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_sample_mosaic(frame: pd.DataFrame, output_path: Path, samples_per_class: int = 6) -> None:
    if frame.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rows = []

    for class_name in CLASS_NAMES:
        class_frame = frame[frame["class_name"] == class_name]
        if class_frame.empty:
            continue
        sample_rows.append(
            class_frame.sample(
                n=min(samples_per_class, len(class_frame)),
                random_state=42,
            )
        )

    sampled = pd.concat(sample_rows, ignore_index=True)
    total = len(sampled)
    cols = min(4, total)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flat", axes))

    for axis, (_, row) in zip(axes, sampled.iterrows()):
        with Image.open(row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f'{row["class_name"]} | {row["width"]}x{row["height"]}')
        axis.axis("off")

    for axis in axes[total:]:
        axis.axis("off")

    fig.suptitle("Mosaico del dataset", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def split_dataframe(
    frame: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    group_column: str | None = None,
    keep_group_duplicates_for_train: bool = False,
) -> Dict[str, pd.DataFrame]:
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("The split ratios must add up to 1.0.")

    split_frame = frame
    if group_column:
        if group_column not in frame.columns:
            raise KeyError(f"'{group_column}' column not found in dataframe.")
        split_frame = frame.drop_duplicates(subset=[group_column]).copy()

    train_frame, temp_frame = train_test_split(
        split_frame,
        train_size=train_size,
        stratify=split_frame["label"],
        random_state=seed,
    )

    val_ratio = val_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        train_size=val_ratio,
        stratify=temp_frame["label"],
        random_state=seed,
    )

    split_map = {
        "train": train_frame.reset_index(drop=True),
        "val": val_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }

    if group_column:
        expanded_map = {}
        for split_name, split_value in split_map.items():
            expanded = frame[frame[group_column].isin(split_value[group_column])].copy()
            if split_name != "train" or not keep_group_duplicates_for_train:
                expanded = expanded.drop_duplicates(subset=[group_column])
            expanded_map[split_name] = expanded.reset_index(drop=True)
        return expanded_map

    return split_map


def summarize_splits(splits: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    summary = {}
    for split_name, split_frame in splits.items():
        class_counts = split_frame["class_name"].value_counts().sort_index().to_dict()
        summary[split_name] = {
            "count": int(len(split_frame)),
            "classes": {key: int(value) for key, value in class_counts.items()},
        }
    return summary


def _apply_augmentation(image: tf.Tensor, image_size: tuple[int, int], seed: int) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_brightness(image, max_delta=0.10, seed=seed)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10, seed=seed)
    image = tf.image.random_saturation(image, lower=0.90, upper=1.10, seed=seed)
    enlarged = tf.image.resize(image, (image_size[0] + 20, image_size[1] + 20))
    image = tf.image.random_crop(enlarged, size=(image_size[0], image_size[1], 3), seed=seed + 23)
    return tf.clip_by_value(image, 0.0, 1.0)


def _load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
    training: bool,
    seed: int,
):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = _apply_augmentation(image, image_size=image_size, seed=seed)
    label = tf.cast(label, tf.float32)
    return image, label


def make_tf_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int = 42,
    image_column: str = "path",
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            frame[image_column].astype(str).tolist(),
            frame["label"].astype("float32").tolist(),
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=max(len(frame), 1),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: _load_and_preprocess_image(
            path,
            label,
            image_size,
            training=training,
            seed=seed,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
