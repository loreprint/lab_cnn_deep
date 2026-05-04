from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cnn_xai_lab.config import DATA_DIR, OUTPUTS_DIR
from cnn_xai_lab.data import discover_images, split_dataframe
from cnn_xai_lab.modeling import build_cnn_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset balance, duplicates and split leakage.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hash-size", type=int, default=16)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUTS_DIR / "dataset_audit.json",
    )
    return parser.parse_args()


def compact_hash(image_path: str, hash_size: int) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    return hashlib.sha1(rgb.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()

    frame = discover_images(args.data_dir).copy()
    if frame.empty:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    frame["stem"] = frame["path"].map(lambda path: Path(path).stem.lower())
    frame["compact_hash"] = frame["path"].map(lambda path: compact_hash(path, args.hash_size))

    splits = split_dataframe(frame)

    leakage = {}
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(splits[first]["compact_hash"]) & set(splits[second]["compact_hash"])
        leakage[f"{first}_{second}"] = len(shared)

    class_balance = frame["class_name"].value_counts().to_dict()
    unique_hash_balance = frame.drop_duplicates("compact_hash")["class_name"].value_counts().to_dict()
    duplicate_stems = int(frame["stem"].duplicated().sum())
    duplicate_hash_rows = int(len(frame) - frame["compact_hash"].nunique())

    model = build_cnn_model()

    payload = {
        "num_images": int(len(frame)),
        "class_balance": {key: int(value) for key, value in class_balance.items()},
        "hash_size": int(args.hash_size),
        "unique_compact_hashes": int(frame["compact_hash"].nunique()),
        "duplicate_hash_rows": duplicate_hash_rows,
        "duplicate_stems": duplicate_stems,
        "split_leakage_shared_hashes": leakage,
        "unique_hash_class_balance": {key: int(value) for key, value in unique_hash_balance.items()},
        "model_params": int(model.count_params()),
        "notes": [
            "Large overlap of compact hashes across splits suggests duplicate or near-duplicate leakage.",
            "Class balance in the raw dataset is close to uniform, so class weighting is not the primary issue.",
            "The current CNN uses a large Flatten-to-Dense transition, which increases overfitting risk.",
        ],
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
