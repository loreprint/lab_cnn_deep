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

from cnn_xai_lab.config import IMAGE_SIZE, MODELS_DIR
from cnn_xai_lab.xai import (
    compute_saliency_map,
    make_gradcam_heatmap,
    predict_probabilities,
    prepare_image,
    save_interpretability_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "model.keras")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "interpretability" / "single_prediction_panel.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = keras.models.load_model(args.model_path)

    with Image.open(args.image_path) as image:
        image_batch = prepare_image(image, image_size=IMAGE_SIZE)

    prediction = predict_probabilities(model, image_batch)
    saliency = compute_saliency_map(model, image_batch, prediction["predicted_index"])
    gradcam = make_gradcam_heatmap(model, image_batch, prediction["predicted_index"])

    save_interpretability_panel(
        image_batch=image_batch,
        saliency=saliency,
        gradcam=gradcam,
        output_path=args.output_path,
        title=f'Prediccion: {prediction["predicted_label"]}',
    )

    print("Prediction")
    print(f'female: {prediction["female_probability"]:.4f}')
    print(f'male:   {prediction["male_probability"]:.4f}')
    print(f'label:  {prediction["predicted_label"]}')
    print(f'panel:  {args.output_path}')


if __name__ == "__main__":
    main()
