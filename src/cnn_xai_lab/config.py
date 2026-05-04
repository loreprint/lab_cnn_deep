from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ("female", "male")
