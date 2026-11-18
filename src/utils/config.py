from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def load_dataset_config():
    path = ROOT / "configs" / "datasets.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]
