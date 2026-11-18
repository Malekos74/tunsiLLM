from pathlib import Path
import argparse
import json

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.utils.config import DATA_DIR, load_dataset_config

def download_one(name: str, cfg: dict):
    raw_dir = DATA_DIR / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    if cfg["kind"] == "huggingface":
        kwargs = {}
        if cfg.get("config"):
            kwargs["name"] = cfg["config"]
        if cfg.get("split"):
            kwargs["split"] = cfg["split"]
        ds = load_dataset(cfg["repo_id"], **kwargs)
        out = raw_dir / "data"
        ds.save_to_disk(str(out))
        print(f"{name}: saved Hugging Face dataset to {out}")
        print(ds)
    elif cfg["kind"] == "csv":
        path = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            repo_type="dataset",
        )
        df = pd.read_csv(path)
        out = raw_dir / cfg["filename"]
        df.to_csv(out, index=False)
        print(f"{name}: saved {len(df)} rows to {out}")
        print("Columns:", list(df.columns))
        print(df.head(5).to_string(index=False))
    else:
        raise ValueError(f"Unsupported dataset kind: {cfg['kind']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="Dataset key from configs/datasets.yaml, or 'all'")
    args = parser.parse_args()
    configs = load_dataset_config()
    names = list(configs) if args.dataset == "all" else [args.dataset]
    for name in names:
        if name not in configs:
            raise SystemExit(f"Unknown dataset '{name}'. Available: {list(configs)}")
        print(f"\n=== Downloading {name} ===")
        download_one(name, configs[name])

if __name__ == "__main__":
    main()
