from pathlib import Path
import argparse
import pandas as pd
from datasets import load_from_disk

from src.utils.config import DATA_DIR

def load_frame(name):
    base = DATA_DIR / "raw" / name
    saved = base / "data"
    if saved.exists():
        ds = load_from_disk(str(saved))
        if isinstance(ds, dict):
            ds = ds[next(iter(ds.keys()))]
        return ds.to_pandas()
    csvs = list(base.glob("*.csv"))
    if csvs:
        return pd.read_csv(csvs[0])
    raise FileNotFoundError(f"No downloaded data for {name}")

def clean_text(value):
    if pd.isna(value):
        return None
    text = str(value).replace("\ufeff", "").strip()
    return text if text else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--text-column", default=None)
    args = parser.parse_args()

    df = load_frame(args.dataset)
    text_col = args.text_column
    if text_col is None:
        candidates = [c for c in df.columns if c.lower() in
                      {"text", "darija", "tunisian", "tunisian_arabic"}]
        if len(candidates) != 1:
            raise SystemExit(
                f"Could not safely infer text column. Columns: {list(df.columns)}. "
                "Pass --text-column explicitly."
            )
        text_col = candidates[0]

    df["text"] = df[text_col].map(clean_text)
    df = df[df["text"].notna()].copy()
    df = df.drop_duplicates(subset=["text"])
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}.jsonl"
    df[["text"]].to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} unique non-empty text rows to {out}")

if __name__ == "__main__":
    main()
