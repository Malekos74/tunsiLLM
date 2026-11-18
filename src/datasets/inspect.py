from pathlib import Path
import argparse
import json
import re
from collections import Counter

import pandas as pd
from datasets import load_from_disk

from src.utils.config import DATA_DIR

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DIGIT_RE = re.compile(r"\d")

def script_label(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"

def summarize_texts(texts):
    texts = [str(x) for x in texts if pd.notna(x)]
    lengths = [len(x) for x in texts]
    words = [len(x.split()) for x in texts]
    labels = Counter(script_label(x) for x in texts)
    report = {
        "rows": len(texts),
        "empty_rows": sum(not x.strip() for x in texts),
        "unique_exact": len(set(texts)),
        "duplicate_rows": len(texts) - len(set(texts)),
        "mean_chars": round(sum(lengths) / max(len(lengths), 1), 2),
        "median_chars": int(pd.Series(lengths).median()) if lengths else 0,
        "mean_words": round(sum(words) / max(len(words), 1), 2),
        "script_counts": dict(labels),
        "rows_with_url": sum(bool(URL_RE.search(x)) for x in texts),
        "rows_with_digits": sum(bool(DIGIT_RE.search(x)) for x in texts),
        "shorter_than_3_words": sum(n < 3 for n in words),
    }
    return report

def load_data(name):
    base = DATA_DIR / "raw" / name
    saved = base / "data"
    if saved.exists():
        ds = load_from_disk(str(saved))
        if hasattr(ds, "column_names"):
            if isinstance(ds.column_names, dict):
                first = next(iter(ds.keys()))
                return ds[first].to_pandas(), first
            return ds.to_pandas(), "dataset"
    csvs = list(base.glob("*.csv"))
    if csvs:
        return pd.read_csv(csvs[0]), csvs[0].name
    raise FileNotFoundError(f"No downloaded data found for '{name}'. Run download first.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    df, split = load_data(args.dataset)
    print(f"\nDataset: {args.dataset} | split/file: {split}")
    print(f"Shape: {df.shape}")
    print("Columns and dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(max_colwidth=160))

    report = {"dataset": args.dataset, "split": split, "columns": {
        c: str(t) for c, t in df.dtypes.items()
    }, "shape": list(df.shape)}

    for col in df.columns:
        if df[col].dtype == "object":
            report[col] = summarize_texts(df[col].dropna().astype(str).tolist())

    reports = DATA_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{args.dataset}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out}")

if __name__ == "__main__":
    main()
