# Use the annotated dataset to create train/test splits for model training and evaluation.
import csv
import random
from pathlib import Path

INPUT_FILE = Path("data/processed/dhiadev/annotated.csv")
OUTPUT_DIR = Path("data/processed/dhiadev")

TEST_SIZE = 0.20
RANDOM_SEED = 42


def read_examples():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "english_original",
            "darija_original",
            "english_corrected",
            "darija_corrected",
            "status",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        examples = []
        seen = set()

        for row in reader:
            status = (row.get("status") or "").strip().lower()

            if status in {"uncertain", "rejected"}:
                continue

            # Do not silently include examples that have not been reviewed.
            if status not in {"accepted", "corrected"}:
                continue

            english = (
                row.get("english_corrected", "").strip()
                or row.get("english_original", "").strip()
            )

            darija = (
                row.get("darija_corrected", "").strip()
                or row.get("darija_original", "").strip()
            )

            if not english or not darija:
                continue

            # Remove exact duplicate pairs, ignoring capitalization
            key = (english.casefold(), darija.casefold())

            if key in seen:
                continue

            seen.add(key)

            examples.append({
                "category": row.get("category", "").strip(),
                "english": english,
                "darija": darija,
            })

    return examples


def write_csv(path, examples):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "english", "darija"],
        )
        writer.writeheader()
        writer.writerows(examples)


def main():
    examples = read_examples()

    if len(examples) < 5:
        raise ValueError(
            f"Only {len(examples)} usable examples found. "
            "Check your annotation statuses and CSV."
        )

    random.Random(RANDOM_SEED).shuffle(examples)

    test_count = max(1, round(len(examples) * TEST_SIZE))

    test_examples = examples[:test_count]
    train_examples = examples[test_count:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT_DIR / "train.csv", train_examples)
    write_csv(OUTPUT_DIR / "test.csv", test_examples)

    print("Dataset preparation complete.")
    print(f"Total examples: {len(examples)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
