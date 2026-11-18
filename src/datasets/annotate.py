# uv run python -m src.datasets.annotate --input data/processed/dhiadev/annotated.csv
import argparse
import csv
import os
from pathlib import Path

DEFAULT_FILE = Path("data/processed/dhiadev/annotated.csv")

REQUIRED_COLUMNS = [
    "category",
    "darija_original",
    "english_original",
    "english_corrected",
    "darija_corrected",
    "status",
    "notes",
]

VALID_STATUSES = ["accepted", "corrected", "uncertain", "rejected"]


def save_csv(file_path, rows, fieldnames):
    """Save safely using a temporary file."""
    temp_path = file_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, file_path)


def get_input(prompt, current=""):
    """Prompt for a value. Enter keeps the existing value."""
    display = current if current else "(empty)"
    value = input(f"{prompt} [{display}]: ").strip()

    if value == "":
        return current

    if value == "!":
        return ""

    return value


def annotate(file_path, include_reviewed=False):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Ensure every row has all expected fields
    for row in rows:
        for col in fieldnames:
            row.setdefault(col, "")

    total = len(rows)
    reviewed = 0

    print("\nTunisian Arabizi Dataset Annotator")
    print("=" * 40)
    print(f"File: {file_path}")
    print(f"Total examples: {total}")
    print("\nCommands:")
    print("  Enter = keep existing value")
    print("  !     = clear a field")
    print("  s     = skip this example")
    print("  q     = quit")
    print("=" * 40)

    for index, row in enumerate(rows):
        status = row.get("status", "").strip().lower()

        if status and not include_reviewed:
            continue

        print(f"\n\nExample {index + 1}/{total}")
        print("-" * 40)
        print(f"Category:       {row['category']}")
        print(f"Original Darija: {row['darija_original']}")
        print(f"Original English:{row['english_original']}")

        print("\nYour corrections:")

        english = get_input(
            "Corrected English",
            row["english_corrected"],
        )

        # Allow quitting or skipping between fields
        if english.lower() == "q":
            break
        if english.lower() == "s":
            continue

        darija = get_input(
            "Corrected Darija (Arabizi)",
            row["darija_corrected"],
        )

        if darija.lower() == "q":
            break
        if darija.lower() == "s":
            continue

        print("\nStatus:")
        for i, option in enumerate(VALID_STATUSES, 1):
            print(f"  {i}. {option}")

        current_status = row["status"]
        status_input = input(
            f"Choose status (1-4) [{current_status or 'not set'}]: "
        ).strip().lower()

        if status_input == "q":
            break
        if status_input == "s":
            continue

        if status_input == "":
            new_status = "corrected"
        elif status_input in ["1", "2", "3", "4"]:
            new_status = VALID_STATUSES[int(status_input) - 1]
        elif status_input in VALID_STATUSES:
            new_status = status_input
        else:
            print("Invalid status. Example skipped.")
            continue

        notes = get_input("Notes", row["notes"])

        if notes.lower() == "q":
            break
        if notes.lower() == "s":
            continue
        if notes == "":
            notes = "personal spelling choice"

        # Update the row
        row["english_corrected"] = english.lower()
        row["darija_corrected"] = darija.lower()
        row["status"] = new_status
        row["notes"] = notes

        # Save immediately after every completed example
        save_csv(file_path, rows, fieldnames)

        reviewed += 1
        print("\nSaved!")

    print("\n" + "=" * 40)
    print("Annotation session finished.")
    print(f"Examples updated this session: {reviewed}")
    print(f"Dataset saved to: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Tunisian Arabizi dataset annotator."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to the annotated CSV.",
    )

    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Review examples that already have a status.",
    )

    args = parser.parse_args()
    annotate(args.input, args.include_reviewed)


if __name__ == "__main__":
    main()
