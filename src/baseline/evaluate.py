
import csv
import json
import re
from pathlib import Path

from src.baseline.ollama_client import translate
from src.baseline.metrics import calculate_metrics


CONFIG_FILE = Path("configs/models.json")
TEST_FILE = Path("data/processed/dhiadev/test.csv")
OUTPUT_DIR = Path("results/baseline")


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            "Test dataset not found. Run prepare_data.py first."
        )

    with TEST_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_filename(model_name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)


def evaluate_model(model_name, examples, config):
    print(f"\nEvaluating: {model_name}")
    print("-" * 50)

    predictions = []
    references = []
    results = []

    for index, example in enumerate(examples, start=1):
        english = example["english"]
        reference = example["darija"]

        print(
            f"[{index}/{len(examples)}] "
            f"Translating: {english}"
        )

        try:
            prediction = translate(
                model_name=model_name,
                english_text=english,
                temperature=config.get("temperature", 0.0),
                num_predict=config.get("num_predict", 256),
            )

        except Exception as error:
            print(f"ERROR: {error}")
            prediction = ""

        predictions.append(prediction)
        references.append(reference)

        results.append({
            "category": example.get("category", ""),
            "english": english,
            "reference_darija": reference,
            "predicted_darija": prediction,
            "human_rating": "",
            "human_notes": "",
        })

        print(f"  Reference:  {reference}")
        print(f"  Prediction: {prediction}")

    # Calculate metrics only for successful generations.
    valid_pairs = [
        (prediction, reference)
        for prediction, reference in zip(predictions, references)
        if prediction
    ]

    if valid_pairs:
        valid_predictions, valid_references = zip(*valid_pairs)
        scores = calculate_metrics(
            list(valid_predictions),
            list(valid_references),
        )
    else:
        scores = {"chrf": 0.0, "bleu": 0.0}

    scores["model"] = model_name
    scores["examples"] = len(examples)
    scores["successful_generations"] = len(valid_pairs)

    output_file = OUTPUT_DIR / f"{safe_filename(model_name)}.csv"

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category",
                "english",
                "reference_darija",
                "predicted_darija",
                "human_rating",
                "human_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nScores for {model_name}:")
    print(f"  chrF: {scores['chrf']}")
    print(f"  BLEU: {scores['bleu']}")
    print(f"  Saved to: {output_file}")

    return scores


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    examples = load_test_data()

    if not examples:
        raise ValueError("The test dataset is empty.")

    print("=" * 50)
    print("TUNISIAN ARABIZI ZERO-SHOT BENCHMARK")
    print("=" * 50)

    print(f"Test examples: {len(examples)}")

    all_scores = []

    for model_name in config["models"]:
        scores = evaluate_model(
            model_name,
            examples,
            config,
        )
        all_scores.append(scores)

    summary_file = OUTPUT_DIR / "summary.csv"

    with summary_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "examples",
                "successful_generations",
                "chrf",
                "bleu",
            ],
        )
        writer.writeheader()
        writer.writerows(all_scores)

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    for score in all_scores:
        print(
            f"{score['model']:20} "
            f"chrF: {score['chrf']:6} "
            f"BLEU: {score['bleu']:6}"
        )

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
