
import json
from pathlib import Path

from src.baseline.ollama_client import translate

CONFIG_FILE = Path("configs/models.json")


def load_models():
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)["models"]


def main():
    models = load_models()

    print("\n" + "=" * 55)
    print("TUNISIZI - INTERACTIVE TRANSLATOR")
    print("=" * 55)

    print("\nAvailable models:")

    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")

    print("\nCommands:")
    print("  Type an English sentence to translate")
    print("  /model   Change the selected model")
    print("  /compare Compare all available models")
    print("  /quit    Exit")
    print("=" * 55)

    selected_model = models[0]

    print(f"\nSelected model: {selected_model}")

    while True:
        print()
        user_input = input("English > ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "/model":
            print("\nSelect a model:")

            for index, model in enumerate(models, start=1):
                print(f"{index}. {model}")

            choice = input("\nModel number > ").strip()

            try:
                selected_model = models[int(choice) - 1]
                print(f"Selected: {selected_model}")
            except (ValueError, IndexError):
                print("Invalid selection.")

            continue

        if user_input.lower() == "/compare":
            sentence = input("English > ").strip()

            if not sentence:
                continue

            for model in models:
                print(f"\n{model}:")
                try:
                    result = translate(
                        model_name=model,
                        english_text=sentence,
                    )
                    print(result)
                except Exception as error:
                    print(f"Error: {error}")

            continue

        try:
            result = translate(
                model_name=selected_model,
                english_text=user_input,
            )

            print(f"\n{selected_model}:")
            print(f"Tunizi > {result}")

        except Exception as error:
            print(f"\nError: {error}")
            print("Check that Ollama is running and the model is installed.")


if __name__ == "__main__":
    main()
