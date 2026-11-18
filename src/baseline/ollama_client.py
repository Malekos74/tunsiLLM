
import ollama

from src.baseline.prompts import build_messages


def translate(
    model_name,
    english_text,
    temperature=0.0,
    num_predict=256,
):
    """
    Translate one English sentence into Tunisian Arabizi.
    """

    response = ollama.chat(
        model=model_name,
        messages=build_messages(english_text),
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
        stream=False,
    )

    return response.message.content.strip()
