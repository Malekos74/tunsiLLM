import ollama
from src.baseline.prompts import build_messages


def translate(
    model_name,
    english_text,
    temperature=0.0,
    num_predict=1024,
    show_thinking=True,
):
    response = ollama.chat(
        model=model_name,
        messages=build_messages(english_text),
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
        stream=False,
    )

    thinking = getattr(response.message, "thinking", None) or ""
    answer = response.message.content or ""

    if not answer.strip():
        raise RuntimeError(
            f"{model_name} returned no final answer. "
            f"Generation stopped because: {response.done_reason}. "
            f"Thinking tokens generated: {len(thinking.split())}"
        )

    # return {
    #     "thinking": thinking.strip() if show_thinking else "",
    #     "translation": answer.strip(),
    # }
    return answer.strip()
