
from sacrebleu.metrics import BLEU, CHRF


def calculate_metrics(predictions, references):
    if not predictions or not references:
        return {
            "chrf": 0.0,
            "bleu": 0.0,
        }

    chrf_metric = CHRF()
    bleu_metric = BLEU()

    chrf_score = chrf_metric.corpus_score(
        predictions,
        [references],
    ).score

    bleu_score = bleu_metric.corpus_score(
        predictions,
        [references],
    ).score

    return {
        "chrf": round(chrf_score, 2),
        "bleu": round(bleu_score, 2),
    }
