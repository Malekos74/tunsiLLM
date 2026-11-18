# Tunisian LLM – Phase 1: Dataset exploration

This repository downloads and profiles candidate Tunisian Arabic datasets. It does **not** train a model yet.

## Setup

Requires Python 3.10+ and `uv`.

```bash
uv sync
```

## Download datasets

Download one dataset at a time initially:

```bash
uv run python -m src.datasets.download --dataset tunizi
uv run python -m src.datasets.download --dataset dhiadev
uv run python -m src.datasets.download --dataset unified_derja
uv run python -m src.datasets.download --dataset linagora
```

The Linagora dataset has multiple configurations. The starter config selects `Derja_tunsi`; do not download every configuration until we have decided which subsets are relevant.

## Inspect

```bash
uv run python -m src.datasets.inspect --dataset tunizi
uv run python -m src.datasets.inspect --dataset dhiadev
uv run python -m src.datasets.inspect --dataset unified_derja
uv run python -m src.datasets.inspect --dataset linagora
```

Reports are written to `data/reports/`.

## Preprocess monolingual text

First inspect the column names. Then specify the text column explicitly where needed:

```bash
uv run python -m src.datasets.preprocess --dataset tunizi --text-column text
uv run python -m src.datasets.preprocess --dataset unified_derja --text-column text
```

For the Dhiadev parallel dataset, do not use the monolingual preprocessing script. We will create a separate parallel-data cleaner after confirming the actual column names and translation direction.

## Important data-handling rules

- Keep raw files unchanged.
- Do not normalize Arabizi spellings yet.
- Keep Arabic-script and Latin-script text distinguishable.
- Do not merge datasets blindly: the unified corpus already aggregates several sources, so combining it with those sources can create duplicates.
- Record dataset licenses and provenance before training or redistribution.
- Keep private WhatsApp exports out of this repository and out of version control.

## Candidate datasets

- `arbml/TUNIZI`: 3,000-row Arabizi sentiment dataset; not a general translation corpus.
- `Dhiadev-tn/tunisian-darija-english`: 553 Arabizi-English pairs; currently documented as CC-BY-NC-SA-4.0.
- `hamzabouajila/tunisian-derja-unified-raw-corpus`: large raw monolingual corpus, CC-BY-SA-4.0.
- `linagora/Tunisian_Derja_Dataset`: multi-configuration corpus, CC-BY-SA-4.0.

Always re-check the dataset cards and licenses before use.
