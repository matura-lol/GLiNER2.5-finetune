# matura.lol GLiNER finetune

Multi-task classification of Polish matura exam questions using GLiNER2 with LoRA fine-tuning.

## Overview

matugen fine-tunes [GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) to classify matura (Polish high school exit exam) questions along four dimensions:

| Task | Description | Labels |
|------|-------------|--------|
| `topic` | Subject-specific topic (per informatory) | Varies by subject |
| `method` | Solution method | 11 labels |
| `answer_form` | Expected answer form | 10 labels |
| `difficulty` | Perceived difficulty | 4 levels |

The model is trained on data from [matura.lol](https://matura.lol) (Jev tags) and uses LoRA adapters for efficient fine-tuning.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

## Usage

### 1. Prepare data

Build training/validation JSONL from the matura.lol corpus and Jev tags:

```bash
uv run prepare_data.py [train_n] [val_n]
```

Defaults: 1500 train, 500 val. Output written to `data/train.jsonl` and `data/val.jsonl`.

### 2. Train

Fine-tune GLiNER2 with LoRA:

```bash
uv run train.py --epochs 2 --batch 4 --out output
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 2 | Number of training epochs |
| `--batch` | 4 | Batch size |
| `--max-len` | 256 | Max sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--max-steps` | -1 | Limit training steps |
| `--benchmark` | false | Disable eval during training |

### 3. Evaluate

```bash
uv run evaluate.py --model output/best --data data/val.jsonl --limit 200
```

Prints overall and per-task accuracy.

### 4. Classify

Run inference on raw text:

```bash
uv run classify.py "Oblicz pole koła o promieniu 5."
```

Output:

```
method: rachunek
answer_form: liczba
difficulty: łatwe
```

### 5. Inspect samples

Show gold vs. predicted labels for a few validation examples:

```bash
uv run samples.py
```

## Project structure

```
├── prepare_data.py   # Build train/val JSONL from corpus + tags
├── train.py          # LoRA fine-tuning with gliner2 ExtractorTrainer
├── evaluate.py       # Per-task accuracy evaluation
├── classify.py       # CLI inference on raw text
├── samples.py        # Print gold vs. predicted for sample questions
├── data/             # Generated train/val splits
├── dataset/          # matura.lol corpus, Jev tags, informatory topics
└── output/           # Trained LoRA adapters and checkpoints
```

## How it works

Each exam question is paired with its Jev tags and converted into a multi-task classification example. The four tasks share a single GLiNER2 encoder; LoRA adapters are applied to the encoder and classifier heads. At inference, a schema defines the label set per task, and `classify_text` returns the predicted label for each.

> **Note:** The `topic` task uses a per-subject label set derived from the CKE informatory, falling back to observed topics when an informatory entry is missing.
