---
license: agpl-3.0
base_model: fastino/GLiNER2.5-multi-Decide
library_name: peft
tags:
  - gliner2
  - lora
  - text-classification
  - token-classification
  - polish
  - matura
  - exam
  - named-entity-recognition
language:
  - pl
  - multilingual
pipeline_tag: token-classification
---

# matura.lol GLiNER2.5-multi-Decide finetune

**Multi-task classification of Polish matura exam questions** — a LoRA
fine-tune of [GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
on the [matura.lol datasets](https://huggingface.co/datasets/matura-lol/matura.lol-datasets)
(Jev tags), by [matura.lol](https://matura.lol).

The model classifies a Polish exam question along four dimensions in a single
forward pass:

| Task | Description | Labels |
|------|-------------|--------|
| `topic` | Subject-specific topic (per CKE informatory) | Varies by subject |
| `method` | Solution method | `fakty`, `analiza_zrodla`, `rachunek`, `interpretacja_danych`, `wypowiedz`, `gramatyka`, `algorytm`, `inne`, `doswiadczenie`, `dowod`, `rysunek_techniczny` |
| `answer_form` | Expected answer form | `krotka_odpowiedz`, `wybor`, `liczba`, `wypracowanie`, `prawda_falsz`, `wyrazenie`, `dowod`, `rysunek`, `inne`, `kod` |
| `difficulty` | Perceived difficulty | `łatwe`, `bardzo łatwe`, `średnie`, `trudne` |

## Model details

- **Base model:** [fastino/GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) (287M, multilingual)
- **Fine-tuning:** LoRA (`r=16`, `alpha=32`, `dropout=0.1`) on the encoder and
  classifier heads, ~2.7M trainable params (0.94% of the model)
- **Training data:** 1,500 questions from the [matura.lol corpus](https://huggingface.co/datasets/matura-lol/matura.lol-datasets),
  paired with their Jev tags; 500 held-out for validation
- **License:** AGPL-3.0 (see below)
- **Organizations:** [github.com/matura-lol](https://github.com/matura-lol) ·
  [huggingface.co/matura-lol](https://huggingface.co/matura-lol)

## License

This model is released under the **GNU Affero General Public License v3.0**
(AGPL-3.0), matching the [matura.lol datasets](https://huggingface.co/datasets/matura-lol/matura.lol-datasets).

The base model [`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
is Apache-2.0, which is permissive and permits relicensing derived works under
a different (including copyleft) license. The fine-tune itself — including
this adapter and the training setup — is distributed under AGPL-3.0, consistent
with the rest of the matura.lol open-source project.

## Usage

```python
from gliner2 import AutoExtractor
from peft import PeftModel

base = AutoExtractor.from_pretrained("fastino/GLiNER2.5-multi-Decide", map_location="cpu")
model = PeftModel.from_pretrained(base, "matura-lol/GLiNER2.5-multi-Decide-finetune")

METHODS = ["fakty", "analiza_zrodla", "rachunek", "interpretacja_danych",
           "wypowiedz", "gramatyka", "algorytm", "inne", "doswiadczenie",
           "dowod", "rysunek_techniczny"]
FORMS = ["krotka_odpowiedz", "wybor", "liczba", "wypracowanie", "prawda_falsz",
         "wyrazenie", "dowod", "rysunek", "inne", "kod"]
DIFFICULTY = ["łatwe", "bardzo łatwe", "średnie", "trudne"]

schema = {
    "method": {"labels": METHODS},
    "answer_form": {"labels": FORMS},
    "difficulty": {"labels": DIFFICULTY},
}

pred = model.classify_text("Oblicz pole koła o promieniu 5.", schema)
print(pred)
# {"method": "rachunek", "answer_form": "liczba", "difficulty": "łatwe"}
```

## Evaluation

Accuracy on the held-out validation split (200 questions per task):

| Task | Accuracy |
|------|---------:|
| `topic` | 0.505 |
| `method` | 0.555 |
| `answer_form` | 0.610 |
| `difficulty` | 0.540 |
| **Overall** | **0.552** |

## Attribution

If you reuse this model or the underlying data — including for AI training or
retrieval-augmented generation — credit [matura.lol](https://matura.lol) with a
link. The dataset is AGPL-3.0; individual pieces of text may carry their own
licences (CKE/OKE materials, third-party solutions).

## Links

- Site: [matura.lol](https://matura.lol)
- GitHub org: [github.com/matura-lol](https://github.com/matura-lol)
- Hugging Face org: [huggingface.co/matura-lol](https://huggingface.co/matura-lol)
- Dataset: [matura.lol datasets](https://huggingface.co/datasets/matura-lol/matura.lol-datasets)
- Base model: [fastino/GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
- Base paper: [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526) · [GLiNER2.5](https://arxiv.org/abs/2507.18546)