"""Build GLiNER2 multi-task classification training data from matura.lol Jev tags.

Each question (from corpus/segments) joined with jev/tags.jsonl by id becomes
one InputExample with 4 classification tasks:
  topic        - dział from the CKE informatory (per-subject label set)
  method       - solution method (fixed 11 labels)
  answer_form  - answer form (fixed 10 labels)
  difficulty   - 1 of 4 difficulty labels

Output: JSONL with the raw-dict format the ExtractorTrainer accepts:
  {"input": text, "output": {"classifications": [{"task":..., "labels":[...],
   "true_label": [...], "multi_label": false}, ...]}}
"""
import json
import os
import random
import sys

SEED = 42
train_n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
val_n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SEG_DIR = "/home/dj/Documents/ml.matugen/dataset/corpus/segments"
TAGS_PATH = "/home/dj/Documents/ml.matugen/dataset/jev/tags.jsonl"
INFORMATOR = "/home/dj/Documents/ml.matugen/dataset/jev/informator_topics.json"
OUT_DIR = "/home/dj/Documents/ml.matugen/data"

METHODS = ["fakty", "analiza_zrodla", "rachunek", "interpretacja_danych",
           "wypowiedz", "gramatyka", "algorytm", "inne", "doswiadczenie",
           "dowod", "rysunek_techniczny"]
FORMS = ["krotka_odpowiedz", "wybor", "liczba", "wypracowanie", "prawda_falsz",
         "wyrazenie", "dowod", "rysunek", "inne", "kod"]
DIFFICULTY = ["łatwe", "bardzo łatwe", "średnie", "trudne"]

random.seed(SEED)
informator = json.load(open(INFORMATOR))

# observed topics per subject (for subjects missing an informator entry)
obs_topics: dict[str, set] = {}
tags: dict[str, dict] = {}
with open(TAGS_PATH) as f:
    for line in f:
        d = json.loads(line)
        tags[d["id"]] = d
        subj = d.get("subject")
        top = d.get("topic")
        if subj and top:
            obs_topics.setdefault(subj, set()).add(top)

def topic_labels(subject: str, topic: str) -> list[str] | None:
    base = list(informator.get(subject, obs_topics.get(subject, [])))
    if not base:
        return None
    labels = sorted(set(base) | {topic})
    return labels

records = []
for fn in os.listdir(SEG_DIR):
    with open(os.path.join(SEG_DIR, fn)) as f:
        paper = json.load(f)
    for q in paper.get("questions", []):
        tag = tags.get(q["id"])
        if tag is None:
            continue
        text = (q.get("text") or "").strip()
        if len(text) < 10:
            continue
        labels = topic_labels(tag.get("subject"), tag.get("topic"))
        if not labels:
            continue
        classifications = [
            {"task": "topic", "labels": labels,
             "true_label": [tag["topic"]] if tag.get("topic") else ["inne"]},
            {"task": "method", "labels": METHODS,
             "true_label": [tag["method"]] if tag.get("method") else ["inne"]},
            {"task": "answer_form", "labels": FORMS,
             "true_label": [tag["answer_form"]] if tag.get("answer_form") else ["inne"]},
            {"task": "difficulty", "labels": DIFFICULTY,
             "true_label": [tag["difficulty_label"]] if tag.get("difficulty_label") else ["średnie"]},
        ]
        # drop tasks whose true label is not in the label set
        classifications = [c for c in classifications if c["true_label"][0] in c["labels"]]
        if len(classifications) < 2:
            continue
        records.append({
            "input": text,
            "output": {"classifications": classifications},
        })

print("total usable records:", len(records))
random.shuffle(records)

os.makedirs(OUT_DIR, exist_ok=True)

val_n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
train, val = records[:train_n], records[train_n:train_n + val_n]

with open(f"{OUT_DIR}/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{OUT_DIR}/val.jsonl", "w") as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(train)} train, {len(val)} val to {OUT_DIR}")

from collections import Counter
methods = Counter()
for r in val:
    for c in r["output"]["classifications"]:
        if c["task"] == "method":
            methods[c["true_label"][0]] += 1
print("val method dist:", dict(methods))