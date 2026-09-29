import json
import random
import torch

from gliner2 import AutoExtractor
from peft import PeftModel

torch.set_num_threads(4)

METHODS = ["fakty", "analiza_zrodla", "rachunek", "interpretacja_danych",
           "wypowiedz", "gramatyka", "algorytm", "inne", "doswiadczenie",
           "dowod", "rysunek_techniczny"]
FORMS = ["krotka_odpowiedz", "wybor", "liczba", "wypracowanie", "prawda_falsz",
         "wyrazenie", "dowod", "rysunek", "inne", "kod"]
DIFFICULTY = ["łatwe", "bardzo łatwe", "średnie", "trudne"]

base = AutoExtractor.from_pretrained("fastino/GLiNER2.5-multi-Decide", map_location="cpu")
model = PeftModel.from_pretrained(base, "output/best")

rows = []
with open("data/val.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
random.seed(7); random.shuffle(rows)

for r in rows[:6]:
    text = r["input"]
    tasks = {c["task"]: c for c in r["output"]["classifications"]}
    schema = {t: {"labels": c["labels"]} for t, c in tasks.items()}
    pred = model.classify_text(text, schema)
    print("=" * 80)
    print("Q:", text[:180].replace("\n", " "))
    for t, c in tasks.items():
        gold = c["true_label"][0]
        print(f"  [{t}] gold={gold!r:20s} pred={pred.get(t)!r}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()