import argparse
import json

from gliner2 import AutoExtractor
from peft import PeftModel

METHODS = ["fakty", "analiza_zrodla", "rachunek", "interpretacja_danych",
           "wypowiedz", "gramatyka", "algorytm", "inne", "doswiadczenie",
           "dowod", "rysunek_techniczny"]
FORMS = ["krotka_odpowiedz", "wybor", "liczba", "wypracowanie", "prawda_falsz",
         "wyrazenie", "dowod", "rysunek", "inne", "kod"]
DIFFICULTY = ["łatwe", "bardzo łatwe", "średnie", "trudne"]
INFORMATOR = json.load(open("/home/dj/Documents/ml.matugen/dataset/jev/informator_topics.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/best")
    ap.add_argument("--base", default="fastino/GLiNER2.5-multi-Decide")
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    base = AutoExtractor.from_pretrained(args.base, map_location="cpu")
    model = PeftModel.from_pretrained(base, args.model)

    obs_topics = {}
    for fn in ["/home/dj/Documents/ml.matugen/data/val.jsonl",
               "/home/dj/Documents/ml.matugen/data/train.jsonl"]:
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                for c in d["output"]["classifications"]:
                    if c["task"] == "topic":
                        obs_topics.setdefault(c["labels"][0], c["labels"])

    rows = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rows.append(json.loads(line))

    total = 0
    correct = 0
    by_task = {}
    for r in rows:
        text = r["input"]
        tasks = {}
        for c in r["output"]["classifications"]:
            tasks[c["task"]] = c
        schema = {}
        for task, c in tasks.items():
            schema[task] = {"labels": c["labels"]}
        pred = model.classify_text(text, schema)
        for task, c in tasks.items():
            gold = c["true_label"][0]
            got = pred.get(task)
            by_task.setdefault(task, {"corr": 0, "tot": 0})
            by_task[task]["tot"] += 1
            total += 1
            if got == gold:
                by_task[task]["corr"] += 1
                correct += 1
    print(f"OVERALL accuracy: {correct}/{total} = {correct / total:.3f}")
    for task, s in by_task.items():
        print(f"  {task}: {s['corr']}/{s['tot']} = {s['corr'] / s['tot']:.3f}")


if __name__ == "__main__":
    main()