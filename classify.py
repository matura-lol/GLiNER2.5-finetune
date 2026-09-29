import sys

from gliner2 import AutoExtractor
from peft import PeftModel

METHODS = ["fakty", "analiza_zrodla", "rachunek", "interpretacja_danych",
           "wypowiedz", "gramatyka", "algorytm", "inne", "doswiadczenie",
           "dowod", "rysunek_techniczny"]
FORMS = ["krotka_odpowiedz", "wybor", "liczba", "wypracowanie", "prawda_falsz",
         "wyrazenie", "dowod", "rysunek", "inne", "kod"]
DIFFICULTY = ["łatwe", "bardzo łatwe", "średnie", "trudne"]

base = AutoExtractor.from_pretrained("fastino/GLiNER2.5-multi-Decide", map_location="cpu")
model = PeftModel.from_pretrained(base, "output/best")

text = " ".join(sys.argv[1:])
schema = {
    "method": {"labels": METHODS},
    "answer_form": {"labels": FORMS},
    "difficulty": {"labels": DIFFICULTY},
}

pred = model.classify_text(text, schema)
for task, label in pred.items():
    print(f"{task}: {label}")
