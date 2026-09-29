---
license: agpl-3.0
language:
  - pl
pretty_name: matura.lol datasets
tags:
  - polish
  - matura
  - exam
  - egzamin
  - CKE
  - OCR
  - jev
  - named-entity-recognition
  - text-classification
size_categories:
  - 10K<n<100K
---

# matura.lol datasets

Open datasets behind **[matura.lol](https://matura.lol)** — a read-only
full-text search engine over the Polish exam corpus (CKE matura, egzamin
ósmioklasisty, egzamin gimnazjalny, vocational arkusze and próbne arkusze).

Source PDFs (CKE/OKE arkusze) are **not** included, and neither are any images.
Everything here is plain, uncompressed JSON/JSONL, produced by the pipeline.

This Hugging Face repo hosts the **uncompressed** copies of the data that is
also published (zstd-compressed) in the
[`matura-lol/datasets`](https://github.com/matura-lol/datasets) GitHub
repository. Use this repo when you want the ready-to-read JSON without
decompressing anything.

## Layout

```
imports/    processed enrichments folded onto the corpus
            inventory.json          paper/PDF metadata + hashes
            math_import.json        curated math index
            site_import.json        site problems + SVG solutions
            odrabiamy_import.json   answers/worked solutions (odrabiamy)
            matematykaorg_import.json
            maturazai_import.json   AI answers/solutions
            zadaniazmatur_import.json
            maturaonline_import.json
            szybkiekorepetycje_import.json
            biologhelp_import.json  CKE marking schemes (bio/chem)
            pytania_import.json
sources/    raw scraped ledgers (one JSONL row per task/arkusz)
            arkusze.jsonl       54-row CKE paper ledger
            arkusze.full.jsonl  478-row full arkusz metadata (60 MB)
corpus/     segments/<letter>/<paper-id>.json — per-paper question segmentation
            sharded by first letter of the paper id (≤ 10k files per dir)
            (question text, page + y span, points, type, key links)
jev/        TypeSafe Jev tagging outputs (see below)
db/         search_matugen.sql — plain-SQL pg_dump (PostgreSQL 16) of the live
            matura.lol search database: papers, files, questions (with the
            Jev columns difficulty / solution_method / answer_form / jev_group /
            group_label / meta_tags), question_group, search_token.
            Restore: createdb search_matugen && psql -d search_matugen -f db/search_matugen.sql
            (the pg_trgm extension is needed; if an older psql rejects the
            `\restrict` lines, delete them). Generated tsvector search columns
            are rebuilt on restore; user-vote tables are schema only, no data.
            The 32-number Jev `vector` is NOT in the database, only in jev/tags.jsonl.
```

## How to navigate this repo

* **Format.** Everything is plain JSON (`.json`/`.jsonl`), no compression.
  `jq` works directly; `pandas.read_json(path, lines=True)` reads the JSONL rows.
* **The corpus.** The per-paper question segmentation is in `corpus/segments/`,
  sharded into first-letter subdirectories (`corpus/segments/a/`, `corpus/segments/m/`,
  …) so no directory exceeds Hugging Face's 10 000-file limit. Each file is one
  paper (the paper id from `imports/inventory.json`); the shard letter is just the
  first character of the paper id.
* **IDs are the join key.** Every question has an id of the form
  `<paper-id>/zad/<number>` — the same id appears in `segments/`, in
  `imports/*.json`, and in `jev/tags.jsonl`. The
  `imports/*.json` files map per-source enrichments (answers, solutions,
  topics) onto those ids; `jev/*.jsonl` maps the Jev decisions onto them.
* **Where each answer/solution comes from.** A question row carries
  `answer_source` / `answer_text_source` / `solution_source` / `topics_source`
  (values like `cke`, `odrabiamy`, `matematykaorg`, `maturazai`, `ai`,
  `jev`) — see `imports/` per source.
* **The Jev layer** is optional enrichment: join `jev/tags.jsonl` by `id`
  to get topic/difficulty/method/group, `jev/answer_checks.jsonl` for the
  answer-fit signal, and `jev/mismatches.jsonl` + `jev/reassignments.jsonl`
  for the flagged/repair rows.

## jev/

Jev (System One) tags each question with typed decisions: the official
**dział** (from the CKE informatory), **difficulty**, **solution method**,
**answer form**, **bloom** level, estimated **time**, and capability/tool
flags. See https://github.com/matura-lol/Jev-categorise for the tooling.

| file | rows | what |
| --- | --- | --- |
| `informator_topics.json` | — | official per-subject dział taxonomy extracted from the CKE informatories |
| `tags.jsonl` | ~47k | one Jev tag record per question (topic + probabilities, difficulty, method, answer form, bloom, time, capabilities, capability vector, group_id) |
| `tags.sample.jsonl` | 500 | plain-text sample of the same schema |
| `answer_checks.jsonl` | ~41k | per-question Noul: does the stored answer/solution fit the task? |
| `mismatches.jsonl` | — | the low-probability subset of `answer_checks` for review |
| `reassignments.jsonl` | — | proposals matching a mismatched answer to the question it actually solves (one-to-one, never applied automatically) |

### Tag schema (one row)

```json
{
  "id": "informator-maturalny-matematyka-2023-poziom-podstawowy/zad/19.2",
  "subject": "matematyka",
  "topic": "Funkcje",
  "topic_confidence": 1.0,
  "topic_probs": {"Funkcje": 1.0, "...": 0.0},
  "difficulty": 2,
  "difficulty_label": "średnie",
  "method": "rachunek",
  "answer_form": "liczba",
  "capabilities": {"wymaga_rachunku": 0.98, "...": 0.0},
  "bloom": 3,
  "time_band": 0,
  "needs_formula_sheet": 0.2,
  "needs_calculator": 0.1,
  "visual": 0.08,
  "subject_ok": 0.99,
  "vector": [0.0, 1.0, "..."],
  "group_id": "matematyka/funkcje/rachunek/2/1e",
  "group_label": "Funkcje · rachunek · średnie",
  "model": "jev-1.13",
  "input_tokens": 2612
}
```

`group_id` is the discrete signature `subject/topic/method/difficulty/capability-bits`;
questions sharing it are solved the same way.

## Source & licence

The dataset (the compilation, the segmentation, and the model-derived metadata)
is released under the **GNU Affero General Public License v3.0** — see
[`LICENSE`](LICENSE).

Individual pieces of **text** inside the data may be licensed differently and
remain under their respective licences; the AGPL does not override them.
In particular:

- exam papers and marking schemes are public materials of the **Centralna
  Komisja Egzaminacyjna (CKE)** and the **Okręgowe Komisje Egzaminacyjne (OKE)**
  and remain theirs;
- question text, answers and worked solutions collected from third-party sites
  (odrabiamy, matematyka.org.pl, zadaniazmatur, matura-online, biologhelp,
  szybkiekorepetycje, maturazai and others) remain subject to those sites'
  terms and licences;
- the code that produced this data lives at
  https://github.com/matura-lol/Jev-categorise (MIT).

If you reuse this data, credit [matura.lol](https://matura.lol) with a link.