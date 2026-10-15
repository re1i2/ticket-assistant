# Intelligent Support Ticket Assistant (English + Arabic)

**Use case:** 1 · Intelligent Support Ticket Assistant · **Demo:** `python app.py`
Capstone for SDAIA Academy *Applied NLP Engineering* (https://sdaiaacademy.github.io/applied-nlp/).

## تشغيل سريع (Arabic quick start)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.prepare_data          # يحمّل بيانات Bitext ويجهّز التقسيم (يحتاج إنترنت)
python -m src.train --config configs/final.yaml   # تدريب الموديل (يفضّل GPU / Colab)
python app.py                       # الواجهة على http://127.0.0.1:7860
```

لو ما عندك GPU: استخدم `configs/quick.yaml` (إيبوك واحد) أو شغّل `notebooks/colab_run.ipynb` على Colab مع T4.
بدون تدريب، `python app.py` يشتغل أيضاً بالموديل الأساسي (e5 + LR) ويكتب لك في الطرفية أي مصنّف يستخدم.

## Problem

Support teams receive English and Arabic (MSA and Gulf dialect) messages and must route, prioritise and answer them
quickly. **Our system takes** a customer message **and produces** an intent, a category, a priority, similar
resolved tickets, relevant help-center sections and a grounded, cited draft reply (or an escalation to a human when
it is not confident), **measured by** macro-F1 of the intent on three held-out test sets, plus retrieval Hit@k/MRR
and reply-grounding checks.

## Requirements checklist 

| Minimum requirement | Where |
|---|---|
| 1. Working end-to-end pipeline + demo | `src/pipeline.py` (`assist(text)`), `app.py` (Gradio) |
| 2. Representation + decision/generation component | e5 embeddings (`src/utils.py`, `src/retrieval.py`); classifier (`src/classifier.py`); grounded reply (`src/pipeline.py`) |
| 3. Baseline + ≥1 improvement on the same held-out test set | `src/run_experiments.py`: `tfidf_lr` → `e5_lr` (course baseline) → `finetuned_*`; retrieval: dense → BM25 → hybrid → hybrid+rerank |
| 4. Fine-tuning (encoder or LoRA) or justified alternative | `src/train.py` + `configs/*.yaml` (encoder fine-tuning, early stopping, temperature calibration) |
| 5. Metrics by slice + error analysis with ≥10 categorised failures | `eval/evaluate.py` (slices: set, lang, dialect), `eval/error_analysis.py` (exports a CSV you categorise by hand) |

## Data

| Split | Size | Languages / slices | Source |
|---|---|---|---|
| train | ≈ 5,400 (200 / intent) | English | Bitext customer-support dataset (CDLA-Sharing-1.0), via `src/prepare_data.py` |
| validation | ≈ 1,080 (40 / intent) | English | same |
| test `en_test` | ≈ 1,080 (40 / intent) | English, **templated** (optimistic) | same |
| test `ar_test` | 108 | Arabic: 54 MSA + 54 Gulf, hand-written | `data/arabic_tickets.csv` (course repo) |
| test `hard_en` | 49 | Messy English: typos, anger, run-ons, hand-written | `data/hard_en_tickets.csv` (written for this project) |
| history (search index) | up to 8,000 resolved tickets with agent replies | English | Bitext |
| help center (RAG KB) | 10 articles, 33 sections | English, fictional store DemoMart | `data/kb/` (course repo) |

Exact counts after your run are written to `data/split_report.json`. 27 intents, 11 categories, priority = business rule
(`priority_for` in `src/prepare_data.py`).

**Known limitations.** Bitext is templated, so `en_test` scores overstate real-world quality. There is no Arabic
training data (zero-shot cross-lingual transfer from English). The Arabic and hard-English sets are small, so read the
bootstrap confidence intervals, not just point estimates. The KB is English-only; Arabic questions are matched
cross-lingually.

**Leakage control.** Before splitting, tickets are normalised (ids, numbers, names masked) and de-duplicated, so the same
template instance cannot land in two splits. `prepare_data` verifies the masked-sentence overlap between train/val/test is
zero and fails otherwise. The Arabic and hard-English sets are never used for training or tuning (`train.py` refuses an
`extra_train_csv` that overlaps them). Hyperparameters and early stopping use the validation set only.

## Approach

```
message ─► classifier (fine-tuned multilingual-e5-small, calibrated) ─► intent + confidence
                │  confidence < tau  →  route = human_review (abstain)
                ├─► category (intent → category map) and priority (business rule + urgency cues)
                ├─► hybrid retrieval (dense e5 + BM25, RRF fusion, cross-encoder rerank) over
                │     • resolved tickets (similar cases)   • help-center sections
                └─► grounded reply: LLM → JSON {reply, citations, needs_human} → Pydantic validation
                      → citation check → "numbers/links must appear in the context" check
                      (no LLM, or any failure → extractive reply or escalate to a human)
```

Why these models: `multilingual-e5-small` covers Arabic and English, is small enough for CPU inference, and is used both
for retrieval and (fine-tuned) for classification. `distilbert-base-multilingual-cased` is the alternative encoder in
`configs/distilbert.yaml` for the model-selection comparison. The LLM (Qwen2.5-1.5B on GPU / 0.5B on CPU, or any
OpenAI-compatible endpoint via `.env`) is used only for wording; every claim must be backed by retrieved text, and the
decision to automate vs. escalate does not depend on it.

Guardrails: the ticket is passed as data inside `<ticket>` tags; messages that look like prompt injection
(`ignore previous instructions`, …) are flagged, never sent to the generator, and escalated; phone numbers, emails and
card-like numbers are masked before text goes to an external endpoint; replies with unsupported numbers or links are
escalated.

## Results (held-out test sets)

`python -m src.run_experiments` writes `eval/results.md`. It reports, for each system, macro-F1 with bootstrap
95% CIs by test set / language / dialect, accuracy, category and priority accuracy, coverage and selective accuracy at the
abstention threshold, single-ticket latency (p50/p95), and the retrieval ablation. With `--llm` it also scores reply
generation (valid reply rate, gold section cited, escalation rate on unanswerable questions).

| System | Macro-F1 en_test | Macro-F1 hard_en | Macro-F1 Arabic | p95 latency | Notes |
|---|---|---|---|---|---|
| Lexical floor: char TF-IDF + LR | | | | | |
| Baseline: e5 embeddings + LR | | | | | course baseline |
| Fine-tuned multilingual-e5-small | | | | | `configs/final.yaml` |
| Fine-tuned DistilBERT-multilingual | | | | | `configs/distilbert.yaml` |

Fine-tuning is measured as baseline row → fine-tuned row on the same test sets.

## Error analysis

```bash
python -m eval.error_analysis eval/predictions/finetuned_encoder.csv --n 40
```

This writes `eval/error_analysis.csv` with a heuristic `auto_cause` for each failure and a `manual_cause` column for the reviewed causes.

| Cause | Count (of N reviewed) | Example | What we'd do |
|---|---|---|---|
| Label noise | | | |
| Ambiguous / overlapping intents | | | |
| Missing coverage (dialect, typos…) | | | |
| Model error | | | |
| Out of scope | | | |

## Reproduce

```bash
pip install -r requirements.txt          # or: uv sync
python -m src.prepare_data
python -m src.train --config configs/final.yaml
python -m src.run_experiments --tau 0.5
python -m eval.evaluate eval/predictions/finetuned_encoder.csv --slice set --slice dialect --tau 0.5
python app.py
```

Or everything at once: `bash scripts/run_all.sh`. Sample inputs and outputs are written to `eval/sample_outputs.json` by `python scripts/make_samples.py`. Seeds: 42 everywhere (`configs/*.yaml`). Pin the model commit by setting
`revision:` in the config. Optional stronger LLM: copy `.env.example` to `.env`, then `python app.py --llm`.

Offline plumbing tests (no downloads, synthetic data): `python tests/test_smoke.py`.

## Repository layout

```
app.py                  Gradio demo
src/prepare_data.py     download, de-duplicate, split, Arabic + hard-English sets, leakage report
src/train.py            encoder fine-tuning + temperature calibration
src/classifier.py       tfidf_lr | e5_lr | finetuned behind one interface
src/retrieval.py        dense | BM25 | hybrid (RRF) | hybrid + cross-encoder rerank
src/pipeline.py         assist(text) -> dict, guardrails, grounded reply
src/run_experiments.py  all comparisons -> eval/results.md, eval/predictions/*.csv
eval/evaluate.py        per-slice metrics with bootstrap CIs
eval/error_analysis.py  failure export for manual categorisation
configs/                final.yaml, distilbert.yaml, quick.yaml
data/                   KB, Arabic tickets, hard English tickets, rag_eval.csv
notebooks/colab_run.ipynb
scripts/run_all.sh, scripts/make_samples.py
tests/test_smoke.py
```

## Limitations and next steps

What would break first with real customer data: dialects beyond Gulf, code-switching (Arabizi), multi-intent messages (one
label per ticket today), and tickets about products missing from the KB (we escalate rather than guess, but coverage drops).
With two more weeks: collect and label real Arabic tickets for training (via `extra_train_csv`), add a multi-label
intent head, LoRA-tune a small LLM on resolved replies and compare it with RAG, translate-and-retrieve for Arabic KB
queries, and calibrate `tau` per language from a labelled validation set in each language.

Data credits: Bitext customer-support dataset (CDLA-Sharing-1.0); the Arabic tickets, KB and `rag_eval.csv` come from the
course repository https://github.com/SDAIAAcademy/applied-nlp.
