from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import classifier as C
from . import utils as U
from .metrics import md_table, slice_table
from .pipeline import TicketAssistant
from .prepare_data import priority_for
from .retrieval import Retriever, kb_index_text


def load_test_sets() -> pd.DataFrame:
    parts = []
    for split, name in [("test", "en_test"), ("ar_test", "ar_test"), ("hard_test", "hard_en")]:
        df = U.load_tickets(split)
        df["set"] = name
        if "dialect" not in df:
            df["dialect"] = "templated_en"
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def predict_system(clf, test: pd.DataFrame, labels: dict) -> pd.DataFrame:
    probs = clf.predict_proba(test.text.tolist())
    idx = probs.argmax(1)
    out = test[["ticket_id", "set", "lang", "dialect", "text"]].copy()
    out["y_true"] = test.intent.values
    out["y_pred"] = [str(clf.classes_[i]) for i in idx]
    out["confidence"] = probs.max(1).round(4)
    i2c = labels["intent_to_category"]
    out["category_true"] = test.category.values
    out["category_pred"] = out.y_pred.map(i2c)
    out["priority_true"] = test.priority.values
    out["priority_pred"] = [priority_for(p, t) for p, t in zip(out.y_pred, out.text)]
    return out


def classification_latency(clf, texts: list[str], n: int = 100) -> dict:
    texts = texts[:n]
    for t in texts[:3]:
        clf.predict_proba([t])
    times = []
    for t in texts:
        t0 = time.perf_counter()
        clf.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1e3)
    return U.latency_stats(times)


def find_finetuned() -> dict[str, Path]:
    found = {}
    if U.ARTIFACTS.exists():
        for meta in sorted(U.ARTIFACTS.glob("*/ticket_meta.json")):
            found[f"finetuned_{meta.parent.name}"] = meta.parent
    return found


def retrieval_ablation() -> pd.DataFrame:
    kb = U.load_kb()
    ids = [s["id"] for s in kb]
    ev = pd.read_csv(U.DATA / "rag_eval.csv")
    ev["gold_section"] = ev.gold_section.fillna("")
    ans = ev[ev.gold_section != ""]
    retr = Retriever([kb_index_text(s) for s in kb], cache_key="kb")
    rows = []
    for mode in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        for lang, g in [("all", ans)] + list(ans.groupby("lang")):
            hits1 = hits3 = 0
            rr = []
            for q, gold in zip(g.question, g.gold_section):
                ranked = [ids[i] for i in retr.search(q, 10, mode)]
                hits1 += gold in ranked[:1]
                hits3 += gold in ranked[:3]
                rr.append(1 / (ranked.index(gold) + 1) if gold in ranked else 0.0)
            rows.append({"mode": mode, "lang": lang, "n": len(g), "hit@1": round(hits1 / len(g), 3),
                         "hit@3": round(hits3 / len(g), 3), "MRR": round(float(np.mean(rr)), 3)})
    return pd.DataFrame(rows)


def generation_eval(assistant: TicketAssistant) -> pd.DataFrame:
    ev = pd.read_csv(U.DATA / "rag_eval.csv")
    ev["gold_section"] = ev.gold_section.fillna("")
    rows = []
    for q, lang, gold in zip(ev.question, ev.lang, ev.gold_section):
        out = assistant(q)
        rows.append({"question": q, "lang": lang, "answerable": bool(gold), "mode": out.get("mode"),
                     "has_reply": out.get("reply") is not None, "needs_human": out["needs_human"],
                     "gold_cited": bool(gold) and gold in out.get("citations", []),
                     "any_cited": bool(out.get("citations")), "issues": len(out.get("grounding_issues", []))})
    df = pd.DataFrame(rows)
    summary = []
    for name, g in [("answerable", df[df.answerable]), ("unanswerable", df[~df.answerable])]:
        if len(g):
            summary.append({"subset": name, "n": len(g), "valid_reply_rate": round(g.has_reply.mean(), 3),
                            "gold_section_cited": round(g.gold_cited.mean(), 3),
                            "escalated_to_human": round(g.needs_human.mean(), 3),
                            "grounding_issues": int(g.issues.sum())})
    df.to_csv(U.ROOT / "eval" / "generation_details.csv", index=False)
    return pd.DataFrame(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM reply generator")
    args = ap.parse_args()

    U.set_seed(42)
    labels = U.load_labels()
    train, test = U.load_tickets("train"), load_test_sets()
    U.PRED_DIR.mkdir(parents=True, exist_ok=True)

    systems: dict = {"tfidf_lr": C.build("tfidf_lr", train), "e5_lr": C.build("e5_lr", train)}
    for name, path in find_finetuned().items():
        systems[name] = C.build("finetuned", model_dir=path)

    md = ["# Results (held-out test sets)\n",
          f"Abstention threshold tau = {args.tau}. Bootstrap 95% CIs (500 resamples). "
          "`en_test` is templated Bitext text (optimistic); `ar_test` and `hard_en` are hand-written.\n"]
    summary_rows, latency = [], {}
    for name, clf in systems.items():
        preds = predict_system(clf, test, labels)
        preds.to_csv(U.PRED_DIR / f"{name}.csv", index=False)
        latency[name] = classification_latency(clf, test.text.tolist())
        tab = slice_table(preds, ["set", "lang", "dialect"], args.tau)
        md += [f"## {name}\n", md_table(tab), ""]
        ar = preds[preds.lang == "ar"]
        summary_rows.append({
            "system": name,
            "macro_f1_en_test": tab.loc[tab.slice == "set=en_test", "macro_f1"].iloc[0],
            "macro_f1_hard_en": tab.loc[tab.slice == "set=hard_en", "macro_f1"].iloc[0],
            "macro_f1_ar": tab.loc[tab.slice == "set=ar_test", "macro_f1"].iloc[0],
            "ar_95ci": tab.loc[tab.slice == "set=ar_test", "macro_f1_95ci"].iloc[0],
            "ar_coverage@tau": round(float((ar.confidence >= args.tau).mean()), 3),
            "p50_ms": round(latency[name]["p50_ms"], 1), "p95_ms": round(latency[name]["p95_ms"], 1)})
        print(f"done: {name}")

    head = ["## Summary: same test sets, every system\n", md_table(pd.DataFrame(summary_rows)), "",
            "Latency = single-ticket classification on this machine (see device in README), not end-to-end.\n"]
    md = md[:2] + head + md[2:]

    md += ["## Retrieval ablation (help center, rag_eval.csv, answerable questions)\n",
           md_table(retrieval_ablation()), ""]

    if args.llm:
        assistant = TicketAssistant(use_llm=True, retrieval_mode="hybrid_rerank")
        md += ["## Grounded reply generation\n", md_table(generation_eval(assistant)), ""]

    (U.ROOT / "eval" / "results.md").write_text("\n".join(md), encoding="utf-8")
    (U.ROOT / "eval" / "results.json").write_text(json.dumps(
        {"tau": args.tau, "summary": summary_rows, "latency": latency}, indent=2), encoding="utf-8")
    print("\n".join(md))
    print("\nWrote eval/results.md, eval/results.json and eval/predictions/*.csv")


if __name__ == "__main__":
    main()
