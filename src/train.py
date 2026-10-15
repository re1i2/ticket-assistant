from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils as U


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.3, 6.0, 115):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = U.load_config(args.config)

    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    U.set_seed(cfg["seed"])
    labels = U.load_labels()["intents"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    prefix, max_len = cfg.get("prefix", ""), cfg.get("max_len", 64)

    train, val = U.load_tickets("train"), U.load_tickets("val")
    if cfg.get("extra_train_csv"):
        extra = pd.read_csv(cfg["extra_train_csv"])
        banned = set(U.load_tickets("ar_test").text) | set(U.load_tickets("hard_test").text)
        assert not (set(extra.text) & banned), "extra_train_csv overlaps with a test set"
        train = pd.concat([train, extra[["text", "intent"]]], ignore_index=True)
        print(f"Added {len(extra)} extra training tickets")

    rev = cfg.get("revision")
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=rev, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df):
            enc = tok([prefix + t for t in df.text], truncation=True, max_length=max_len)
            self.items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                           "labels": label2id[y]} for i, y in enumerate(df.intent)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro", zero_division=0),
                "accuracy": accuracy_score(p.label_ids, pred)}

    out_dir = Path(cfg.get("output_dir", U.ARTIFACTS / "encoder"))
    if not out_dir.is_absolute():
        out_dir = U.ROOT / out_dir
    ckpt = out_dir.parent / (out_dir.name + "_ckpt")
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    ratio = cfg.get("warmup_ratio", 0.1)
    warmup = {"warmup_ratio": ratio} if "warmup_ratio" in fields else {"warmup_steps": ratio}
    targs = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=128,
        weight_decay=cfg.get("weight_decay", 0.01), **warmup,
        label_smoothing_factor=cfg.get("label_smoothing", 0.0),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        seed=cfg["seed"], fp16=torch.cuda.is_available(), report_to="none", logging_steps=50,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=DS(train), eval_dataset=DS(val), processing_class=tok,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("patience", 2))],
    )
    trainer.train()

    pred = trainer.predict(DS(val))
    y = pred.label_ids
    temperature = fit_temperature(pred.predictions, y)
    val_f1 = float(f1_score(y, pred.predictions.argmax(-1), average="macro"))
    print(f"validation macro-F1 = {val_f1:.4f} | fitted temperature = {temperature:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(out_dir)
    meta = {"classes": labels, "prefix": prefix, "max_len": max_len, "temperature": temperature,
            "base_model": cfg["model"], "revision": rev, "seed": cfg["seed"], "val_macro_f1": val_f1,
            "config": cfg, "log_history": trainer.state.log_history}
    (out_dir / "ticket_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
