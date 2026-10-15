from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import utils as U


class TfidfClassifier:
    name = "tfidf_lr"

    def __init__(self, train_df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        X = self.vec.fit_transform(train_df.text)
        self.clf = LogisticRegression(max_iter=3000, C=20.0).fit(X, train_df.intent)
        self.classes_ = self.clf.classes_

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(list(texts)))


class E5LRClassifier:
    name = "e5_lr"

    def __init__(self, train_df, C: float = 20.0):
        from sklearn.linear_model import LogisticRegression

        X = U.embed(train_df.text, prefix="query: ", cache_key="train")
        self.clf = LogisticRegression(max_iter=3000, C=C).fit(X, train_df.intent)
        self.classes_ = self.clf.classes_

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(U.embed(list(texts), prefix="query: "))


class FinetunedClassifier:
    name = "finetuned"

    def __init__(self, model_dir: str | Path, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dir = Path(model_dir)
        if not (self.dir / "config.json").exists():
            raise FileNotFoundError(f"No fine-tuned model in {self.dir}. Run `python -m src.train` first.")
        meta = json.loads((self.dir / "ticket_meta.json").read_text(encoding="utf-8"))
        self.prefix, self.max_len = meta["prefix"], meta["max_len"]
        self.temperature = float(meta.get("temperature", 1.0))
        self.classes_ = np.array(meta["classes"])
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).to(U.device()).eval()
        self._torch = torch

    def logits(self, texts) -> np.ndarray:
        torch, out = self._torch, []
        texts = [self.prefix + t for t in texts]
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)))

    def predict_proba(self, texts) -> np.ndarray:
        z = self.logits(list(texts)) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


def build(kind: str, train_df=None, model_dir: str | Path | None = None):
    if kind == "tfidf_lr":
        return TfidfClassifier(train_df)
    if kind == "e5_lr":
        return E5LRClassifier(train_df)
    if kind == "finetuned":
        return FinetunedClassifier(model_dir or U.ARTIFACTS / "encoder")
    raise ValueError(f"unknown classifier: {kind}")
