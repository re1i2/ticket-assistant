from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("TA_DATA_DIR", ROOT / "data"))
ARTIFACTS = Path(os.getenv("TA_ARTIFACTS_DIR", ROOT / "artifacts"))
CACHE = ROOT / "cache"
PRED_DIR = ROOT / "eval" / "predictions"
SMOKE = os.getenv("TA_SMOKE", "0") == "1"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELS = {
    "embed": "intfloat/multilingual-e5-small",
    "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "llm_gpu": "Qwen/Qwen2.5-1.5B-Instruct",
    "llm_cpu": "Qwen/Qwen2.5-0.5B-Instruct",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str | Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare_data` first.")
    return path


def load_tickets(split: str) -> pd.DataFrame:
    return pd.read_csv(_require(DATA / f"tickets_{split}.csv"))


def load_labels() -> dict:
    return json.loads(_require(DATA / "labels.json").read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_kb() -> list[dict]:
    sections = []
    for path in sorted((DATA / "kb").glob("*.md")):
        doc = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip()
        current, buf = None, []
        for line in lines[1:] + ["## __end__"]:
            if line.startswith("## "):
                if current and "\n".join(buf).strip():
                    sections.append({"id": f"{doc}#{slugify(current)}", "doc": doc, "title": title,
                                     "section": current, "text": "\n".join(buf).strip()})
                current, buf = line[3:].strip(), []
            elif current:
                buf.append(line)
    return sections


_EMBEDDERS: dict = {}


def get_embedder(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    name = name or MODELS["embed"]
    if name not in _EMBEDDERS:
        _EMBEDDERS[name] = SentenceTransformer(name, device=device())
    return _EMBEDDERS[name]


def _hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm="l2")
    return hv.transform(texts).toarray().astype("float32")


def embed(texts: Iterable[str], prefix: str = "query: ", model: str | None = None,
          batch_size: int = 64, cache_key: str | None = None) -> np.ndarray:
    texts = list(texts)
    if SMOKE:
        return _hash_embed([prefix + t for t in texts])
    model = model or MODELS["embed"]
    f = None
    if cache_key:
        CACHE.mkdir(exist_ok=True)
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:10]
        f = CACHE / f"{slugify(model)}__{cache_key}__{slugify(prefix) or 'noprefix'}__{len(texts)}_{digest}.npy"
        if f.exists():
            return np.load(f)
    vecs = get_embedder(model).encode(
        [prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=len(texts) > 500, convert_to_numpy=True,
    ).astype("float32")
    if f is not None:
        np.save(f, vecs)
    return vecs


@dataclass
class LLM:

    model: str | None = None
    backend: str | None = None
    last_usage: dict = field(default_factory=dict)

    def __post_init__(self):
        self.backend = self.backend or ("openai" if os.getenv("LLM_BASE_URL") else "hf")
        if self.backend == "openai":
            from openai import OpenAI

            self.model = self.model or os.environ["LLM_MODEL"]
            self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.getenv("LLM_API_KEY", "none"))
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dev = device()
            self.model = self.model or (MODELS["llm_gpu"] if dev == "cuda" else MODELS["llm_cpu"])
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype).to(dev)
            self.hf_model.eval()
        print(f"LLM ready: {self.model} via {self.backend}")

    def chat(self, messages: list[dict], temperature: float = 0.0, max_new_tokens: int = 300) -> str:
        t0 = time.perf_counter()
        if self.backend == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_new_tokens)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            self.last_usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                               "completion_tokens": getattr(u, "completion_tokens", None)}
        else:
            import torch

            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(self.hf_model.device)
            gen = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            if temperature and temperature > 0:
                gen.update(do_sample=True, temperature=temperature)
            else:
                gen.update(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
            with torch.no_grad():
                out = self.hf_model.generate(**enc, **gen)
            new = out[0][enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new, skip_special_tokens=True)
            self.last_usage = {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": int(len(new))}
        self.last_usage["latency_s"] = time.perf_counter() - t0
        return text.strip()


_LLM: LLM | None = None


def get_llm(**kwargs) -> LLM:
    global _LLM
    if _LLM is None or kwargs:
        _LLM = LLM(**kwargs)
    return _LLM


def bootstrap_ci(y_true, y_pred, metric, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(lo), float(hi)


def latency_stats(times_ms: list[float]) -> dict:
    t = np.array(times_ms)
    return {"n": len(t), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    df = df[df.true != df.pred]
    return df.value_counts().head(k).rename("count").reset_index()
