from __future__ import annotations

import re

import numpy as np

from . import utils as U

_TOKEN = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(U.MODELS["reranker"], device=U.device())
    return _RERANKER


class Retriever:
    def __init__(self, texts: list[str], cache_key: str | None = None, rrf_k: int = 60, pool: int = 20):
        from rank_bm25 import BM25Okapi

        self.texts = texts
        self.rrf_k, self.pool = rrf_k, pool
        self.M = U.embed(texts, prefix="passage: ", cache_key=cache_key)
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        self._faiss = None
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.M.shape[1])
            self._faiss.add(self.M)
        except ImportError:
            pass

    def _dense(self, query: str, n: int) -> list[int]:
        q = U.embed([query], prefix="query: ")
        n = min(n, len(self.texts))
        if self._faiss is not None:
            return [int(i) for i in self._faiss.search(q, n)[1][0]]
        return [int(i) for i in np.argsort(-(self.M @ q[0]))[:n]]

    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query) or ["_"])
        return [int(i) for i in np.argsort(-scores)[:n]]

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int) -> list[int]:
        score: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
        return sorted(score, key=score.get, reverse=True)

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[int]:
        if mode == "dense":
            return self._dense(query, k)
        if mode == "bm25":
            return self._bm25(query, k)
        fused = self._rrf([self._dense(query, self.pool), self._bm25(query, self.pool)], self.rrf_k)
        if mode == "hybrid":
            return fused[:k]
        if mode == "hybrid_rerank":
            cand = fused[: self.pool]
            if U.SMOKE:
                return cand[:k]
            scores = get_reranker().predict([(query, self.texts[i]) for i in cand])
            return [cand[j] for j in np.argsort(-np.asarray(scores))[:k]]
        raise ValueError(f"unknown mode: {mode}")


def kb_index_text(section: dict) -> str:
    return f"{section['title']} > {section['section']}\n{section['text']}"
