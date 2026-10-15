from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import utils as U


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def summarize(df: pd.DataFrame, tau: float | None = None, n_boot: int = 500) -> dict:
    f1, lo, hi = U.bootstrap_ci(df.y_true.astype(str), df.y_pred.astype(str), macro_f1, n=n_boot)
    row = {"n": len(df), "accuracy": round(accuracy_score(df.y_true, df.y_pred), 3),
           "macro_f1": round(f1, 3), "macro_f1_95ci": f"[{lo:.3f}, {hi:.3f}]"}
    if "category_true" in df:
        row["category_acc"] = round(float((df.category_true == df.category_pred).mean()), 3)
    if "priority_true" in df:
        row["priority_acc"] = round(float((df.priority_true == df.priority_pred).mean()), 3)
    if tau is not None and "confidence" in df:
        kept = df[df.confidence >= tau]
        row["coverage"] = round(len(kept) / len(df), 3)
        row["selective_acc"] = round(float((kept.y_true == kept.y_pred).mean()), 3) if len(kept) else float("nan")
    return row


def slice_table(df: pd.DataFrame, slices: list[str], tau: float | None = None) -> pd.DataFrame:
    rows = [{"slice": "all", **summarize(df, tau)}]
    for col in slices:
        for value, g in df.groupby(col):
            rows.append({"slice": f"{col}={value}", **summarize(g, tau)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
