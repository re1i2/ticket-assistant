import argparse

import pandas as pd

from src import utils as U
from src.metrics import md_table

CONFUSABLE = [
    {"check_invoice", "get_invoice"}, {"get_refund", "track_refund"}, {"get_refund", "check_refund_policy"},
    {"track_refund", "check_refund_policy"}, {"cancel_order", "check_cancellation_fee"},
    {"contact_customer_service", "contact_human_agent"}, {"create_account", "registration_problems"},
    {"edit_account", "switch_account"}, {"change_order", "cancel_order"}, {"delivery_options", "delivery_period"},
    {"set_up_shipping_address", "change_shipping_address"}, {"place_order", "change_order"},
    {"payment_issue", "check_payment_methods"}, {"complaint", "review"}, {"track_order", "delivery_period"},
]


def auto_cause(row, tau: float) -> str:
    pair = {row.y_true, row.y_pred}
    if row.y_true == row.y_pred:
        return "abstained_but_correct (threshold too strict)"
    if any(pair == c for c in CONFUSABLE):
        return "ambiguous / overlapping intents"
    if row.lang == "ar":
        return ("missing coverage: Arabic dialect" if row.dialect == "gulf"
                else "missing coverage: Arabic (cross-lingual gap)")
    if len(str(row.text).split()) <= 4:
        return "under-specified (very short)"
    if row.confidence < tau:
        return "low-confidence model error"
    return "confident model error (review: label noise?)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=40, help="max rows to export for manual review")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    wrong = df[df.y_true != df.y_pred]
    abstained_ok = df[(df.y_true == df.y_pred) & (df.confidence < args.tau)]
    cand = pd.concat([wrong, abstained_ok])
    cand = cand.assign(auto_cause=[auto_cause(r, args.tau) for r in cand.itertuples()])

    per_set = max(args.n // max(cand["set"].nunique(), 1), 1)
    sample = pd.concat([g.sample(min(len(g), per_set), random_state=0) for _, g in cand.groupby("set")])
    sample = sample.assign(manual_cause="")
    out_csv = U.ROOT / "eval" / "error_analysis.csv"
    sample[["ticket_id", "set", "lang", "dialect", "text", "y_true", "y_pred", "confidence", "auto_cause",
            "manual_cause"]].to_csv(out_csv, index=False)

    counts = cand.groupby("auto_cause").size().rename("count").reset_index().sort_values("count", ascending=False)
    ex = cand.groupby("auto_cause").first().reset_index()[["auto_cause", "text", "y_true", "y_pred"]]
    ex.columns = ["auto_cause", "example", "true", "predicted"]
    table = counts.merge(ex, on="auto_cause")
    per_set_err = (df.assign(err=(df.y_true != df.y_pred)).groupby("set").err.agg(["size", "mean"]).round(3)
                   .reset_index().rename(columns={"size": "n", "mean": "error_rate"}))
    md = [f"# Error analysis: {args.predictions}\n",
          f"{len(wrong)} misclassified + {len(abstained_ok)} correct-but-below-tau out of {len(df)} tickets.\n",
          "Causes below are heuristic. Review `eval/error_analysis.csv` and fill `manual_cause` before reporting.\n",
          md_table(table), "", "## Per test set error rate\n", md_table(per_set_err), ""]
    (U.ROOT / "eval" / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out_csv} ({len(sample)} rows to categorise by hand)")


if __name__ == "__main__":
    main()
