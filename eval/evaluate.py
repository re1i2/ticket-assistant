import argparse

import pandas as pd
from sklearn.metrics import classification_report

from src import utils as U
from src.metrics import md_table, slice_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--slice", action="append", default=[], help="column to slice by (repeatable)")
    ap.add_argument("--tau", type=float, default=None, help="abstention threshold for coverage metrics")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df["y_pred"] = df["y_pred"].fillna("(abstain)")
    print(md_table(slice_table(df, args.slice, args.tau)))
    print("\nMost frequent confusions:")
    print(md_table(U.top_confusions(df.y_true, df.y_pred, k=10)))
    print()
    print(classification_report(df.y_true, df.y_pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
