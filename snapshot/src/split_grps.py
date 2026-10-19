# split_groups.py
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

META = ["frame_idx","pid","video","label"]
csv_path = "../data/10_biu_features_enriched.csv"

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("csv_path", type=str)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(csv_path)
    for col in META:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    groups = (df["video"].astype(str) + "/" + df["pid"].astype(str)).values
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    tr_idx, te_idx = next(gss.split(df, groups=groups))

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    df_tr.to_csv("train_10_enriched.csv", index=False)
    df_te.to_csv("test_10_enriched.csv", index=False)

    # Quick sanity: class balance
    print(f"Train: {len(df_tr)} rows | pos%={df_tr['label'].mean()*100:.2f}")
    print(f"Test : {len(df_te)} rows | pos%={df_te['label'].mean()*100:.2f}")
    print("Wrote train.csv and test.csv")

if __name__ == "__main__":
    main()
