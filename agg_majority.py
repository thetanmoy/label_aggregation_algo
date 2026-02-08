import numpy as np
import pandas as pd

def majority_vote(df, seed=None):
    rng = np.random.default_rng(seed)
    preds = []

    for tid, g in df.groupby("orig_id"):
        votes = g["response"].to_numpy()
        c1 = np.sum(votes == 1)
        c0 = np.sum(votes == 0)

        if c1 > c0:
            pred = 1
        elif c0 > c1:
            pred = 0
        else:
            pred = int(rng.integers(0, 2))  # random 0 or 1

        preds.append((tid, pred))

    return pd.DataFrame(preds, columns=["orig_id", "pred"])


if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")
    out = majority_vote(df[df["orig_id"].isin([25, 35])], seed=0)
    print(out)

