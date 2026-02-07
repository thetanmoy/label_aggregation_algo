import numpy as np
import pandas as pd

def sample_k(df, k, seed=None):
    """
    For each task (orig_id), randomly sample k annotations (rows).
    """
    rng = np.random.default_rng(seed)

    sampled_groups = []
    for tid, g in df.groupby("orig_id"):
        # g has 10 rows normally
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        sampled_groups.append(df.loc[idx])

    return pd.concat(sampled_groups).reset_index(drop=True)

if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")
    df_k3 = sample_k(df, k=3, seed=0)
    print("rows k=3:", len(df_k3))   # expected 800*3 = 2400
    print(df_k3.head())
