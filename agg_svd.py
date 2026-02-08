import numpy as np
import pandas as pd

def svd_aggregate(df: pd.DataFrame, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tasks = np.sort(df["orig_id"].unique())
    workers = np.sort(df["!amt_worker_ids"].unique())

    t2i = {t: i for i, t in enumerate(tasks)}
    w2i = {w: i for i, w in enumerate(workers)}

    T = len(tasks)
    W = len(workers)

    L = np.zeros((T, W), dtype=float)

    for _, row in df.iterrows():
        ti = t2i[row["orig_id"]]
        wi = w2i[row["!amt_worker_ids"]]
        y = 1.0 if int(row["response"]) == 1 else -1.0
        L[ti, wi] = y

    # SVD
    # L = U S V^T
    U, S, Vt = np.linalg.svd(L, full_matrices=False)
    u1 = U[:, 0]  # top left singular vector

    # sign -> prediction
    # if u1 == 0, random tie-break
    preds = []
    for i, t in enumerate(tasks):
        if u1[i] > 0:
            pred = 1
        elif u1[i] < 0:
            pred = 0
        else:
            pred = int(rng.integers(0, 2))
        preds.append((t, pred))

    return pd.DataFrame(preds, columns=["orig_id", "pred"])


if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")
    out = svd_aggregate(df, seed=0)
    print(out.head())
    print("num tasks:", len(out))
