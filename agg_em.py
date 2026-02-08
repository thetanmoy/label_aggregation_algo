import numpy as np
import pandas as pd

def em_aggregate(df, max_iter=50, tol=1e-6, seed=0):

    rng = np.random.default_rng(seed)

    tasks = df["orig_id"].unique()
    workers = df["!amt_worker_ids"].unique()

    t2i = {t:i for i,t in enumerate(tasks)}
    w2i = {w:i for i,w in enumerate(workers)}

    T = len(tasks)
    W = len(workers)

    t_idx = df["orig_id"].map(t2i).to_numpy()
    w_idx = df["!amt_worker_ids"].map(w2i).to_numpy()
    y = df["response"].astype(int).to_numpy()

    q = np.zeros(T, dtype=float)
    for t in range(T):
        yt = y[t_idx == t]
        if len(yt) == 0:
            q[t] = 0.5
        else:
            q[t] = np.mean(yt)  # fraction of 1s

    a = np.full(W, 0.7, dtype=float)
    b = np.full(W, 0.7, dtype=float)

    eps = 1e-12

    prev_ll = None

    for it in range(max_iter):

        num_a = np.zeros(W); den_a = np.zeros(W)
        num_b = np.zeros(W); den_b = np.zeros(W)

        for ti, wi, yi in zip(t_idx, w_idx, y):
            den_a[wi] += q[ti]
            num_a[wi] += q[ti] * (yi == 1)

            den_b[wi] += (1 - q[ti])
            num_b[wi] += (1 - q[ti]) * (yi == 0)

        a = (num_a + eps) / (den_a + 2*eps)
        b = (num_b + eps) / (den_b + 2*eps)

        a = np.clip(a, 1e-3, 1-1e-3)
        b = np.clip(b, 1e-3, 1-1e-3)

        q_new = np.zeros(T, dtype=float)

        pi = np.clip(np.mean(q), 1e-3, 1-1e-3)

        for t in range(T):
            idxs = np.where(t_idx == t)[0]
            if len(idxs) == 0:
                q_new[t] = 0.5
                continue

            logp1 = np.log(pi)
            logp0 = np.log(1 - pi)

            for j in idxs:
                wi = w_idx[j]
                yi = y[j]
                p_y_z1 = a[wi] if yi == 1 else (1 - a[wi])
                p_y_z0 = (1 - b[wi]) if yi == 1 else b[wi]

                logp1 += np.log(p_y_z1 + eps)
                logp0 += np.log(p_y_z0 + eps)


            m = max(logp1, logp0)
            p1 = np.exp(logp1 - m)
            p0 = np.exp(logp0 - m)
            q_new[t] = p1 / (p1 + p0 + eps)

        diff = np.max(np.abs(q_new - q))
        q = q_new

        if diff < tol:
            break

    pred = (q >= 0.5).astype(int)

    out = pd.DataFrame({"orig_id": tasks, "pred": pred})
    return out


if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")
    from sample_k import sample_k
    df_k3 = sample_k(df, k=3, seed=0)

    pred_df = em_aggregate(df_k3, max_iter=50, seed=0)
    print(pred_df.head())
    print("num tasks:", len(pred_df))

