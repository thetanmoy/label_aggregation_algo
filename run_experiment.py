import pandas as pd
import numpy as np

from sample_k import sample_k
from agg_majority import majority_vote
from agg_em import em_aggregate
from agg_svd import svd_aggregate
from eval import compute_error


def run_majority_experiment(df, repeats=30, seed=0):
    rng = np.random.default_rng(seed)
    gold_df = df.groupby("orig_id")["gold"].first().reset_index()

    results = []
    for k in range(1, 11):
        errs = []
        for _ in range(repeats):
            s = int(rng.integers(0, 10**9))
            df_k = sample_k(df, k=k, seed=s)
            pred_df = majority_vote(df_k, seed=s)
            errs.append(compute_error(pred_df, gold_df))

        avg_err = float(np.mean(errs))
        results.append([k, avg_err])
        print(f"[MAJ] k={k} avg_error={avg_err:.4f}")

    return pd.DataFrame(results, columns=["k", "avg_error"])


def run_em_experiment(df, repeats=30, seed=0):
    rng = np.random.default_rng(seed)
    gold_df = df.groupby("orig_id")["gold"].first().reset_index()

    results = []
    for k in range(1, 11):
        errs = []
        for _ in range(repeats):
            s = int(rng.integers(0, 10**9))
            df_k = sample_k(df, k=k, seed=s)
            pred_df = em_aggregate(df_k, seed=s)
            errs.append(compute_error(pred_df, gold_df))

        avg_err = float(np.mean(errs))
        results.append([k, avg_err])
        print(f"[EM]  k={k} avg_error={avg_err:.4f}")

    return pd.DataFrame(results, columns=["k", "avg_error"])


def run_svd_experiment(df, repeats=30, seed=0):
    rng = np.random.default_rng(seed)
    gold_df = df.groupby("orig_id")["gold"].first().reset_index()

    results = []
    for k in range(1, 11):
        errs = []
        for _ in range(repeats):
            s = int(rng.integers(0, 10**9))
            df_k = sample_k(df, k=k, seed=s)
            pred_df = svd_aggregate(df_k, seed=s)
            errs.append(compute_error(pred_df, gold_df))

        avg_err = float(np.mean(errs))
        results.append([k, avg_err])
        print(f"[SVD] k={k} avg_error={avg_err:.4f}")

    return pd.DataFrame(results, columns=["k", "avg_error"])


if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")

    out_majority = run_majority_experiment(df, repeats=30, seed=0)
    out_majority.to_csv("results_majority.csv", index=False)

    out_em = run_em_experiment(df, repeats=30, seed=0)
    out_em.to_csv("results_em.csv", index=False)

    out_svd = run_svd_experiment(df, repeats=30, seed=0)
    out_svd.to_csv("results_svd.csv", index=False)

    print("\nSaved: results_majority.csv, results_em.csv, results_svd.csv")
