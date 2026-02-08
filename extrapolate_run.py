import pandas as pd
import numpy as np

from sample_k import sample_k
from agg_svd import svd_aggregate
from eval import compute_error

def estimate_worker_skill(df):
    worker_acc = df.groupby("!amt_worker_ids").apply(
        lambda g: (g["response"] == g["gold"]).mean()
    )
    return worker_acc.to_dict()

def extrapolate_labels(df, k_target, seed=0):

    rng = np.random.default_rng(seed)

    gold_per_task = df.groupby("orig_id")["gold"].first().to_dict()

    skill = estimate_worker_skill(df)

    orig_tasks = df["orig_id"].unique()
    workers = df["!amt_worker_ids"].unique()

    new_rows = [df.copy()]

    extra = k_target - 10
    if extra <= 0:
        return df.copy()

    synthetic = []
    for tid in orig_tasks:
        gold = gold_per_task[tid]
        for _ in range(extra):
            w = rng.choice(workers)
            p = skill.get(w, 0.5)

            y = gold if rng.random() < p else 1 - gold

            synthetic.append({
                "!amt_annotation_ids": f"syn_{tid}_{rng.integers(1e12)}",
                "!amt_worker_ids": w,
                "orig_id": tid,
                "response": int(y),
                "gold": int(gold),
            })

    syn_df = pd.DataFrame(synthetic)
    return pd.concat([df, syn_df], ignore_index=True)

def run_svd_extrapolation(df, k_values, repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    gold_df = df.groupby("orig_id")["gold"].first().reset_index()

    results = []
    for k in k_values:
        errs = []
        for r in range(repeats):
            s = int(rng.integers(0, 10**9))
            df_k = extrapolate_labels(df, k_target=k, seed=s)
            pred_df = svd_aggregate(df_k, seed=s)
            err = compute_error(pred_df, gold_df)
            errs.append(err)

        avg_err = float(np.mean(errs))
        results.append([k, avg_err])
        print(f"[EXTRAP SVD] k={k} avg_error={avg_err:.4f}")

    return pd.DataFrame(results, columns=["k", "avg_error"])


if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")

    k_values = [10, 12, 15, 20, 30, 40, 50]

    out = run_svd_extrapolation(df, k_values=k_values, repeats=10, seed=0)
    out.to_csv("results_svd_extrap.csv", index=False)

    print("\nSaved: results_svd_extrap.csv")