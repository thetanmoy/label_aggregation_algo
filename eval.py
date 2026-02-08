import pandas as pd

def compute_error(pred_df, gold_df):
    merged = pred_df.merge(gold_df, on="orig_id", how="inner")
    wrong = (merged["pred"] != merged["gold"]).sum()
    total = len(merged)
    return wrong / total

if __name__ == "__main__":
    df = pd.read_csv("rte.standardized.tsv", sep="\t")

    gold_df = df.groupby("orig_id")["gold"].first().reset_index()

    pred_df = gold_df.copy()
    pred_df["pred"] = pred_df["gold"] 
    err = compute_error(pred_df[["orig_id", "pred"]], gold_df)
    print("error (should be 0.0):", err)

