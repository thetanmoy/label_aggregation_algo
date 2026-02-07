import pandas as pd

def load_rte(path="rte.standardized.tsv"):
    df = pd.read_csv(path, sep="\t")
    return df

if __name__ == "__main__":
    df = load_rte()
    print(df.head())
    print(df.columns)
    print("Total rows:", len(df))
