import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_svd_extrap.csv")

plt.figure()
plt.plot(df["k"], df["avg_error"], marker="o")
plt.xlabel("k (labels per task)")
plt.ylabel("Average error")
plt.title("SVD Extrapolation Performance (k>10)")
plt.grid(True)
plt.savefig("plot_svd_extrap.png", dpi=200)
print("Saved: plot_svd_extrap.png")