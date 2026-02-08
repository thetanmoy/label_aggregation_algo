import pandas as pd
import matplotlib.pyplot as plt

maj = pd.read_csv("results_majority.csv")
em  = pd.read_csv("results_em.csv")
svd = pd.read_csv("results_svd.csv")

plt.figure()
plt.plot(maj["k"], maj["avg_error"], marker="o", label="Majority")
plt.plot(em["k"], em["avg_error"], marker="o", label="EM")
plt.plot(svd["k"], svd["avg_error"], marker="o", label="SVD")

plt.xlabel("k (labels per task)")
plt.ylabel("Average error")
plt.title("Label Aggregation: Majority vs EM vs SVD")
plt.legend()
plt.grid(True)
plt.savefig("plot_all_three.png", dpi=200)
print("Saved: plot_all_three.png")