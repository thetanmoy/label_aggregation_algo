import pandas as pd
import matplotlib.pyplot as plt

maj = pd.read_csv("results_majority.csv")
em  = pd.read_csv("results_em.csv")

plt.plot(maj["k"], maj["avg_error"], marker="o", label="Majority")
plt.plot(em["k"],  em["avg_error"],  marker="o", label="EM")

plt.xlabel("k (labels per task)")
plt.ylabel("Average error")
plt.title("Label Aggregation: Majority vs EM")
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot_majority_vs_em.png", dpi=200)
print("Saved: plot_majority_vs_em.png")
