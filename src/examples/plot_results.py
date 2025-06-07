import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

SRC = Path("selective_results_rf.csv")
assert SRC.exists(), "selective_results_rf.csv  が見つかりません"

df = pd.read_csv(SRC)

df["rmse"] = np.sqrt(df["mse"]).round(2) 
df["mse"] = df["mse"].round(2)

# method × α で要約
summary = (
    df.groupby(["method", "fgw_alpha"])
      .agg(mean_mse   = ("mse", "mean"),
           std_mse    = ("mse", "std"),
           # q25_mse    = ("mse", lambda x: x.quantile(.25)),
           # median_mse = ("mse", "median"),
           # q75_mse    = ("mse", lambda x: x.quantile(.75)),
           min_mse    = ("mse", "min"),
           max_mse    = ("mse", "max"),
           # n_targets  = ("mse", "size"), 
           mean_rmse = ("rmse", "mean"),
           std_rmse    = ("rmse", "std"),
           min_rmse    = ("rmse", "min"),
           max_rmse    = ("rmse", "max"),
          )
      .reset_index()
      .sort_values(["fgw_alpha", "method"])
)

# summary.to_csv("summary_df.csv", index=False)
# print("✅ summary_df.csv を出力しました")


fig, ax = plt.subplots(figsize=(6, 4))

alphas   = sorted(summary["fgw_alpha"].unique()) # [0, 50, 100]
methods  = ["bottomk", "random", "topk"]
bar_w    = 0.25
x_idx    = np.arange(len(alphas))# [0, 1, 2]

for i, method in enumerate(methods):
    sub = summary[summary["method"] == method].sort_values("fgw_alpha")
    ax.bar(x_idx + i*bar_w,
           sub["mean_rmse"],
           width=bar_w,
           yerr=sub["std_rmse"],
           capsize=4,
           label=method)

ax.set_xlabel("FGW α")
ax.set_ylabel("RMSE")
ax.set_xticks(x_idx + bar_w)
ax.set_xticklabels(alphas)
ax.set_title("RMSE by Method and FGW α (mean ± SD)")
ax.legend()
plt.tight_layout()
plt.show()
# fig.savefig("rmse_barplot.png", dpi=300)   # ← PNG 保存したい場合はコメント解除