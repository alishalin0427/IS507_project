# statsbomb_benchmark.py
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pred = pd.read_csv("data/processed/predictions.csv")
pred = pred[pred["split"] == "valid"]

sb = pd.concat(pd.read_parquet(f, columns=["id", "shot_statsbomb_xg"])
               for f in glob.glob("data/processed/shots_*_1516.parquet"))
df = pred.merge(sb.rename(columns={"id": "shot_id", "shot_statsbomb_xg": "p_sb"}),
                on="shot_id", how="left")
print("缺 StatsBomb xG 的射門：", df["p_sb"].isna().sum())
df = df.dropna(subset=["p_sb"])
df["p_sb"] = df["p_sb"].clip(1e-6, 1 - 1e-6)

y = df["is_goal"]
for name, col in [("Baseline 0", "p_b0"), ("Baseline 1", "p_b1"), ("StatsBomb xG", "p_sb")]:
    print(f"{name:13s} log_loss {log_loss(y, df[col]):.4f}  brier {brier_score_loss(y, df[col]):.4f}")

fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
for name, col in [("Baseline 1", "p_b1"), ("StatsBomb xG", "p_sb")]:
    b = df.assign(bin=pd.qcut(df[col], 10, duplicates="drop")).groupby("bin", observed=True) \
          .agg(pr=(col, "mean"), ac=("is_goal", "mean"))
    ax.plot(b["pr"], b["ac"], "o-", label=name)
lim = 0.5
ax.plot([0, lim], [0, lim], "--", color="gray", label="Perfect")
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title("StatsBomb xG vs Baseline 1 (validation)", loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/statsbomb_benchmark_calibration.png")