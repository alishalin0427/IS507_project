"""
calibration_by_strength.py — 依球隊強弱（賠率熱門／冷門）檢查模型校準
用法（在專案資料夾執行）：
    python calibration_by_strength.py p_b1      # Baseline 1（預設）
    python calibration_by_strength.py p_m3      # 之後換成 Model 3，只改這個參數

輸入：data/processed/predictions.csv   （baseline.py 產生；要有 PRED 欄位）
      data/processed/match_odds.csv    （odd.py 產生；賽前 Bet365 賠率，每場一列）
輸出：終端機的 實際 vs 預期進球 + z 表、figures/calib_by_strength_<PRED>.png

熱門規則（要寫進 data_log）：射門球隊的隱含勝率 > 對手的隱含勝率 → favourite，否則 underdog。
平局機率不參與判斷；規則在看 validation 結果之前定好，不事後調整。
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)
PRED = sys.argv[1] if len(sys.argv) > 1 else "p_b1"   # ← 唯一要改的地方

# ---------- 1. 讀資料：只用 validation（test 封存） ----------
pred = pd.read_csv("data/processed/predictions.csv")
if PRED not in pred.columns:
    sys.exit(f"predictions.csv 沒有欄位 {PRED}；現有欄位：{list(pred.columns)}")
pred = pred[pred["split"] == "valid"]

odds = pd.read_csv("data/processed/match_odds.csv")
df = pred.merge(odds[["match_id", "home_team", "p_home", "p_away"]],
                on="match_id", how="left")
print(f"validation 射門：{len(pred)}；沒有賠率的射門（排除）：{df['p_home'].isna().sum()}")
df = df.dropna(subset=["p_home"]).copy()

# ---------- 2. 標記 favourite / underdog ----------
is_home = df["team"] == df["home_team"]
p_team = np.where(is_home, df["p_home"], df["p_away"])
p_opp = np.where(is_home, df["p_away"], df["p_home"])
df["group"] = np.where(p_team > p_opp, "favourite", "underdog")

# ---------- 3. 實際 vs 預期進球 + z ----------
p = df[PRED]
df["var"] = p * (1 - p)
g = df.groupby("group").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"),
                            xg=(PRED, "sum"), var=("var", "sum"))
g["actual_rate"] = g["goals"] / g["shots"]
g["predicted_rate"] = g["xg"] / g["shots"]
g["goals_minus_xg"] = g["goals"] - g["xg"]
g["z"] = g["goals_minus_xg"] / np.sqrt(g["var"])          # |z| > 2 → 不太可能只是隨機
g["verdict"] = np.where(g["z"] > 2, "UNDER-predicts",
               np.where(g["z"] < -2, "OVER-predicts", "ok"))
print(f"\nCalibration by team strength ({PRED}, validation)")
print("goals_minus_xg > 0: model under-predicts; < 0: model over-predicts")
print(g.drop(columns="var").round(3).to_string())

# ---------- 4. 每組一條 calibration curve（5 組） ----------
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
lim = 0.0
for name, color in [("favourite", "#2a78d6"), ("underdog", "#eb6834")]:
    d = df[df["group"] == name]
    d = d.assign(bin=pd.qcut(d[PRED], 5, duplicates="drop"))
    c = d.groupby("bin", observed=True).agg(pr=(PRED, "mean"), ac=("is_goal", "mean"),
                                            n=("is_goal", "size"))
    se = np.sqrt(c["ac"] * (1 - c["ac"]) / c["n"])
    ax.errorbar(c["pr"], c["ac"], yerr=1.96 * se, fmt="o-", capsize=2, color=color,
                label=f"{name} (n={len(d)})")
    lim = max(lim, c["pr"].max(), c["ac"].max())
lim *= 1.1
ax.plot([0, lim], [0, lim], "--", color="gray", label="Perfect calibration")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title(f"{PRED} calibration: favourite vs underdog (validation)", loc="left", fontsize=9)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"figures/calib_by_strength_{PRED}.png"); plt.close(fig)
print(f"\nSaved figures/calib_by_strength_{PRED}.png")