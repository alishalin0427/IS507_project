"""
models.py — 在 Baseline 1 上逐步加入資訊，修正錯誤分析找到的三個盲點
用法（在 IS507_project/ 資料夾執行，需先跑過 baseline.py 產生 split.csv）：
    python models.py > models_output.txt 2>&1

模型（每一步都包含上一步的所有特徵）：
  Baseline 1：距離、角度
  Model 2   ：+ 身體部位          → 修正「高估頭球」
  Model 3   ：+ 比賽情境          → 修正「低估反擊、高估角球」
  Model 4   ：+ log(距離)         → 修正「距離效果不是直線」

輸出：
  - validation 上的比較表（也存成 data/processed/model_results.csv）
  - Model 4 的係數（勝算比）
  - Baseline 1 vs Model 4 的分組錯誤分析（訓練資料）
  - figures/model4_calibration.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

LEAGUES = ["pl", "laliga", "ligue1", "seriea"]

# ---------- 1. 讀資料、清理（和 baseline.py 完全相同） ----------
shots = pd.concat(
    [pd.read_parquet(f"data/processed/shots_{c}_1516.parquet") for c in LEAGUES],
    ignore_index=True,
)
shots = shots[shots["shot_type"] != "Penalty"].copy()
shots["x"] = shots["x"].clip(0, 120)
shots["y"] = shots["y"].clip(0, 80)
shots["distance"] = np.hypot(120 - shots["x"], 40 - shots["y"])
shots["angle"] = np.abs(np.arctan2(44 - shots["y"], 120 - shots["x"])
                        - np.arctan2(36 - shots["y"], 120 - shots["x"]))
shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

# ---------- 2. 新特徵 ----------
# 身體部位：以「腳」為基準，另外標記頭球和其他部位
shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
shots["is_other_body"] = (shots["shot_body_part"] == "Other").astype(int)

# 比賽情境：以「一般進攻 Regular Play」為基準；少見的類別併成 other_pattern
pattern_dummies = {
    "from_counter": "From Counter",
    "from_corner": "From Corner",
    "from_free_kick": "From Free Kick",
    "from_throw_in": "From Throw In",
}
for col, value in pattern_dummies.items():
    shots[col] = (shots["play_pattern"] == value).astype(int)
shots["other_pattern"] = (~shots["play_pattern"].isin(
    list(pattern_dummies.values()) + ["Regular Play"])).astype(int)

# log(距離)：讓距離的效果可以「先陡後緩」；clip 避免 log(0)
shots["log_distance"] = np.log(shots["distance"].clip(lower=0.5))

# ---------- 3. 使用和 baseline.py 相同的切分 ----------
split = pd.read_csv("data/processed/split.csv")
shots = shots.merge(split, on="match_id", how="left")
assert shots["split"].notna().all(), "Some matches are missing from split.csv: rerun baseline.py"
train = shots[shots["split"] == "train"].copy()
valid = shots[shots["split"] == "valid"].copy()
# test set 封存到期末

# ---------- 4. 逐步定義模型 ----------
body = ["is_header", "is_other_body"]
pattern = list(pattern_dummies) + ["other_pattern"]
MODELS = {
    "Baseline 1: distance + angle": ["distance", "angle"],
    "Model 2: + body part":         ["distance", "angle"] + body,
    "Model 3: + play pattern":      ["distance", "angle"] + body + pattern,
    "Model 4: + log(distance)":     ["distance", "angle", "log_distance"] + body + pattern,
}

# ---------- 5. 訓練並在 validation 上評分 ----------
y = valid["is_goal"]
base_rate = train["is_goal"].mean()
rows = [{"model": "Baseline 0: average rate",
         "log_loss": log_loss(y, np.full(len(valid), base_rate)),
         "brier": brier_score_loss(y, np.full(len(valid), base_rate))}]
fitted = {}
for name, feats in MODELS.items():
    m = LogisticRegression(max_iter=2000)
    m.fit(train[feats], train["is_goal"])
    p = m.predict_proba(valid[feats])[:, 1]
    fitted[name] = (m, feats)
    valid[name] = p
    rows.append({"model": name, "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)})

# 外部參考：StatsBomb 官方 xG（只拿來比較，從不當特徵）
# 它用了更多資訊（例如 freeze frame），也是用其他資料訓練的，所以不是公平的對手，而是「專業水準」的參考線
sb_xg = valid["shot_statsbomb_xg"].clip(1e-6, 1 - 1e-6)   # 避免機率剛好是 0 或 1 時 log loss 無法計算
valid["StatsBomb xG"] = sb_xg
rows.append({"model": "Reference: StatsBomb xG",
             "log_loss": log_loss(y, sb_xg), "brier": brier_score_loss(y, sb_xg)})

results = pd.DataFrame(rows)
b1 = results.loc[results["model"].str.startswith("Baseline 1"), "log_loss"].iloc[0]
results["vs_baseline1_%"] = (100 * (results["log_loss"] - b1) / b1).round(1)
results["vs_previous_%"] = (100 * results["log_loss"].pct_change()).round(1)
results.loc[results["model"].str.startswith("Reference"), "vs_previous_%"] = np.nan  # 參考線不是「下一步」
results["mean_prediction"] = [np.nan] + [valid[n].mean() for n in MODELS] + [sb_xg.mean()]
results.loc[0, "mean_prediction"] = base_rate
results["actual_goal_rate"] = y.mean()
results = results.round({"log_loss": 4, "brier": 4, "mean_prediction": 4, "actual_goal_rate": 4})
print("Validation results (lower is better):")
print(results.to_string(index=False))

# StatsBomb xG 和 Model 3 差多少？這段差距就是位置、身體部位、情境以外的資訊（例如防守站位）
m3 = results.loc[results["model"].str.startswith("Model 3"), "log_loss"].iloc[0]
sb = results.loc[results["model"].str.startswith("Reference"), "log_loss"].iloc[0]
gap_total = b1 - sb
print(f"\nGap from Baseline 1 to StatsBomb xG: {gap_total:.4f} log loss")
print(f"Share of that gap closed by Model 3: {100 * (b1 - m3) / gap_total:.0f}%")
results.to_csv("data/processed/model_results.csv", index=False)

# ---------- 6. Model 4 的係數，換成勝算比（odds ratio） ----------
final_name = "Model 4: + log(distance)"
m4, f4 = fitted[final_name]
coef = pd.DataFrame({"feature": f4, "coef": m4.coef_[0]})
coef["odds_ratio"] = np.exp(coef["coef"])
print(f"\n{final_name} coefficients:")
print(coef.round(3).to_string(index=False))
print("odds_ratio < 1：降低進球勝算；> 1：提高。例如 0.5 = 勝算減半（其他條件相同時）")

# ---------- 7. 分組錯誤分析：Baseline 1 vs Model 4（訓練資料） ----------
diag = train.copy()
for name in ["Baseline 1: distance + angle", final_name]:
    m, feats = fitted[name]
    diag[name] = m.predict_proba(diag[feats])[:, 1]
diag["distance_band"] = pd.cut(diag["distance"], [0, 6, 12, 18, 25, 200],
                               labels=["0-6", "6-12", "12-18", "18-25", "25+"])
common = ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In"]
diag["play_pattern_grp"] = diag["play_pattern"].where(diag["play_pattern"].isin(common), "Other")

def compare_errors(col):
    """每組：實際進球 vs 兩個模型的預期進球；z 超過 ±2 代表偏差不太可能只是隨機"""
    out = diag.groupby(col, observed=True).agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
    for name, short in [("Baseline 1: distance + angle", "B1"), (final_name, "M4")]:
        g = diag.groupby(col, observed=True)[name]
        xg = g.sum()
        var = g.apply(lambda p: (p * (1 - p)).sum())
        out[f"{short}_goals_minus_xg"] = (out["goals"] - xg).round(1)
        out[f"{short}_z"] = ((out["goals"] - xg) / np.sqrt(var)).round(1)
    return out

print("\n" + "=" * 70)
print("Error analysis: Baseline 1 (B1) vs Model 4 (M4), training set")
print("goals_minus_xg > 0: under-predicts; < 0: over-predicts; |z| > 2: unlikely to be chance")
print("=" * 70)
for col in ["shot_body_part", "play_pattern_grp", "league", "distance_band"]:
    print(f"\nBy {col}:")
    print(compare_errors(col).to_string())

# ---------- 8. Calibration：Baseline 1 vs Model 4（validation） ----------
def calib_table(p):
    d = pd.DataFrame({"p": p, "y": y.values})
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"))

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
c1 = calib_table(valid["Baseline 1: distance + angle"])
c4 = calib_table(valid[final_name])
csb = calib_table(valid["StatsBomb xG"])
lim = max(c1.max().max(), c4.max().max(), csb.max().max()) * 1.1
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
ax.plot(c1["predicted"], c1["actual"], "o-", color=BLUE, lw=2, ms=5, label="Baseline 1")
ax.plot(c4["predicted"], c4["actual"], "s-", color=ORANGE, lw=2, ms=5, label="Model 4")
ax.plot(csb["predicted"], csb["actual"], "^-", color=AQUA, lw=2, ms=5, label="StatsBomb xG (reference)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title("Calibration on validation (10 bins)", loc="left", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/model4_calibration.png"); plt.close(fig)
print("\nModel 4 calibration table (validation):")
print(c4.round(3).to_string())
print("\nSaved figures/model4_calibration.png and data/processed/model_results.csv")