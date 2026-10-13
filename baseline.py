"""
baseline.py — Baseline 0 與 Baseline 1
做的事：
  1. 讀入四個完整聯賽的射門，排除點球，計算距離與角度
  2. 按「比賽」切分 train / validation / test（70% / 15% / 15%）
  3. Baseline 0：每球都預測訓練資料的平均進球率
  4. Baseline 1：只用距離和角度的 logistic regression
  5. 在 validation set 上計算 log loss 和 Brier score
  6. 畫 Baseline 1 的 calibration curve

輸出：終端機的結果表、data/processed/split.csv、figures/baseline_calibration.png
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

LEAGUES = ["pl", "laliga", "ligue1", "seriea"]   # 德甲已排除（data_log D3）
SEED = 507                                        # 固定亂數，讓切分每次都一樣

# ---------- 1. 讀資料、清理、算特徵 ----------
shots = pd.concat(
    [pd.read_parquet(f"data/processed/shots_{c}_1516.parquet") for c in LEAGUES],
    ignore_index=True,
)
shots = shots[shots["shot_type"] != "Penalty"].copy()       # data_log Q2：排除點球
shots["x"] = shots["x"].clip(0, 120)                         # data_log Q7：座標拉回球場內
shots["y"] = shots["y"].clip(0, 80)

# 距離：射門點到球門中心 (120, 40)
shots["distance"] = np.hypot(120 - shots["x"], 40 - shots["y"])
# 角度：從射門點看兩根門柱 (120, 36) 與 (120, 44) 的夾角，單位是弧度
shots["angle"] = np.abs(np.arctan2(44 - shots["y"], 120 - shots["x"])
                        - np.arctan2(36 - shots["y"], 120 - shots["x"]))
shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

print(f"Non-penalty shots: {len(shots)}, goals: {shots['is_goal'].sum()}, "
      f"goal rate: {shots['is_goal'].mean():.3f}")

# ---------- 2. 按比賽切分 ----------
# 每場比賽隨機分到一組，同一場的射門一定在同一組（data_log R1）
rng = np.random.default_rng(SEED)
match_ids = shots["match_id"].unique()
rng.shuffle(match_ids)
n = len(match_ids)
n_train, n_valid = int(0.70 * n), int(0.15 * n)
split_of = {m: "train" for m in match_ids[:n_train]}
split_of.update({m: "valid" for m in match_ids[n_train:n_train + n_valid]})
split_of.update({m: "test" for m in match_ids[n_train + n_valid:]})
shots["split"] = shots["match_id"].map(split_of)

# 存下切分結果，之後所有模型都用同一份
pd.DataFrame({"match_id": list(split_of), "split": list(split_of.values())}) \
  .to_csv("data/processed/split.csv", index=False)

train = shots[shots["split"] == "train"]
valid = shots[shots["split"] == "valid"]
print("\nSplit (matches / shots / goal rate):")
for name, d in shots.groupby("split"):
    print(f"  {name:5s}  {d['match_id'].nunique():4d} matches  {len(d):6d} shots  {d['is_goal'].mean():.3f}")
# test set 封存到期末，這裡不會用到

# ---------- 3. Baseline 0：每球都預測訓練資料的平均進球率 ----------
base_rate = train["is_goal"].mean()                  # 只用 train 算，不能偷看 valid / test
p0 = np.full(len(valid), base_rate)

# ---------- 4. Baseline 1：距離 + 角度的 logistic regression ----------
features = ["distance", "angle"]
model = LogisticRegression()
model.fit(train[features], train["is_goal"])         # 訓練：在 train 上找出最好的 β
p1 = model.predict_proba(valid[features])[:, 1]      # 預測：替 valid 的每一球算進球機率

print("\nBaseline 1 coefficients:")
print(f"  intercept  {model.intercept_[0]:+.3f}")
for f, b in zip(features, model.coef_[0]):
    print(f"  {f:9s}  {b:+.3f}")
print("  (負數 = 這個值越大，進球機率越低；正數 = 越大越容易進)")

# ---------- 5. 在 validation set 上評分（越低越好） ----------
y = valid["is_goal"]
results = pd.DataFrame({
    "model": ["Baseline 0: average rate", "Baseline 1: distance + angle"],
    "log_loss": [log_loss(y, p0), log_loss(y, p1)],
    "brier": [brier_score_loss(y, p0), brier_score_loss(y, p1)],
    "mean_prediction": [p0.mean(), p1.mean()],
    "actual_goal_rate": [y.mean(), y.mean()],
}).round(4)
print("\nValidation results:")
print(results.to_string(index=False))

# ---------- 6. Calibration curve（Baseline 1） ----------
# 依預測機率分成 10 組（每組射門數相同），比較每組的平均預測與實際進球率
valid = valid.assign(p1=p1)
valid["bin"] = pd.qcut(valid["p1"], 10, duplicates="drop")
calib = valid.groupby("bin", observed=True).agg(
    predicted=("p1", "mean"), actual=("is_goal", "mean"), shots=("is_goal", "size"))
se = np.sqrt(calib["actual"] * (1 - calib["actual"]) / calib["shots"])
print("\nCalibration table (Baseline 1, validation):")
print(calib.round(3).to_string())

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
lim = max(calib["predicted"].max(), calib["actual"].max()) * 1.1
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
ax.errorbar(calib["predicted"], calib["actual"], yerr=1.96 * se, fmt="o-", color=BLUE,
            lw=2, ms=5, capsize=2, label="Baseline 1 (distance + angle)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title("Baseline 1 calibration (validation, 10 bins)", loc="left", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/baseline_calibration.png"); plt.close(fig)
print("\nSaved figures/baseline_calibration.png")

# 7. Error analysis: where does Baseline 1 go wrong?
# ======================================================================
# 用「訓練資料」做診斷：樣本較大（約 2.6 萬球），分組後仍有足夠射門數。
# 這是在找模型的盲點，不是在評分；評分仍然只用 validation set。
diag = train.assign(xg=model.predict_proba(train[features])[:, 1]).copy()
 
# 距離分段，方便看非線性問題
diag["distance_band"] = pd.cut(diag["distance"], [0, 6, 12, 18, 25, 200],
                               labels=["0-6", "6-12", "12-18", "18-25", "25+"])
# 少見的比賽情境併成 Other，避免小樣本雜訊
common = ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In"]
diag["play_pattern_grp"] = diag["play_pattern"].where(diag["play_pattern"].isin(common), "Other")
 
def group_errors(df, col):
    """每組比較：實際進球數 vs 模型預期進球數（xG 加總）"""
    g = df.groupby(col, observed=True).agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
        xg=("xg", "sum"),
        var=("xg", lambda p: (p * (1 - p)).sum()),   # 進球數的變異數（在模型正確的假設下）
    )
    g["actual_rate"] = g["goals"] / g["shots"]
    g["predicted_rate"] = g["xg"] / g["shots"]
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["z"] = g["goals_minus_xg"] / np.sqrt(g["var"])   # |z| > 2 → 偏差不太可能只是隨機
    g["verdict"] = np.where(g["z"] > 2, "UNDER-predicts",
                   np.where(g["z"] < -2, "OVER-predicts", "ok"))
    out = g[["shots", "goals", "xg", "actual_rate", "predicted_rate", "goals_minus_xg", "z", "verdict"]]
    return out.round({"xg": 1, "actual_rate": 3, "predicted_rate": 3, "goals_minus_xg": 1, "z": 1})
 
print("\n" + "=" * 70)
print("Error analysis of Baseline 1 (training set)")
print("goals_minus_xg > 0: model under-predicts; < 0: model over-predicts")
print("=" * 70)
for col in ["shot_body_part", "play_pattern_grp", "league", "distance_band"]:
    print(f"\nBy {col}:")
    print(group_errors(diag, col).to_string())
 