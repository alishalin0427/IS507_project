"""
eda.py — 探索性資料分析（EDA）
用法（在 IS507_project/ 資料夾執行）：
    python eda.py                     # 分析 data/processed/ 裡所有聯賽
    python eda.py pl laliga ligue1    # 只分析指定聯賽（代碼 = 檔名中間那段）

輸入：data/processed/shots_<code>_1516.parquet   （download_leagues.py 產生）
      data/processed/matches_<code>_1516.csv
輸出：終端機印出的表格（貼進報告）＋ figures/ 裡的圖
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # 不開視窗，直接存檔
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)
pd.set_option("display.width", 140)

# ---------- 圖表樣式：淡色格線、單一藍色、進球用橘色 ----------
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "font.size": 9,
})

def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

# ---------- 0. 讀資料 ----------
codes = sys.argv[1:]
if codes:
    shot_files = [f"data/processed/shots_{c}_1516.parquet" for c in codes]
else:
    shot_files = sorted(glob.glob("data/processed/shots_*_1516.parquet"))
shots = pd.concat([pd.read_parquet(f) for f in shot_files], ignore_index=True)
match_files = [f.replace("shots_", "matches_").replace(".parquet", ".csv") for f in shot_files]
matches = pd.concat([pd.read_csv(f) for f in match_files if os.path.exists(f)], ignore_index=True)

# 距離與角度（StatsBomb 球場 120 x 80，球門中心 (120, 40)，門柱 y = 36 與 44）
shots["distance"] = np.hypot(120 - shots["x"], 40 - shots["y"])
shots["angle"] = np.abs(np.arctan2(44 - shots["y"], 120 - shots["x"])
                        - np.arctan2(36 - shots["y"], 120 - shots["x"]))
shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
shots["is_penalty"] = shots["shot_type"] == "Penalty"

# ---------- 1. 資料規模與結構 ----------
section("1. 資料規模（每個聯賽）")
size = shots.groupby("league").agg(
    matches=("match_id", "nunique"),
    shots=("id", "size"),
    goals=("is_goal", "sum"),
    penalties=("is_penalty", "sum"),
)
size["shots_per_match"] = (size["shots"] / size["matches"]).round(1)
size.loc["TOTAL"] = size.sum(numeric_only=True)
size.loc["TOTAL", "shots_per_match"] = round(size.loc["TOTAL", "shots"] / size.loc["TOTAL", "matches"], 1)
print(size)
print("\n比賽清單中的場次：", len(matches), "｜ 有射門資料的場次：", shots["match_id"].nunique())

# ---------- 2. 資料品質：重複、缺失、異常值 ----------
section("2. 資料品質")
print("重複的射門事件 ID：", shots["id"].duplicated().sum())
print("重複的比賽 ID（比賽清單）：", matches["match_id"].duplicated().sum() if len(matches) else "n/a")
bad_xy = ~shots["x"].between(0, 120) | ~shots["y"].between(0, 80)
print("座標超出球場範圍：", bad_xy.sum())
print("座標缺失：", shots[["x", "y"]].isna().any(axis=1).sum())

print("\n射門結果分布：")
print(shots["shot_outcome"].value_counts(dropna=False).to_string())
print("\n射門類型分布：")
print(shots["shot_type"].value_counts(dropna=False).to_string())

print("\n各欄位缺失比例（%）：")
print((shots.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())
print("\n注意：StatsBomb 的是非欄位（under_pressure、shot_first_time）只在「是」的時候記錄，"
      "\n缺失代表「否」，不是真的缺資料。報告要說明這個處理。")

ff = shots["shot_freeze_frame"].notna() if "shot_freeze_frame" in shots else pd.Series(False, index=shots.index)
print("\n有 freeze frame 的射門比例：", f"{ff.mean():.1%}")
print("有助攻傳球（shot_key_pass_id）的射門比例：", f"{shots['shot_key_pass_id'].notna().mean():.1%}")

# ---------- 3. 重複觀測：射門集中在比賽、球隊、球員 ----------
section("3. 重複觀測（叢集結構）")
per_match = shots.groupby("match_id").size()
print("每場射門數：", per_match.describe().round(1).to_dict())
per_player = shots.groupby("player").size().sort_values(ascending=False)
print(f"\n共 {len(per_player)} 名球員射門；射門最多的 10 人：")
print(per_player.head(10).to_string())
print(f"\n射門最多的 10% 球員，貢獻了 {per_player.head(len(per_player)//10).sum()/len(shots):.0%} 的射門")
print("→ 射門不是彼此獨立的，所以切分資料要以「比賽」為單位。")

# ---------- 4. 目標變數：進球率（排除點球） ----------
section("4. 目標變數：進球率（排除點球）")
np_shots = shots[~shots["is_penalty"]].copy()
rate = np_shots.groupby("league")["is_goal"].agg(["size", "sum", "mean"])
rate.columns = ["shots", "goals", "goal_rate"]
rate["goal_rate"] = rate["goal_rate"].round(3)
print(rate)
print(f"\n整體進球率：{np_shots['is_goal'].mean():.3f} → 類別不平衡，不能用 accuracy 評估")
pen = shots[shots["is_penalty"]]
if len(pen):
    print(f"點球進球率：{pen['is_goal'].mean():.3f}（{len(pen)} 次）→ 排除並分開處理的理由")

# ---------- 5. 分布描述（排除點球） ----------
section("5. 各特徵的進球率")
for col in ["shot_body_part", "play_pattern", "shot_technique"]:
    t = np_shots.groupby(col)["is_goal"].agg(["size", "mean"]).sort_values("size", ascending=False)
    t.columns = ["shots", "goal_rate"]
    t["goal_rate"] = t["goal_rate"].round(3)
    print(f"\n{col}：\n{t.to_string()}")

# 圖 1：進球率 vs 距離（每 2 公尺一組）— 最重要的一張圖
bins = np.arange(0, 42, 2)
np_shots["dist_bin"] = pd.cut(np_shots["distance"], bins)
by_dist = np_shots.groupby("dist_bin", observed=True)["is_goal"].agg(["mean", "size"])
by_dist = by_dist[by_dist["size"] >= 30]                      # 樣本太少的組別不畫
mid = [b.mid for b in by_dist.index]
se = np.sqrt(by_dist["mean"] * (1 - by_dist["mean"]) / by_dist["size"])
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.errorbar(mid, by_dist["mean"], yerr=1.96 * se, fmt="o-", color=BLUE, lw=2, ms=5,
            ecolor=BLUE, elinewidth=1, capsize=2)
ax.set_xlabel("Distance to goal centre (StatsBomb units ≈ yards)")
ax.set_ylabel("Goal rate")
ax.set_title("Goal rate falls steeply with distance (non-penalty shots, 95% CI)", loc="left", color=INK)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.tight_layout(); fig.savefig("figures/eda_goal_rate_by_distance.png"); plt.close(fig)

# 圖 2：射門距離分布，進球 vs 未進
fig, ax = plt.subplots(figsize=(6.5, 3.4))
kw = dict(bins=np.arange(0, 45, 1.5), density=True, histtype="step", lw=2)
ax.hist(np_shots.loc[np_shots.is_goal == 0, "distance"], color=BLUE, label="No goal", **kw)
ax.hist(np_shots.loc[np_shots.is_goal == 1, "distance"], color=ORANGE, label="Goal", **kw)
ax.set_xlabel("Distance to goal centre"); ax.set_ylabel("Density")
ax.set_title("Goals come from much closer than the average shot", loc="left", color=INK)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/eda_distance_hist.png"); plt.close(fig)

# 圖 3：射門位置（半場），顏色 = 進球率
fig, ax = plt.subplots(figsize=(5.2, 4.2))
hb = ax.hexbin(np_shots["y"], np_shots["x"], C=np_shots["is_goal"], reduce_C_function=np.mean,
               gridsize=22, mincnt=15, cmap="Blues", extent=(0, 80, 60, 120))
ax.plot([36, 44], [120, 120], color=INK, lw=3)                 # 球門
ax.plot([18, 18, 62, 62], [120, 102, 102, 120], color=MUTED, lw=1)  # 禁區
ax.set_xlim(0, 80); ax.set_ylim(60, 121); ax.set_aspect("equal"); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(hb, ax=ax, shrink=0.8, label="Goal rate (cells with ≥15 shots)")
ax.set_title("Where shots are taken and how often they score", loc="left", color=INK)
fig.tight_layout(); fig.savefig("figures/eda_shot_map.png"); plt.close(fig)

# 圖 4：各聯賽進球率（橫條圖，附 95% CI）
r = rate.sort_values("goal_rate")
se = np.sqrt(r["goal_rate"] * (1 - r["goal_rate"]) / r["shots"])
fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(r) + 1.2))
ax.barh(r.index, r["goal_rate"], xerr=1.96 * se, color=BLUE, height=0.55,
        error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
ax.set_xlabel("Goal rate (non-penalty)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Conversion rate by league (95% CI)", loc="left", color=INK)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/eda_goal_rate_by_league.png"); plt.close(fig)

# ---------- 6. 和 StatsBomb 官方 xG 對照（只當參考，不當特徵） ----------
section("6. 實際進球 vs StatsBomb 官方 xG（排除點球）")
cmp_ = np_shots.groupby("league").agg(goals=("is_goal", "sum"), statsbomb_xg=("shot_statsbomb_xg", "sum"))
cmp_["statsbomb_xg"] = cmp_["statsbomb_xg"].round(1)
cmp_["goals_minus_xg"] = (cmp_["goals"] - cmp_["statsbomb_xg"]).round(1)
print(cmp_)

# ---------- 7. 資料量夠不夠（初步） ----------
section("7. 資料量")
n_goals = int(np_shots["is_goal"].sum())
print(f"非點球射門 {len(np_shots)} 次，其中進球 {n_goals} 次。")
print(f"以「每個參數至少約 10–20 個進球」的經驗法則，約可支撐 {n_goals // 20}–{n_goals // 10} 個參數。")
print("正式判斷請在建模後畫 learning curve。")

print("\n圖已存到 figures/：", sorted(os.listdir("figures")))