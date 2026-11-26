"""
test_eval.py — Task 16: evaluate the final model (Model 3) on the sealed test set, once
Usage (run in the project folder, after baseline.py, models.py and odd.py):
    python test_eval.py > outputs/test_output.txt 2>&1

Rules (decided before running this script, see data_log M7):
  - The final model is Model 3. Nothing is tuned or changed after seeing these results.
  - Models are fitted on the training set only, exactly as in models.py,
    so the coefficients are the same as the ones reported for validation.
  - Baseline 0, Baseline 1 and StatsBomb xG are shown only as reference points.

Outputs:
  - test log loss table (also saved as data/processed/test_results.csv)
  - error analysis by body part and play pattern (Baseline 1 vs Model 3)
  - calibration by team strength for Model 3 (same favourite rule as calibration_by_strength.py)
  - figures/test_calibration.png
  - data/processed/test_predictions.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

LEAGUES = ["pl", "laliga", "ligue1", "seriea"]   # Bundesliga excluded (data_log D3)

# ---------- 1. Load, clean and build features (same steps as models.py) ----------
shots = pd.concat(
    [pd.read_parquet(f"data/processed/shots_{c}_1516.parquet") for c in LEAGUES],
    ignore_index=True,
)
shots = shots[shots["shot_type"] != "Penalty"].copy()        # data_log Q2
shots["x"] = shots["x"].clip(0, 120)                          # data_log Q7
shots["y"] = shots["y"].clip(0, 80)
shots["distance"] = np.hypot(120 - shots["x"], 40 - shots["y"])
shots["angle"] = np.abs(np.arctan2(44 - shots["y"], 120 - shots["x"])
                        - np.arctan2(36 - shots["y"], 120 - shots["x"]))
shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)        # reference: foot
shots["is_other_body"] = (shots["shot_body_part"] == "Other").astype(int)
PATTERNS = {                                                                 # reference: Regular Play
    "from_counter": "From Counter",
    "from_corner": "From Corner",
    "from_free_kick": "From Free Kick",
    "from_throw_in": "From Throw In",
}
for col, value in PATTERNS.items():
    shots[col] = (shots["play_pattern"] == value).astype(int)
shots["other_pattern"] = (~shots["play_pattern"].isin(
    list(PATTERNS.values()) + ["Regular Play"])).astype(int)

split = pd.read_csv("data/processed/split.csv")
shots = shots.merge(split, on="match_id", how="left")
assert shots["split"].notna().all(), "Some matches are missing from split.csv: rerun baseline.py"
train = shots[shots["split"] == "train"].copy()
test = shots[shots["split"] == "test"].copy()
print(f"Test: {test['match_id'].nunique()} matches, {len(test)} shots, "
      f"{test['is_goal'].sum()} goals, goal rate {test['is_goal'].mean():.4f}")

# ---------- 2. Fit on train (same as models.py), predict the test set ----------
BODY = ["is_header", "is_other_body"]
PATTERN = list(PATTERNS) + ["other_pattern"]
MODELS = {
    "Baseline 1": ["distance", "angle"],
    "Model 3":    ["distance", "angle"] + BODY + PATTERN,
}
base_rate = train["is_goal"].mean()
test["Baseline 0"] = base_rate
fitted = {}
for name, feats in MODELS.items():
    m = LogisticRegression(max_iter=2000)
    m.fit(train[feats], train["is_goal"])
    fitted[name] = (m, feats)
    test[name] = m.predict_proba(test[feats])[:, 1]
test["StatsBomb xG"] = test["shot_statsbomb_xg"].clip(1e-6, 1 - 1e-6)   # benchmark only

# Check: the coefficients must match models.py (same data, same settings)
m3, f3 = fitted["Model 3"]
print("\nModel 3 coefficients (should match models_output.txt):")
print(pd.DataFrame({"feature": ["intercept"] + f3,
                    "coef": [m3.intercept_[0]] + list(m3.coef_[0])}).round(3).to_string(index=False))

# ---------- 3. Test log loss ----------
y = test["is_goal"]
NAMES = ["Baseline 0", "Baseline 1", "Model 3", "StatsBomb xG"]
results = pd.DataFrame({
    "model": NAMES,
    "log_loss": [log_loss(y, test[n]) for n in NAMES],
    "mean_prediction": [test[n].mean() for n in NAMES],
})
results["actual_goal_rate"] = y.mean()
print("\nTest results (lower log loss is better):")
print(results.round(4).to_string(index=False))
ll = results.set_index("model")["log_loss"]
gap = ll["Baseline 1"] - ll["StatsBomb xG"]
print(f"Share of the Baseline 1 -> StatsBomb gap closed by Model 3: "
      f"{100 * (ll['Baseline 1'] - ll['Model 3']) / gap:.0f}%")
results.to_csv("data/processed/test_results.csv", index=False)

# ---------- 4. Error analysis on test: do the midterm biases stay fixed? ----------
common = ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In"]
test["play_pattern_grp"] = test["play_pattern"].where(test["play_pattern"].isin(common), "Other")

def group_errors(df, col, models):
    """Per group: actual goals vs expected goals (sum of p) for each model, and z."""
    out = df.groupby(col, observed=True).agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
    for name, short in models:
        g = df.groupby(col, observed=True)[name]
        xg = g.sum()
        var = g.apply(lambda p: (p * (1 - p)).sum())
        out[f"{short}_xg"] = xg.round(1)
        out[f"{short}_goals_minus_xg"] = (out["goals"] - xg).round(1)
        out[f"{short}_z"] = ((out["goals"] - xg) / np.sqrt(var)).round(1)
    return out

print("\n" + "=" * 70)
print("Error analysis on test: Baseline 1 (B1) vs Model 3 (M3)")
print("goals_minus_xg > 0: under-predicts; < 0: over-predicts; |z| > 2: unlikely to be chance")
print("=" * 70)
for col in ["shot_body_part", "play_pattern_grp"]:
    print(f"\nBy {col}:")
    print(group_errors(test, col, [("Baseline 1", "B1"), ("Model 3", "M3")]).to_string())

# ---------- 5. Calibration by team strength on test (rule from calibration_by_strength.py) ----------
odds = pd.read_csv("data/processed/match_odds.csv")
s = test.merge(odds[["match_id", "home_team", "p_home", "p_away"]], on="match_id", how="left")
print(f"\nTest shots without odds (excluded): {s['p_home'].isna().sum()}")
s = s.dropna(subset=["p_home"]).copy()
is_home = s["team"] == s["home_team"]
p_team = np.where(is_home, s["p_home"], s["p_away"])
p_opp = np.where(is_home, s["p_away"], s["p_home"])
s["group"] = np.where(p_team > p_opp, "favourite", "underdog")
print("\nCalibration by team strength on test (Model 3):")
print(group_errors(s, "group", [("Model 3", "M3")]).to_string())

# ---------- 6. Calibration plot on test: Model 3 vs StatsBomb xG ----------
def calib_table(p):
    d = pd.DataFrame({"p": np.asarray(p), "y": y.values})
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"),
                                               shots=("y", "size"))

tables = {"Model 3": calib_table(test["Model 3"]), "StatsBomb xG": calib_table(test["StatsBomb xG"])}
print("\nModel 3 calibration table (test):")
print(tables["Model 3"].round(3).to_string())

ORANGE, AQUA, MUTED = "#eb6834", "#1baf7a", "#52514e"
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
lim = max(t[["predicted", "actual"]].max().max() for t in tables.values()) * 1.1
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
for (name, t), color, marker in zip(tables.items(), [ORANGE, AQUA], ["s", "^"]):
    ax.plot(t["predicted"], t["actual"], marker=marker, ls="-", color=color, lw=2, ms=5, label=name)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title("Calibration on test (10 bins)", loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/test_calibration.png"); plt.close(fig)

# ---------- 7. Save test predictions (kept separate from predictions.csv) ----------
out = test[["id", "match_id", "team", "league", "is_goal", "Baseline 0", "Baseline 1", "Model 3"]]
out.columns = ["shot_id", "match_id", "team", "league", "is_goal", "p_b0", "p_b1", "p_m3"]
out.to_csv("data/processed/test_predictions.csv", index=False)
print("\nSaved figures/test_calibration.png, data/processed/test_results.csv, "
      "data/processed/test_predictions.csv")