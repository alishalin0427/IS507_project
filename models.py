"""
models.py — Model 2 and Model 3: add body part, then play pattern, to Baseline 1
Usage (run in the project folder, after baseline.py has written split.csv and predictions.csv):
    python models.py > outputs/models_output.txt 2>&1

Models (each one keeps every feature of the one before):
  Baseline 1 : distance, angle
  Model 2    : + body part      -> targets "headers over-predicted"
  Model 3    : + play pattern   -> targets "corners over-predicted, counters under-predicted"

Outputs:
  - validation comparison table (also saved as data/processed/model_results.csv)
  - coefficients and odds ratios of Model 2 and Model 3
  - error analysis by group, Baseline 1 vs Model 3 (training set)
  - figures/model_calibration.png (validation)
  - adds columns p_m2, p_m3 to data/processed/predictions.csv
The test set is not used here (sealed until the final model is chosen).
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

# ---------- 1. Load and clean (exactly the same steps as baseline.py) ----------
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

# ---------- 2. Turn text columns into numbers (dummy variables) ----------
# Body part: reference = foot (Left Foot and Right Foot together).
# A 1 in is_header means "header instead of a foot shot"; its coefficient is the
# difference from a foot shot taken from the same distance and angle.
shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
shots["is_other_body"] = (shots["shot_body_part"] == "Other").astype(int)

# Play pattern: reference = Regular Play. Rare patterns are grouped as other_pattern
# so that each dummy has enough shots.
PATTERNS = {
    "from_counter": "From Counter",
    "from_corner": "From Corner",
    "from_free_kick": "From Free Kick",
    "from_throw_in": "From Throw In",
}
for col, value in PATTERNS.items():
    shots[col] = (shots["play_pattern"] == value).astype(int)
shots["other_pattern"] = (~shots["play_pattern"].isin(
    list(PATTERNS.values()) + ["Regular Play"])).astype(int)

# ---------- 3. Use the same split as baseline.py ----------
split = pd.read_csv("data/processed/split.csv")
shots = shots.merge(split, on="match_id", how="left")
assert shots["split"].notna().all(), "Some matches are missing from split.csv: rerun baseline.py"
train = shots[shots["split"] == "train"].copy()
valid = shots[shots["split"] == "valid"].copy()

# ---------- 4. Define the models ----------
BODY = ["is_header", "is_other_body"]
PATTERN = list(PATTERNS) + ["other_pattern"]
MODELS = {
    "Baseline 1": ["distance", "angle"],
    "Model 2":    ["distance", "angle"] + BODY,
    "Model 3":    ["distance", "angle"] + BODY + PATTERN,
}

# ---------- 5. Fit on train, score on validation ----------
y = valid["is_goal"]
base_rate = train["is_goal"].mean()
rows = [{"model": "Baseline 0", "features": "none (training goal rate)",
         "log_loss": log_loss(y, np.full(len(valid), base_rate)), "mean_prediction": base_rate}]
fitted = {}
for name, feats in MODELS.items():
    m = LogisticRegression(max_iter=2000)          # same settings as baseline.py
    m.fit(train[feats], train["is_goal"])
    p = m.predict_proba(valid[feats])[:, 1]
    fitted[name] = (m, feats)
    valid[name] = p
    rows.append({"model": name, "features": ", ".join(feats),
                 "log_loss": log_loss(y, p), "mean_prediction": p.mean()})

# Benchmark only, never a feature (data_log F2): StatsBomb's own xG
sb = valid["shot_statsbomb_xg"].clip(1e-6, 1 - 1e-6)   # avoid log(0)
valid["StatsBomb xG"] = sb
rows.append({"model": "StatsBomb xG (benchmark)", "features": "-",
             "log_loss": log_loss(y, sb), "mean_prediction": sb.mean()})

results = pd.DataFrame(rows)
results["actual_goal_rate"] = y.mean()
ll = results.set_index("model")["log_loss"]
results["vs_previous_%"] = (100 * results["log_loss"].pct_change()).round(1)
results.loc[results["model"].str.startswith("StatsBomb"), "vs_previous_%"] = np.nan
print(f"Validation: {len(valid)} shots, {y.sum()} goals")
print("\nValidation results (lower log loss is better):")
print(results.drop(columns="features").round(4).to_string(index=False))

# How much of the gap between Baseline 1 and StatsBomb xG do our features close?
gap = ll["Baseline 1"] - ll["StatsBomb xG (benchmark)"]
for name in ["Model 2", "Model 3"]:
    print(f"Share of the Baseline 1 -> StatsBomb gap closed by {name}: "
          f"{100 * (ll['Baseline 1'] - ll[name]) / gap:.0f}%")
results.to_csv("data/processed/model_results.csv", index=False)

# ---------- 6. Coefficients as odds ratios ----------
# odds ratio = e^coef: < 1 lowers the odds of a goal, > 1 raises them,
# compared with the reference (foot, Regular Play) and holding the other features fixed.
for name in ["Model 2", "Model 3"]:
    m, feats = fitted[name]
    coef = pd.DataFrame({"feature": ["intercept"] + feats,
                         "coef": [m.intercept_[0]] + list(m.coef_[0])})
    coef["odds_ratio"] = np.exp(coef["coef"])
    coef.loc[0, "odds_ratio"] = np.nan              # an odds ratio for the intercept means nothing
    print(f"\n{name} coefficients:")
    print(coef.round(3).to_string(index=False))

# ---------- 7. Error analysis: Baseline 1 vs Model 3 (same method as baseline.py) ----------
# Training set: same data as the midterm analysis.
# NOTE: on the training set, a logistic regression with a dummy for a group matches that
# group's goal total by construction, so Model 3's z for body part and play pattern is ~0
# there automatically. The validation set (not used for fitting) is the honest check.
common = ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In"]

def make_diag(df):
    d = df.copy()
    for name in ["Baseline 1", "Model 3"]:
        m, feats = fitted[name]
        d[name] = m.predict_proba(d[feats])[:, 1]
    d["distance_band"] = pd.cut(d["distance"], [0, 6, 12, 18, 25, 200],
                                labels=["0-6", "6-12", "12-18", "18-25", "25+"])
    d["play_pattern_grp"] = d["play_pattern"].where(d["play_pattern"].isin(common), "Other")
    return d

def compare_errors(diag, col):
    """Per group: actual goals vs expected goals (sum of p) for each model, and z."""
    out = diag.groupby(col, observed=True).agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
    for name, short in [("Baseline 1", "B1"), ("Model 3", "M3")]:
        g = diag.groupby(col, observed=True)[name]
        xg = g.sum()
        var = g.apply(lambda p: (p * (1 - p)).sum())
        out[f"{short}_xg"] = xg.round(1)
        out[f"{short}_goals_minus_xg"] = (out["goals"] - xg).round(1)
        out[f"{short}_z"] = ((out["goals"] - xg) / np.sqrt(var)).round(1)
    return out

for set_name, df in [("training set", train), ("validation set", valid)]:
    diag = make_diag(df)
    print("\n" + "=" * 70)
    print(f"Error analysis: Baseline 1 (B1) vs Model 3 (M3), {set_name}")
    print("goals_minus_xg > 0: under-predicts; < 0: over-predicts; |z| > 2: unlikely to be chance")
    print("=" * 70)
    for col in ["shot_body_part", "play_pattern_grp", "league", "distance_band"]:
        print(f"\nBy {col}:")
        print(compare_errors(diag, col).to_string())

# ---------- 8. Calibration on validation: Baseline 1, Model 3, StatsBomb xG ----------
def calib_table(p):
    d = pd.DataFrame({"p": np.asarray(p), "y": y.values})
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"),
                                               shots=("y", "size"))

tables = {"Baseline 1": calib_table(valid["Baseline 1"]),
          "Model 3": calib_table(valid["Model 3"]),
          "StatsBomb xG": calib_table(valid["StatsBomb xG"])}
print("\nModel 3 calibration table (validation):")
print(tables["Model 3"].round(3).to_string())

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
lim = max(t[["predicted", "actual"]].max().max() for t in tables.values()) * 1.1
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
for (name, t), color, marker in zip(tables.items(), [BLUE, ORANGE, AQUA], ["o", "s", "^"]):
    ax.plot(t["predicted"], t["actual"], marker=marker, ls="-", color=color, lw=2, ms=5, label=name)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Predicted goal probability"); ax.set_ylabel("Actual goal rate")
ax.set_title("Calibration on validation (10 bins)", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#e6e5e1", lw=0.6)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/model_calibration.png"); plt.close(fig)
print("\nSaved figures/model_calibration.png and data/processed/model_results.csv")

# ---------- 9. Add p_m2, p_m3 to predictions.csv (train + validation only, like p_b1) ----------
pred = pd.read_csv("data/processed/predictions.csv")
pred = pred.drop(columns=["p_m2", "p_m3"], errors="ignore")   # safe to rerun this script
new = pd.concat([train, valid])[["id"] + MODELS["Model 3"]].rename(columns={"id": "shot_id"})
for name, col in [("Model 2", "p_m2"), ("Model 3", "p_m3")]:
    m, feats = fitted[name]
    new[col] = m.predict_proba(new[feats])[:, 1]
pred = pred.merge(new[["shot_id", "p_m2", "p_m3"]], on="shot_id", how="left")
assert pred[["p_m2", "p_m3"]].notna().all().all(), "Some shots in predictions.csv got no prediction"
pred.to_csv("data/processed/predictions.csv", index=False)
print(f"Added p_m2, p_m3 to data/processed/predictions.csv ({len(pred)} shots)")