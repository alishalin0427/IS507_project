"""
data_profile.py — 描述兩個資料集的型態：大小、維度、資料類型、一列是什麼
用法（在 IS507_project/ 資料夾執行）：
    python profile_data.py > profile_output.txt 2>&1

會檢查四份資料：
  A. StatsBomb 原始事件檔（data/raw/<league>_1516/*.pkl）
  B. StatsBomb 比賽清單（data/processed/matches_<league>_1516.csv）
  C. 我們整理出的射門表（data/processed/shots_<league>_1516.parquet）
  D. football-data.co.uk 賠率檔（data/raw/odds/<code>_1516.csv）
同時寫出 docs/data_profile.md，可以直接貼進報告或簡報。
"""
import os
import glob
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("docs", exist_ok=True)

LEAGUES = {"pl": "E0", "laliga": "SP1", "ligue1": "F1", "seriea": "I1"}  # StatsBomb 代碼 → football-data 代碼
md = []   # 收集要寫進 docs/data_profile.md 的內容


def kind_of(s):
    """判斷一欄的資料類型：數值、文字/類別、是非、日期、巢狀（list/dict）"""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    first = s.dropna()
    if first.empty:
        return "empty"
    v = first.iloc[0]
    if isinstance(v, (list, dict, np.ndarray)):
        return "nested (list/dict)"
    if isinstance(v, (bool, np.bool_)):
        return "boolean"
    return "text / category"


def type_summary(df):
    """每種資料類型有幾欄"""
    return pd.Series({c: kind_of(df[c]) for c in df.columns}).value_counts()


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    md.append(f"\n## {title}\n")


# ---------- A. StatsBomb 原始事件檔 ----------
section("A. StatsBomb raw events (one row = one on-ball action)")
event_counts, shot_counts, all_cols = [], [], set()
type_totals = pd.Series(dtype=float)
example = None
for code in LEAGUES:
    for f in sorted(glob.glob(f"data/raw/{code}_1516/*.pkl")):
        ev = pd.read_pickle(f)
        event_counts.append(len(ev))
        shot_counts.append(int((ev["type"] == "Shot").sum()))
        all_cols |= set(ev.columns)
        type_totals = type_totals.add(ev["type"].value_counts(), fill_value=0)
        if example is None:
            example = ev
if event_counts:
    ev_counts = np.array(event_counts)
    lines = [
        f"Match files: {len(ev_counts):,}",
        f"Total events (rows): {ev_counts.sum():,}",
        f"Events per match: mean {ev_counts.mean():,.0f}, min {ev_counts.min():,}, max {ev_counts.max():,}",
        f"Distinct columns across all files: {len(all_cols)}",
        f"Shots: {sum(shot_counts):,} ({sum(shot_counts) / ev_counts.sum():.1%} of all events)",
    ]
    for l in lines:
        print(l)
    md += [f"- {l}" for l in lines]
    print("\nColumn types in one example match:")
    print(type_summary(example).to_string())
    top = (type_totals.sort_values(ascending=False).head(10) / type_totals.sum() * 100).round(1)
    print("\nMost common event types (% of all events):")
    print(top.to_string())
    md.append("\n| Event type | Share of events |\n| --- | --- |")
    md += [f"| {t} | {p}% |" for t, p in top.items()]
else:
    print("No raw event files found in data/raw/<league>_1516/")

# ---------- B. StatsBomb 比賽清單 ----------
section("B. StatsBomb match list (one row = one match)")
matches = pd.concat([pd.read_csv(f"data/processed/matches_{c}_1516.csv") for c in LEAGUES],
                    ignore_index=True)
print(f"Rows: {len(matches):,}   Columns: {matches.shape[1]}")
print(type_summary(matches).to_string())
md.append(f"- Rows: {len(matches):,}; columns: {matches.shape[1]}")

# ---------- C. 射門表（建模用的資料） ----------
section("C. Shot table we built (one row = one shot)")
shots = pd.concat([pd.read_parquet(f"data/processed/shots_{c}_1516.parquet") for c in LEAGUES],
                  ignore_index=True)
n_pen = int((shots["shot_type"] == "Penalty").sum())
print(f"Rows: {len(shots):,} (non-penalty: {len(shots) - n_pen:,})   Columns: {shots.shape[1]}")
print(type_summary(shots).to_string())
md.append(f"- Rows: {len(shots):,} shots ({len(shots) - n_pen:,} after removing {n_pen} penalties); "
          f"columns: {shots.shape[1]}")

# 每一欄的資料字典：類型、例子、缺失比例
role = {
    "x": "X (location)", "y": "X (location)",
    "shot_body_part": "X (Model 2)", "play_pattern": "X (Model 3)",
    "shot_outcome": "Y (goal or not)", "shot_statsbomb_xg": "Reference only, never a feature",
    "shot_type": "Used to remove penalties", "match_id": "Used to split train/valid/test",
    "shot_freeze_frame": "Not used yet: player positions (nested list stored as JSON text)",
    "under_pressure": "Not used yet; blank = False", "shot_first_time": "Not used yet; blank = False",
    "shot_technique": "Not used yet", "shot_key_pass_id": "Not used yet; blank = no assist",
    "league": "Used for error analysis", "team": "Identifier", "player": "Identifier",
}
rows = []
for c in shots.columns:
    s = shots[c]
    ex = s.dropna().iloc[0] if s.notna().any() else ""
    ex = str(ex)[:40] + ("..." if len(str(ex)) > 40 else "")
    rows.append({"column": c, "type": kind_of(s), "example": ex,
                 "missing_%": round(100 * s.isna().mean(), 1), "role": role.get(c, "")})
dictionary = pd.DataFrame(rows)
print("\nData dictionary:")
print(dictionary.to_string(index=False))
md.append("\n| Column | Type | Example | Missing | Role in project |\n| --- | --- | --- | --- | --- |")
md += [f"| `{r.column}` | {r.type} | {r.example} | {r['missing_%']}% | {r.role} |"
       for _, r in dictionary.iterrows()]

# ---------- D. football-data.co.uk ----------
section("D. football-data.co.uk odds files (one row = one match)")
frames = []
for sb_code, fd_code in LEAGUES.items():
    path = f"data/raw/odds/{fd_code}_1516.csv"
    if os.path.exists(path):
        d = pd.read_csv(path).dropna(subset=["HomeTeam"])
        d["league_code"] = fd_code
        frames.append(d)
if frames:
    odds = pd.concat(frames, ignore_index=True)
    per_file = odds.groupby("league_code").size()
    print(f"Rows: {len(odds):,}   Columns (union of files): {odds.shape[1] - 1}")
    print("Rows per file:", per_file.to_dict())
    print(type_summary(odds.drop(columns="league_code")).to_string())
    odds_cols = [c for c in odds.columns if c.endswith(("H", "D", "A")) and odds[c].dtype.kind == "f"]
    print(f"Columns that look like bookmaker odds (home/draw/away): about {len(odds_cols)}")
    print("Columns we use: Date, HomeTeam, AwayTeam, B365H, B365D, B365A")
    md += [f"- Rows: {len(odds):,} matches; columns: {odds.shape[1] - 1}",
           f"- Rows per file: {per_file.to_dict()}",
           "- Columns we use: Date, HomeTeam, AwayTeam, B365H, B365D, B365A (pre-match only)"]
else:
    print("No odds files found in data/raw/odds/")

with open("docs/data_profile.md", "w") as fh:
    fh.write("# Data profile\n" + "\n".join(md) + "\n")
print("\nSaved docs/data_profile.md")