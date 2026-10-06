import os, glob, time, json, warnings
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")                       # 隱藏「open data only」警告
os.chdir(os.path.dirname(os.path.abspath(__file__)))    # 以本檔所在位置為準
os.makedirs("data/processed", exist_ok=True)

# ===== 要下載的聯賽：名稱必須和 StatsBomb 的寫法一模一樣 =====
LEAGUES = {
    "Premier League": "pl",
    "La Liga": "laliga",
    "Ligue 1": "ligue1",
}
SEASON = "2015/2016"

SHOT_COLS = ["match_id", "id", "team", "player", "period", "minute", "second",
             "location", "shot_outcome", "shot_type", "shot_body_part",
             "shot_technique", "play_pattern", "shot_first_time",
             "under_pressure", "shot_statsbomb_xg", "shot_key_pass_id",
             "shot_freeze_frame"]

# 先印出這一季所有可用的聯賽，方便核對名稱
comps = sb.competitions()
print(comps[comps.season_name == SEASON][["competition_id", "season_id", "competition_name"]])

all_shots = []

for name, code in LEAGUES.items():
    # ---- 1. 找代碼 ----
    row = comps[(comps.competition_name == name) & (comps.season_name == SEASON)]
    if row.empty:
        print(f"\n找不到「{name}」，請對照上面印出的名稱修改 LEAGUES")
        continue
    cid, sid = int(row.competition_id.iloc[0]), int(row.season_id.iloc[0])

    # ---- 2. 比賽清單 ----
    matches = sb.matches(competition_id=cid, season_id=sid)
    matches.to_csv(f"data/processed/matches_{code}_1516.csv", index=False)
    print(f"\n=== {name}：{len(matches)} 場 ===")

    # ---- 3. 逐場下載事件（已下載的會跳過）----
    raw_dir = f"data/raw/{code}_1516"
    os.makedirs(raw_dir, exist_ok=True)
    for i, mid in enumerate(matches.match_id, 1):
        path = f"{raw_dir}/{mid}.pkl"
        if os.path.exists(path):
            continue
        for attempt in range(3):                        # 失敗最多重試 3 次
            try:
                sb.events(match_id=mid).to_pickle(path)
                break
            except Exception as e:
                print(f"  比賽 {mid} 下載失敗（{e}），5 秒後重試")
                time.sleep(5)
        if i % 20 == 0:
            print(f"  已處理 {i}/{len(matches)}")
        time.sleep(0.2)

    files = glob.glob(f"{raw_dir}/*.pkl")
    print(f"  已下載 {len(files)}/{len(matches)} 場")

    # ---- 4. 篩出射門 ----
    ev = pd.concat(pd.read_pickle(f) for f in files)
    shots = ev[ev["type"] == "Shot"].copy()
    shots = shots[[c for c in SHOT_COLS if c in shots.columns]]
    shots["x"] = shots["location"].str[0]
    shots["y"] = shots["location"].str[1]
    shots = shots.drop(columns="location")
    if "shot_freeze_frame" in shots:                    # 巢狀欄位轉成文字才能存 parquet
        shots["shot_freeze_frame"] = shots["shot_freeze_frame"].apply(
            lambda v: json.dumps(v, default=str) if isinstance(v, list) else None)
    shots["league"] = name

    shots.to_parquet(f"data/processed/shots_{code}_1516.parquet", index=False)
    all_shots.append(shots)
    print(f"  射門 {len(shots)} 筆，已存檔")

# ---- 5. 合併並印出摘要 ----
combined = pd.concat(all_shots, ignore_index=True)
combined.to_parquet("data/processed/shots_1516_part1.parquet", index=False)

summary = combined.groupby("league").agg(
    shots=("id", "size"),
    goals=("shot_outcome", lambda s: (s == "Goal").sum()),
    penalties=("shot_type", lambda s: (s == "Penalty").sum()),
)
summary["goal_rate"] = (summary.goals / summary.shots).round(3)
print("\n", summary)
print("重複的射門 ID：", combined["id"].duplicated().sum())