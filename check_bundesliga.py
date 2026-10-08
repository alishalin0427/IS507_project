import os
import glob
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- 步驟 1：StatsBomb 提供哪些德甲資料？ ----------
comps = sb.competitions()
bl = comps[comps["competition_name"].str.contains("Bundesliga", case=False)]
print("=== StatsBomb Bundesliga entries ===")
print(bl[["competition_id", "season_id", "competition_name", "season_name"]].to_string(index=False))

# ---------- 步驟 2：已下載的 34 場是什麼？ ----------
m = pd.read_csv("data/processed/matches_bundesliga_1516.csv")
m["match_date"] = pd.to_datetime(m["match_date"])
teams = pd.concat([m["home_team"], m["away_team"]])

print("\n=== Downloaded match list ===")
print("Matches:", len(m))
print("Date range:", m["match_date"].min().date(), "to", m["match_date"].max().date())
print("Distinct teams:", teams.nunique())                  # 完整賽季應該是 18 隊
print("Teams appearing most often:")
print(teams.value_counts().head(5).to_string())              # 如果某隊出現 34 次 → 只有那一隊的比賽
print("Raw event files on disk:", len(glob.glob("data/raw/bundesliga_1516/*.pkl")))

# ---------- 步驟 3：和 football-data 的 306 場比對 ----------
odds = pd.read_csv("data/raw/odds/D1_1516.csv").dropna(subset=["HomeTeam"])
name_map = pd.read_csv("data/team_name_map.csv")
NAME_MAP = dict(zip(name_map["fd_name"], name_map["sb_name"]))
odds["home_team"] = odds["HomeTeam"].replace(NAME_MAP)
odds["away_team"] = odds["AwayTeam"].replace(NAME_MAP)

cmp_ = odds[["Date", "home_team", "away_team"]].merge(
    m[["match_id", "home_team", "away_team"]],
    on=["home_team", "away_team"], how="outer", indicator=True)
print("\n=== football-data (306) vs StatsBomb ===")
print(cmp_["_merge"].value_counts().to_string())
# left_only  = football-data 有、StatsBomb 沒有 → 缺的比賽
# right_only = StatsBomb 有、football-data 沒有 → 不是 2015/16 德甲的比賽（例如抓錯賽季）
print("\nStatsBomb matches not in 2015/16 football-data (first 5):")
print(cmp_[cmp_["_merge"] == "right_only"].head().to_string(index=False))