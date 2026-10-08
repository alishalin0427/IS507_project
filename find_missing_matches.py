import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. 讀 football-data 的法甲資料，套用名稱對照表
odds = pd.read_csv("data/raw/odds/F1_1516.csv")
name_map = pd.read_csv("data/team_name_map.csv")
NAME_MAP = dict(zip(name_map["fd_name"], name_map["sb_name"]))
odds["home_team"] = odds["HomeTeam"].replace(NAME_MAP)
odds["away_team"] = odds["AwayTeam"].replace(NAME_MAP)
odds["fd_date"] = pd.to_datetime(odds["Date"], format="mixed", dayfirst=True)

# 2. 讀 StatsBomb 的法甲比賽清單
sb = pd.read_csv("data/processed/matches_ligue1_1516.csv")
sb["sb_date"] = pd.to_datetime(sb["match_date"])

# 3. 用「主隊＋客隊」做外部合併；indicator 會標出每一列來自哪一邊
m = odds[["fd_date", "home_team", "away_team", "FTHG", "FTAG"]].merge(
    sb[["match_id", "sb_date", "home_team", "away_team"]],
    on=["home_team", "away_team"], how="outer", indicator=True)

print(m["_merge"].value_counts())   # 預期：both 377、left_only 3、right_only 0

# left_only = 只在 football-data 出現 → StatsBomb 缺的比賽
print("\nIn football-data but missing from StatsBomb:")
print(m[m["_merge"] == "left_only"][["fd_date", "home_team", "away_team", "FTHG", "FTAG"]]
      .to_string(index=False))

# right_only = 只在 StatsBomb 出現 → 代表名稱對照還沒完成
print("\nIn StatsBomb but unmatched (check the name map):")
print(m[m["_merge"] == "right_only"][["sb_date", "home_team", "away_team"]]
      .to_string(index=False))

# 4. 配對成功但日期差超過 1 天 → 可能是延期比賽
both = m[m["_merge"] == "both"]
diff = both[(both["fd_date"] - both["sb_date"]).abs() > pd.Timedelta(days=1)]
print(f"\nMatched but dates differ by more than 1 day: {len(diff)}")
print(diff[["home_team", "away_team", "fd_date", "sb_date"]].to_string(index=False))

odds = odds.dropna(subset=["HomeTeam"])   # drop the blank trailing row in the CSV