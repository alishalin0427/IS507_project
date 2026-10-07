import os, glob
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data/raw/odds", exist_ok=True)

LEAGUE_CODES = {"Premier League": "E0", "La Liga": "SP1", "Ligue 1": "F1",
                "1. Bundesliga": "D1", "Serie A": "I1"}

# 1. 下載五個聯賽的 CSV（下載過就直接讀本地檔）
frames = []
for league, code in LEAGUE_CODES.items():
    path = f"data/raw/odds/{code}_1516.csv"
    if not os.path.exists(path):
        url = f"https://www.football-data.co.uk/mmz4281/1516/{code}.csv"
        pd.read_csv(url, encoding="latin-1").to_csv(path, index=False)
    df = pd.read_csv(path)
    df["league"] = league
    frames.append(df)
odds = pd.concat(frames, ignore_index=True)

# 2. 只保留賽前資訊（比分、射門數等是賽後資料，不能用，會造成資料洩漏）
odds = odds[["league", "Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]].dropna()
odds["match_date"] = pd.to_datetime(odds["Date"], format="mixed", dayfirst=True)

# 3. 賠率轉成機率：1/賠率，再除以總和，去掉莊家抽成
inv = 1 / odds[["B365H", "B365D", "B365A"]]
inv = inv.div(inv.sum(axis=1), axis=0)
odds["p_home"], odds["p_draw"], odds["p_away"] = inv["B365H"], inv["B365D"], inv["B365A"]
print("賠率資料：", len(odds), "場")   # 預期 380×4 + 306 = 1,826

# 4. 找出名稱對不上的球隊
matches = pd.concat(pd.read_csv(f) for f in glob.glob("data/processed/matches_*_1516.csv"))
matches["match_date"] = pd.to_datetime(matches["match_date"])
sb_teams = set(matches["home_team"]) | set(matches["away_team"])
fd_teams = set(odds["HomeTeam"]) | set(odds["AwayTeam"])
print("\nStatsBomb 有、football-data 沒有的名稱：")
print(sorted(sb_teams - fd_teams))
print("\nfootball-data 有、StatsBomb 沒有的名稱：")
print(sorted(fd_teams - sb_teams))