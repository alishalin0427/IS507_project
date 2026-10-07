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

# 2. 保留賽前資訊，並處理缺少的 Bet365 賠率
odds_cols = ["league", "Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]

# 如果 B365 缺失，嘗試用其他主流博彩公司（如 Interwetten 或 William Hill）替換，或以各家平均填補
for h, d, a in [("IWH", "IWD", "IWA"), ("WHH", "WHD", "WHA"), ("VCH", "VCD", "VCA")]:
    if h in odds.columns:
        odds["B365H"] = odds["B365H"].fillna(odds[h])
        odds["B365D"] = odds["B365D"].fillna(odds[d])
        odds["B365A"] = odds["B365A"].fillna(odds[a])

odds = odds[odds_cols].dropna()
odds["match_date"] = pd.to_datetime(odds["Date"], format="mixed", dayfirst=True)

# 3. 賠率轉成機率
inv = 1 / odds[["B365H", "B365D", "B365A"]]
inv = inv.div(inv.sum(axis=1), axis=0)
odds["p_home"], odds["p_draw"], odds["p_away"] = inv["B365H"], inv["B365D"], inv["B365A"]
print("賠率資料：", len(odds), "場")   # 此時會是完整的 1,826 場

# 3. 賠率轉成機率：1/賠率，再除以總和，去掉莊家抽成
inv = 1 / odds[["B365H", "B365D", "B365A"]]
inv = inv.div(inv.sum(axis=1), axis=0)
odds["p_home"], odds["p_draw"], odds["p_away"] = inv["B365H"], inv["B365D"], inv["B365A"]
print("賠率資料：", len(odds), "場")   # 預期 380×4 + 306 = 1,826

# 4. 讀取 matches 資料並確保有名稱對照欄位
matches = pd.concat(pd.read_csv(f) for f in glob.glob("data/processed/matches_*_1516.csv"))
matches["match_date"] = pd.to_datetime(matches["match_date"])

# 檢查 matches 裡的聯賽欄位名稱
# 如果 matches 沒有 "league"，但有 "competition_name"，把它統一命名為 league
if "league" not in matches.columns and "competition_name" in matches.columns:
    matches["league"] = matches["competition_name"]

# 讀取球隊名稱對照表並替換 odds 裡的名字
name_map = pd.read_csv("data/team_name_map.csv")
NAME_MAP = dict(zip(name_map["fd_name"], name_map["sb_name"]))
odds["HomeTeam"] = odds["HomeTeam"].replace(NAME_MAP)
odds["AwayTeam"] = odds["AwayTeam"].replace(NAME_MAP)

# 5. 合併比賽與賠率
# 如果 matches 依然沒有 league 欄位，直接用主客隊 (home_team, away_team) 合併即可：
if "league" in matches.columns:
    merged = pd.merge(
        matches,
        odds,
        left_on=["league", "home_team", "away_team"],
        right_on=["league", "HomeTeam", "AwayTeam"],
        how="left"
    )
else:
    # 每個賽季相同主客場只會對戰一次，直接用隊名對齊
    merged = pd.merge(
        matches,
        odds,
        left_on=["home_team", "away_team"],
        right_on=["HomeTeam", "AwayTeam"],
        how="left"
    )

unmatched = merged[merged["B365H"].isna()]
print(f"對不上賠率的比賽數量：{len(unmatched)}")
if len(unmatched) > 0:
    print("\n對不上的比賽：")
    print(unmatched[["match_date_x", "home_team", "away_team"]])