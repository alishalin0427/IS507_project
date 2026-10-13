"""
download_odds.py — 從 football-data.co.uk 下載 2015/16 五個聯賽的賽果與賠率檔

輸出：data/raw/odds/<code>_1516.csv
  E0 = Premier League, SP1 = La Liga, F1 = Ligue 1, I1 = Serie A, D1 = Bundesliga
德甲也下載：data_log D3 的檢查（34 場 vs 306 場）需要用到
"""
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("data/raw/odds", exist_ok=True)

CODES = {"E0": "Premier League", "SP1": "La Liga", "F1": "Ligue 1", "I1": "Serie A", "D1": "Bundesliga"}
EXPECTED = {"E0": 380, "SP1": 380, "F1": 380, "I1": 380, "D1": 306}   # 每個聯賽整季應有的場次

for code, name in CODES.items():
    url = f"https://www.football-data.co.uk/mmz4281/1516/{code}.csv"
    path = f"data/raw/odds/{code}_1516.csv"
    df = pd.read_csv(url, encoding="latin-1")          # 網站的檔案不是 UTF-8，球隊名有重音符號
    df.to_csv(path, index=False)                        # 原樣存下，清理留給 odds.py
    n = df["HomeTeam"].notna().sum()                    # 不算空白列（data_log J3）
    flag = "OK" if n == EXPECTED[code] else f"CHECK: expected {EXPECTED[code]}"
    print(f"{code:4s} {name:15s} {n:4d} matches  {df.shape[1]:3d} columns  -> {path}  [{flag}]")

print("\nDone.")