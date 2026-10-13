"""
show_shot.py — 產生 slide 6 要截圖的兩段輸出

1) 一筆「射門」事件的原始 JSON（從 StatsBomb GitHub 讀取，需要網路）
2) 我們射門表 shots_pl_1516.parquet 的前 5 列（只顯示有用到的欄位）
"""
import os
import json
import requests
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MATCH_ID = 3754173   # 英超的一場比賽（射門表中的第一個 match_id）

# ---------- 1. 原始 JSON：找出這場比賽的第一次射門 ----------
url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{MATCH_ID}.json"
events = requests.get(url, timeout=60).json()
shot = next(e for e in events if e["type"]["name"] == "Shot")

# freeze frame 很長（每位球員一筆），只留前 2 筆方便截圖
ff = shot["shot"].get("freeze_frame", [])
if len(ff) > 2:
    shot["shot"]["freeze_frame"] = ff[:2] + [f"... {len(ff) - 2} more players ..."]

# 只保留和射門有關的欄位，畫面才放得下
keep = ["minute", "second", "type", "play_pattern", "team", "player", "location", "shot"]
print("=" * 60)
print("1) RAW STATSBOMB EVENT (one shot, nested JSON)")
print("=" * 60)
print(json.dumps({k: shot[k] for k in keep if k in shot}, indent=2, ensure_ascii=False))

# ---------- 2. 攤平後的射門表 ----------
cols = ["match_id", "team", "player", "x", "y", "shot_body_part",
        "play_pattern", "shot_type", "shot_outcome", "shot_statsbomb_xg"]
shots = pd.read_parquet("data/processed/shots_pl_1516.parquet")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n" + "=" * 60)
print("2) OUR SHOT TABLE (one row = one shot)")
print("=" * 60)
print(shots[cols].head(5).round(3).to_string(index=False))
print(f"\n{len(shots):,} rows x {shots.shape[1]} columns (Premier League only)")