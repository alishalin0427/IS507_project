import os
import glob
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 讀入五個聯賽的射門資料
shots = pd.concat(
    [pd.read_parquet(f) for f in glob.glob("data/processed/shots_*_1516.parquet")],
    ignore_index=True,
)

# StatsBomb 球場範圍：x 在 0–120，y 在 0–80
out_of_range = shots[~shots["x"].between(0, 120) | ~shots["y"].between(0, 80)].copy()

# 計算超出邊界多少（0 代表在範圍內）
out_of_range["x_overshoot"] = np.maximum(out_of_range["x"] - 120, 0) + np.maximum(0 - out_of_range["x"], 0)
out_of_range["y_overshoot"] = np.maximum(out_of_range["y"] - 80, 0) + np.maximum(0 - out_of_range["y"], 0)

cols = ["league", "match_id", "team", "player", "minute", "x", "y",
        "x_overshoot", "y_overshoot", "shot_type", "shot_outcome", "shot_statsbomb_xg"]
print(f"Shots outside the pitch: {len(out_of_range)}\n")
print(out_of_range[cols].to_string(index=False))

# 順便檢查：在自己半場（x < 60）的射門，可能是方向標反了
own_half = shots[shots["x"] < 60]
print(f"\nShots from own half (x < 60): {len(own_half)}")
print(own_half[["league", "player", "x", "y", "shot_outcome"]].head(10).to_string(index=False))