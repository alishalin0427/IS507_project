import glob
import pandas as pd

shots = pd.concat(pd.read_parquet(f) for f in glob.glob("data/processed/shots_*_1516.parquet"))
missing = shots[shots["shot_freeze_frame"].isna()]

print("缺 freeze frame 的射門：", len(missing))
print(missing["shot_type"].value_counts())      # 是不是集中在點球或自由球？
print(missing["play_pattern"].value_counts())   # 是不是集中在某種情境？
print(missing["league"].value_counts())         # 是不是集中在某個聯賽？
print("集中在幾場比賽：", missing["match_id"].nunique())
print("進球率：", (missing["shot_outcome"] == "Goal").mean())   # 和整體 9.5% 比較