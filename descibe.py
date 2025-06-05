import pandas as pd
import json

df = pd.read_csv("./data/badminton_speed.csv")
print(df["speed"].describe())

with open("./data/smash_data_pseudo_label.json", "r") as f:
    data = json.load(f)

speeds = pd.Series([item["speed"] for item in data])
print(speeds.describe())
