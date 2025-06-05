import pandas as pd
import json

from pandas._libs.tslibs import timestamps

df = pd.read_csv("./data/badminton_speed.csv")

timestamps = ["hit_timestamp", "land_timestamp"]

for timestamp in timestamps:
    # 將 date_column 轉換為 datetime 型別
    df[timestamp] = pd.to_datetime(df[timestamp])

    # 轉換為 timestamp (毫秒)
    df[timestamp] = df[timestamp].astype('int64') // 10**6

print(df)

with open ("./data/smash_data.json", "r") as f:
    data = json.load(f)
    unlabeled_sequences = [item["waveform"] for item in data]

unlabeled_df_list = [pd.DataFrame(sequences) for sequences in unlabeled_sequences]

label = {}
for i, unlabeled_df in enumerate(unlabeled_df_list):
    mean_timestamp = unlabeled_df['ts'].mean()
    for y in range(len(df)):
        diff_hit = abs(mean_timestamp - df.iloc[y]['hit_timestamp'])
        diff_land = abs(mean_timestamp - df.iloc[y]['land_timestamp'])
        if (diff_hit <= 1200) or (diff_land <= 1200):
            label[i] = float(df.iloc[y]['speed'])


for i ,item in enumerate(data):
    item['speed'] = label.get(i, None)

with open("./data/smash_data_label.json", "w") as f:
    json.dump(data, f)