import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/badminton_speed.csv")
print("Real Label Speeds Description:")
print(df["speed"].describe())

with open("./data/smash_data_speed_predictor.json", "r") as f:
    data = json.load(f)

speeds = pd.Series([item["speed"] for item in data])
print("Pseudo Label Speeds Description:")
print(speeds.describe())

# 可视化伪标签速度分布
plt.figure(figsize=(12, 6))

# 直方图
plt.subplot(1, 2, 1)
sns.histplot(speeds, kde=True, bins=30)
plt.title('Pseudo Label Speeds Distribution (Histogram)')
plt.xlabel('Speed (km/h)')
plt.ylabel('Frequency')

# 箱线图
plt.subplot(1, 2, 2)
sns.boxplot(y=speeds)
plt.title('Pseudo Label Speeds Distribution (Box Plot)')
plt.ylabel('Speed (km/h)')

plt.tight_layout()
plt.show()


