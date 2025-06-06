import pandas as pd
import numpy as np
import json

# 假设 data_list 是一个包含所有动作数据的列表
# 示例数据结构：
# data_list = [
#     {
#         "waveform": [...],
#         "speed": 178.99520949676167
#     },
#     ...
# ]

with open("./data/smash_data_speed_predictor.json", "r") as f:
    data = json.load(f)
data_list = [{"waveform":item['waveform'], "speed": item["speed"]} for item in data if item["speed"]]
print(len(data_list))

# 1. 提取特征并构建特征矩阵
features = []

for item in data_list:
    waveform = item['waveform']
    
    # 提取加速度计和陀螺仪数据
    ax = [abs(point['ax']) for point in waveform]
    ay = [abs(point['ay']) for point in waveform]
    az = [abs(point['az']) for point in waveform]
    gx = [abs(point['gx']) for point in waveform]
    gy = [abs(point['gy']) for point in waveform]
    gz = [abs(point['gz']) for point in waveform]

    # 提取统计特征（均值、标准差、最大值、最小值）
    feature = {
        'ax_mean': np.mean(ax),
        'ax_std': np.std(ax),
        'ax_max': np.max(ax),
        'ax_min': np.min(ax),
        
        'ay_mean': np.mean(ay),
        'ay_std': np.std(ay),
        'ay_max': np.max(ay),
        'ay_min': np.min(ay),
        
        'az_mean': np.mean(az),
        'az_std': np.std(az),
        'az_max': np.max(az),
        'az_min': np.min(az),
        
        'gx_mean': np.mean(gx),
        'gx_std': np.std(gx),
        'gx_max': np.max(gx),
        'gx_min': np.min(gx),
        
        'gy_mean': np.mean(gy),
        'gy_std': np.std(gy),
        'gy_max': np.max(gy),
        'gy_min': np.min(gy),
        
        'gz_mean': np.mean(gz),
        'gz_std': np.std(gz),
        'gz_max': np.max(gz),
        'gz_min': np.min(gz),
        
        'total_mean_abs': np.mean(ax) + np.mean(ay) + np.mean(az) + np.mean(gx) + np.mean(gy) + np.mean(gz),
        'speed': item['speed']
    }
    
    features.append(feature)

# 2. 构建 DataFrame
df_features = pd.DataFrame(features)

# 3. 计算与 speed 的相关性
correlation = df_features.corr()['speed']

# 4. 打印相关性
print("特征与 speed 的相关性（皮尔逊）：")
print(correlation)

# 5. 可视化（可选）
import seaborn as sns
import matplotlib.pyplot as plt

# 构建完整的相关性矩阵
corr_matrix = df_features.corr()

# 可视化热图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of correlation between features and speed")
plt.show()

# 6. 绘制散布图
fig, axes = plt.subplots(2, 3, figsize=(30, 12))
axes = axes.flatten()

features_to_plot = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean']

for i, feature in enumerate(features_to_plot):
    sns.scatterplot(x=feature, y='speed', data=df_features, ax=axes[i])
    axes[i].set_title(f'Scatter plot of {feature} vs Speed')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Speed')

# Hide any unused subplots
for j in range(len(features_to_plot), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.tight_layout()
plt.show()

# 7. 绘制六轴平均总和与速度的散布图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_mean_abs', y='speed', data=df_features)
plt.title('Scatter plot of Total Mean Abs vs Speed')
plt.xlabel('Total Mean Abs')
plt.ylabel('Speed')
plt.show()