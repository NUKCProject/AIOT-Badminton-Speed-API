import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 設置matplotlib支持中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設置隨機種子確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 1. 從JSON文件加載數據 -------------------------------------------------
def load_data_from_json(file_path):
   with open(file_path, 'r') as f:
       data = json.load(f)
   
   X = []
   y = []
   
   for sample in data:
       # 提取球速
       y.append(sample['speed'])
       
       # 提取波形數據
       waveform_data = []
       for frame in sample['waveform']:
           # 按順序提取6個特徵：ax, ay, az, gx, gy, gz
           waveform_data.append([
               frame['ax'],
               frame['ay'],
               frame['az'],
               frame['gx'],
               frame['gy'],
               frame['gz']
           ])
       
       # 確保每個樣本有30個時間步
       if len(waveform_data) != 30:
           print(f"警告: 樣本有 {len(waveform_data)} 幀數據，但需要30幀。跳過此樣本。")
           continue
           
       X.append(waveform_data)
   
   return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 2. 特徵工程函數 -------------------------------------------------------
def extract_statistical_features(X):
   """從時間序列中提取統計特徵"""
   features = []
   for sample in X:
       sample_features = []
       for feature_idx in range(sample.shape[1]):  # 對每個感測器特徵
           feature_data = sample[:, feature_idx]
           # 提取統計特徵
           sample_features.extend([
               np.mean(feature_data),
               np.std(feature_data),
               np.max(feature_data),
               np.min(feature_data),
               np.max(feature_data) - np.min(feature_data),  # 範圍
               np.percentile(feature_data, 75) - np.percentile(feature_data, 25)  # IQR
           ])
       features.append(sample_features)
   return np.array(features)

# 替換為您的JSON文件路徑
json_file_path = './data/smash_data_speed_predictor.json'
X, y = load_data_from_json(json_file_path)

print(f"加載數據: {len(X)} 個樣本")
print(f"特徵形狀: {X.shape}")
print(f"目標值範圍: {y.min():.2f} - {y.max():.2f} km/h")
print(f"目標值分佈: 平均={y.mean():.2f}, 標準差={y.std():.2f}")

# 3. 數據預處理 ------------------------------------------------------
# 方法1: 標準化時間序列特徵
all_features = X.reshape(-1, 6)
feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(all_features)
X_scaled = scaled_features.reshape(X.shape[0], X.shape[1], X.shape[2])

# 方法2: 標準化目標值
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

print(f"標準化後目標值範圍: {y_scaled.min():.3f} - {y_scaled.max():.3f}")

# 分割數據集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\n數據集形狀:")
print(f"訓練集: {X_train.shape} ({len(y_train)} 樣本)")
print(f"驗證集: {X_val.shape} ({len(y_val)} 樣本)")
print(f"測試集: {X_test.shape} ({len(y_test)} 樣本)")

# 4. 構建改進的模型 ---------------------------------------------------
def create_improved_model():
   model = Sequential([
       # 輕量化的CNN層
       Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, 6)),
       BatchNormalization(),
       MaxPooling1D(pool_size=2),
       Dropout(0.2),
       
       # 簡化的LSTM層
       LSTM(64, return_sequences=True, recurrent_dropout=0.1),
       BatchNormalization(),
       Dropout(0.2),
       
       LSTM(32, return_sequences=False, recurrent_dropout=0.1),
       BatchNormalization(),
       
       # 全連接層
       Dense(32, activation='relu'),
       Dropout(0.2),
       Dense(16, activation='relu'),
       Dense(1, activation='linear')
   ])
   return model

# 5. 創建並編譯模型 ---------------------------------------------------
model = create_improved_model()

# 使用較大的學習率
optimizer = Adam(learning_rate=0.001)  # 增加學習率
model.compile(optimizer=optimizer,
             loss='mse',
             metrics=['mae'])

model.summary()

# 6. 設置回調函數 -----------------------------------------------------
early_stop = EarlyStopping(
   monitor='val_loss',
   patience=30,
   restore_best_weights=True,
   verbose=1
)

# 學習率調度器
lr_scheduler = ReduceLROnPlateau(
   monitor='val_loss',
   factor=0.5,
   patience=10,
   min_lr=1e-6,
   verbose=1
)

# 7. 訓練模型 -------------------------------------------------------
print("\n開始訓練模型...")
history = model.fit(
   X_train, y_train,
   epochs=200,
   batch_size=64,  # 增加批次大小
   validation_data=(X_val, y_val),
   callbacks=[early_stop, lr_scheduler],
   verbose=1
)

# 8. 評估模型 -------------------------------------------------------
# 預測測試集
y_pred_scaled = model.predict(X_test).flatten()

# 將預測結果轉換回原始尺度
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 計算評估指標
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"\n===== 模型評估結果 =====")
print(f"平均絶對誤差 (MAE): {mae:.2f} km/h")
print(f"均方根誤差 (RMSE): {rmse:.2f} km/h")
print(f"決定係數 (R²): {r2:.4f}")

print(f"\n預測值統計:")
print(f"  範圍: {y_pred.min():.2f} - {y_pred.max():.2f} km/h")
print(f"  平均: {y_pred.mean():.2f} km/h")
print(f"  標準差: {y_pred.std():.2f} km/h")
print(f"  唯一值數量: {len(np.unique(np.round(y_pred, 1)))}")

print(f"\n真實值統計:")
print(f"  範圍: {y_test_original.min():.2f} - {y_test_original.max():.2f} km/h")
print(f"  平均: {y_test_original.mean():.2f} km/h")
print(f"  標準差: {y_test_original.std():.2f} km/h")

# 9. 可視化結果 -----------------------------------------------------
# 訓練歷史
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.title('模型損失')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='訓練MAE')
plt.plot(history.history['val_mae'], label='驗證MAE')
plt.title('平均絶對誤差')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# 真實值 vs 預測值
plt.subplot(1, 3, 3)
plt.scatter(y_test_original, y_pred, alpha=0.6, s=20)
plt.plot([y_test_original.min(), y_test_original.max()],
        [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('真實速度 (km/h)')
plt.ylabel('預測速度 (km/h)')
plt.title('真實值 vs 預測值')
plt.grid(True)

# 添加R²和RMSE到圖上
plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.1f}',
        transform=plt.gca().transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('./ml_model/model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 誤差分析 ------------------------------------------------------
errors = np.abs(y_pred - y_test_original)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('絶對誤差 (km/h)')
plt.ylabel('頻次')
plt.title('預測誤差分佈')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_original, errors, alpha=0.6)
plt.xlabel('真實速度 (km/h)')
plt.ylabel('絶對誤差 (km/h)')
plt.title('誤差 vs 真實值')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./ml_model/error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 保存模型和預處理器 --------------------------------------------
model.save('./ml_model/badminton_speed_predictor.h5')
print("\n模型已保存為 './ml_model/badminton_speed_predictor.h5'")

import joblib
joblib.dump(feature_scaler, './ml_model/feature_scaler.pkl')
joblib.dump(target_scaler, './ml_model/target_scaler.pkl')
print("預處理器已保存")

# 12. 預測示例 ------------------------------------------------------
print(f"\n===== 預測示例 =====")
for i in range(5):
   idx = np.random.randint(0, len(X_test))
   sample_input = X_test[idx][np.newaxis, ...]
   pred_scaled = model.predict(sample_input, verbose=0)[0][0]
   pred_original = target_scaler.inverse_transform([[pred_scaled]])[0][0]
   actual_original = target_scaler.inverse_transform([[y_test[idx]]])[0][0]
   
   print(f"樣本 {i+1}:")
   print(f"  真實: {actual_original:.1f} km/h")
   print(f"  預測: {pred_original:.1f} km/h")
   print(f"  誤差: {abs(pred_original - actual_original):.1f} km/h")

# 13. 模型複雜度分析 ------------------------------------------------
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"\n===== 模型資訊 =====")
print(f"總參數數量: {total_params:,}")
print(f"可訓練參數: {trainable_params:,}")
print(f"訓練耗時: {len(history.history['loss'])} epochs")