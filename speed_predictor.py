import numpy as np
import json
from typing import List, Dict
import pandas as pd

class IMU30SamplesPredictor:
    def __init__(self, imu_to_racket_distance=0.35, impact_coefficient=0.7):
        """
        基於30筆IMU資料的羽毛球速度預測器
        
        Parameters:
        -----------
        imu_to_racket_distance : float
            IMU到球拍擊球點的距離 (單位: 米)
        impact_coefficient : float
            撞擊係數，羽毛球約0.6-0.8
        """
        self.L = imu_to_racket_distance  # 距離參數
        self.e = impact_coefficient      # 撞擊係數
        self.dt = 0.02                  # 時間間隔 20ms
        self.calibration_factor = 1.2   # 經驗校正係數
        
    def parse_imu_data(self, imu_list: List[Dict]) -> Dict:
        """
        解析IMU資料列表
        
        Parameters:
        -----------
        imu_list : List[Dict]
            包含30筆IMU資料的列表
            
        Returns:
        --------
        parsed_data : Dict
            解析後的資料
        """
        if len(imu_list) != 30:
            raise ValueError(f"需要30筆資料，但收到 {len(imu_list)} 筆")
        
        # 提取時間序列和感測器資料
        timestamps = [data['ts'] for data in imu_list]
        ax = np.array([data['ax'] for data in imu_list])
        ay = np.array([data['ay'] for data in imu_list])
        az = np.array([data['az'] for data in imu_list])
        gx = np.array([data['gx'] for data in imu_list])
        gy = np.array([data['gy'] for data in imu_list])
        gz = np.array([data['gz'] for data in imu_list])
        
        return {
            'timestamps': timestamps,
            'ax': ax, 'ay': ay, 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz
        }
    
    def calculate_derivatives(self, data: Dict) -> Dict:
        """
        計算加速度和角速度的變化率
        
        Parameters:
        -----------
        data : Dict
            解析後的IMU資料
            
        Returns:
        --------
        derivatives : Dict
            包含各種導數和特徵值
        """
        # 計算加速度向量大小
        accel_magnitude = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
        
        # 計算角速度向量大小
        gyro_magnitude = np.sqrt(data['gx']**2 + data['gy']**2 + data['gz']**2)
        
        # 計算加速度變化率 (jerk)
        jerk_x = np.gradient(data['ax'], self.dt)
        jerk_y = np.gradient(data['ay'], self.dt)
        jerk_z = np.gradient(data['az'], self.dt)
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        # 計算角加速度
        angular_accel_x = np.gradient(data['gx'], self.dt)
        angular_accel_y = np.gradient(data['gy'], self.dt)
        angular_accel_z = np.gradient(data['gz'], self.dt)
        angular_accel_magnitude = np.sqrt(angular_accel_x**2 + angular_accel_y**2 + angular_accel_z**2)
        
        return {
            'accel_magnitude': accel_magnitude,
            'gyro_magnitude': gyro_magnitude,
            'jerk_magnitude': jerk_magnitude,
            'angular_accel_magnitude': angular_accel_magnitude
        }
    
    def find_impact_characteristics(self, data: Dict, derivatives: Dict) -> Dict:
        """
        找出擊球特徵
        
        Parameters:
        -----------
        data : Dict
            原始IMU資料
        derivatives : Dict
            導數資料
            
        Returns:
        --------
        impact_features : Dict
            擊球特徵參數
        """
        # 找出最大值及其索引
        max_accel_idx = np.argmax(derivatives['accel_magnitude'])
        max_gyro_idx = np.argmax(derivatives['gyro_magnitude'])
        max_jerk_idx = np.argmax(derivatives['jerk_magnitude'])
        max_angular_accel_idx = np.argmax(derivatives['angular_accel_magnitude'])
        
        # 擊球特徵值
        features = {
            'max_accel': derivatives['accel_magnitude'][max_accel_idx],
            'max_gyro': derivatives['gyro_magnitude'][max_gyro_idx],
            'max_jerk': derivatives['jerk_magnitude'][max_jerk_idx],
            'max_angular_accel': derivatives['angular_accel_magnitude'][max_angular_accel_idx],
            
            # 峰值時刻
            'accel_peak_time': max_accel_idx * self.dt,
            'gyro_peak_time': max_gyro_idx * self.dt,
            
            # 衝擊持續時間（從開始到峰值）
            'impact_duration': max_accel_idx * self.dt,
            
            # 平均值（整個時間窗口）
            'avg_accel': np.mean(derivatives['accel_magnitude']),
            'avg_gyro': np.mean(derivatives['gyro_magnitude']),
            
            # 標準差（反映變化劇烈程度）
            'std_accel': np.std(derivatives['accel_magnitude']),
            'std_gyro': np.std(derivatives['gyro_magnitude'])
        }
        
        return features
    
    def predict_ball_speed_formula(self, features: Dict) -> Dict:
        """
        核心數學公式：基於特徵計算球速
        
        這個公式結合了多個物理原理：
        1. 線性動量傳遞：基於最大加速度
        2. 角動量轉換：基於角速度和力臂
        3. 衝擊時間效應：基於jerk和持續時間
        4. 綜合能量考量：結合多個特徵的加權
        
        Parameters:
        -----------
        features : Dict
            擊球特徵參數
            
        Returns:
        --------
        prediction : Dict
            球速預測結果
        """
        # === 核心數學公式 ===
        
        # 1. 線性速度分量 (基於加速度積分)
        # V_linear = a_peak * t_impact + jerk_contribution
        v_linear = features['max_accel'] * features['impact_duration'] + \
                  features['max_jerk'] * (features['impact_duration']**2) * 0.5
        
        # 2. 角速度貢獻的線速度分量
        # V_angular = ω_max * L (力臂效應)
        v_angular = np.deg2rad(features['max_gyro']) * self.L
        
        # 3. 動態衝擊效應 (基於角加速度)
        # 反映球拍加速過程的貢獻
        v_dynamic = np.deg2rad(features['max_angular_accel']) * self.L * features['impact_duration']
        
        # 4. 綜合速度計算
        # 使用向量合成概念
        v_total = np.sqrt(v_linear**2 + v_angular**2 + v_dynamic**2)
        
        # 5. 經驗修正項 (基於統計特徵)
        # 考慮整個擊球過程的穩定性
        stability_factor = 1 + (features['std_accel'] / features['avg_accel']) * 0.1
        energy_factor = 1 + np.log(1 + features['avg_gyro'] / 100) * 0.2
        
        # 6. 最終球速預測公式
        predicted_racket_speed = v_total * stability_factor * energy_factor
        predicted_ball_speed = (predicted_racket_speed * self.e * self.calibration_factor / 5)
        
        # === 額外分析公式 ===
        
        # 擊球力度指標 (0-100)
        impact_intensity = min(100, (features['max_accel'] / 10) * 
                             (features['max_gyro'] / 200) * 100)
        
        # 擊球技術指標 (基於時序協調性)
        technique_score = 100 - abs(features['accel_peak_time'] - features['gyro_peak_time']) * 50
        technique_score = max(0, min(100, technique_score))
        
        return {
            'predicted_ball_speed_ms': predicted_ball_speed,
            'predicted_ball_speed_kmh': predicted_ball_speed * 3.6,
            'racket_tip_speed_ms': predicted_racket_speed,
            'racket_tip_speed_kmh': predicted_racket_speed * 3.6,
            
            # 分解分析
            'linear_component_ms': v_linear,
            'angular_component_ms': v_angular,
            'dynamic_component_ms': v_dynamic,
            
            # 擊球分析
            'impact_intensity': impact_intensity,
            'technique_score': technique_score,
            'stability_factor': stability_factor,
            'energy_factor': energy_factor,
            
            # 物理特徵
            'impact_duration_ms': features['impact_duration'] * 1000,
            'peak_acceleration_g': features['max_accel'],
            'peak_angular_velocity_dps': features['max_gyro']
        }
    
    def analyze_30_samples(self, imu_list: List[Dict]) -> Dict:
        """
        完整分析30筆IMU資料並預測球速
        
        Parameters:
        -----------
        imu_list : List[Dict]
            30筆連續的IMU資料
            
        Returns:
        --------
        result : Dict
            完整的分析結果
        """
        try:
            # 1. 解析資料
            data = self.parse_imu_data(imu_list)
            
            # 2. 計算導數
            derivatives = self.calculate_derivatives(data)
            
            # 3. 提取擊球特徵
            features = self.find_impact_characteristics(data, derivatives)
            
            # 4. 預測球速
            prediction = self.predict_ball_speed_formula(features)
            
            # 5. 組合結果
            result = {
                'success': True,
                'prediction': prediction,
                'features': features,
                'time_window_ms': 30 * 20,  # 30筆 × 20ms
                'sampling_rate_hz': 50      # 1000ms / 20ms
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }
    
    def calibrate_with_real_data(self, measured_speeds: List[float], 
                                predicted_speeds: List[float]):
        """
        使用實測資料校正預測係數
        
        Parameters:
        -----------
        measured_speeds : List[float]
            實測球速 (km/h)
        predicted_speeds : List[float]
            預測球速 (km/h)
        """
        if len(measured_speeds) != len(predicted_speeds):
            raise ValueError("實測值和預測值數量必須相同")
        
        measured = np.array(measured_speeds)
        predicted = np.array(predicted_speeds)
        
        # 線性回歸校正
        optimal_factor = np.sum(measured * predicted) / np.sum(predicted**2)
        self.calibration_factor = optimal_factor
        
        # 計算校正後的準確度
        corrected_predictions = predicted * optimal_factor
        mse = np.mean((measured - corrected_predictions)**2)
        rmse = np.sqrt(mse)
        
        print(f"校正完成！")
        print(f"新校正係數: {self.calibration_factor:.3f}")
        print(f"預測誤差 (RMSE): {rmse:.2f} km/h")
        
        return {
            'calibration_factor': self.calibration_factor,
            'rmse': rmse,
            'accuracy': 100 - (rmse / np.mean(measured) * 100)
        }


# === 使用示例 ===
def example_usage():
    """使用示例"""
    
    # 建立預測器
    predictor = IMU30SamplesPredictor(
        imu_to_racket_distance=0.35,  # 35cm
        impact_coefficient=0.7
    )

    with open("./data/smash_data_real_label.json", "r") as f:
        data = json.load(f)

    speeds = []
    for item in data:
        sample_data = item["waveform"]

        # 分析並預測
        result = predictor.analyze_30_samples(sample_data)
        
        if result['success']:
            pred = result['prediction']
            speeds.append(pred['predicted_ball_speed_kmh'])
        
        if item["speed"] == None:
            item["speed"] = pred['predicted_ball_speed_kmh']

    print(pd.Series(speeds).describe())

    with open("./data/smash_data_speed_predictor.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    example_usage()