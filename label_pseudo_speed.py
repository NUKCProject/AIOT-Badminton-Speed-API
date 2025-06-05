import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json

class IMUSequencePseudoLabelGenerator:
    def __init__(self, labeled_sequences):
        """
        初始化时序伪标签生成器
        :param labeled_sequences: 已标记数据列表，格式 [{"sequence": [...], "speed": float}, ...]
        """
        self.labeled_sequences = labeled_sequences
        self.mu_s = 138.88  # 基础球速均值 (km/h)
        self.sigma_s = 40.12  # 基础球速标准差
        self.alpha = 0.0  # 角速度-球速敏感系数
        self.beta = 0.25  # 挥拍稳定性系数
        self.g = 9.81  # 重力加速度
        self.fs = 100  # 采样频率 (Hz)，根据时间戳计算
        
        # 从标签数据校准模型参数
        self._calibrate_model()
        
    def _calculate_fs(self, sequence):
        """从时间戳计算采样频率"""
        timestamps = [frame['ts'] for frame in sequence]
        if len(timestamps) < 2:
            return 100  # 默认值
        
        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        return 1000 / avg_interval  # 转换为Hz
    
    def _calibrate_model(self):
        """从已标记数据校准模型参数"""
        # 提取特征和标签
        features = []
        speeds = []
        
        for data in self.labeled_sequences:
            sequence = data["sequence"]
            fs = self._calculate_fs(sequence)
            
            # 处理时序数据
            processed = self.process_sequence(sequence, fs)
            
            if processed is not None:  # 确保处理成功
                # 获取特征
                features.append([
                    processed['omega_eff'],
                    processed['snr'],
                    processed['stability']
                ])
                speeds.append(data["speed"])
        
        if len(features) < 3:
            print("警告：标记数据不足，使用默认参数")
            return
            
        # 拟合线性模型 v = k * omega + b
        model = LinearRegression()
        model.fit(np.array(features)[:, 0].reshape(-1, 1), np.array(speeds))
        self.alpha = model.coef_[0]  # 斜率 k
        
        # 更新基础分布参数
        self.mu_s = np.mean(speeds)
        self.sigma_s = np.std(speeds)
        
        print(f"模型校准完成: μ={self.mu_s:.1f} km/h, σ={self.sigma_s:.1f} km/h, α={self.alpha:.2f} m/rad")
    
    def _butter_lowpass(self, data, cutoff, fs, order=2):
        """低通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def _correct_bias(self, sequence, fs):
        """零偏校正"""
        # 使用前1/4序列作为静态段
        static_frames = sequence[:max(1, len(sequence)//4)]
        
        # 计算零偏
        gyro_bias = np.array([
            np.mean([f['gx'] for f in static_frames]),
            np.mean([f['gy'] for f in static_frames]),
            np.mean([f['gz'] for f in static_frames])
        ])
        
        accel_bias = np.array([
            np.mean([f['ax'] for f in static_frames]),
            np.mean([f['ay'] for f in static_frames]),
            np.mean([f['az'] for f in static_frames]) - self.g
        ])
        
        # 应用校正
        corrected = []
        for frame in sequence:
            corrected.append({
                'ts': frame['ts'],
                'ax': frame['ax'] - accel_bias[0],
                'ay': frame['ay'] - accel_bias[1],
                'az': frame['az'] - accel_bias[2],
                'gx': frame['gx'] - gyro_bias[0],
                'gy': frame['gy'] - gyro_bias[1],
                'gz': frame['gz'] - gyro_bias[2],
                'mic_level': frame['mic_level']
            })
            
        return corrected
    
    def _detect_impact(self, sequence, fs):
        """检测击球点"""
        # 方法1: 使用麦克风信号检测击球点
        mic_levels = np.array([f['mic_level'] for f in sequence])
        
        # 确保距离参数至少为1
        min_distance = max(1, int(0.05 * fs))
        
        try:
            # 寻找麦克风峰值
            peaks, _ = find_peaks(
                mic_levels, 
                height=np.max(mic_levels)*0.7, 
                distance=min_distance
            )
            
            if len(peaks) > 0:
                # 取最大峰值
                main_peak = peaks[np.argmax(mic_levels[peaks])]
                return main_peak
        except Exception as e:
            print(f"麦克风峰值检测错误: {e}")
        
        # 方法2: 使用加速度信号作为备选
        accel_mag = np.array([np.sqrt(f['ax']**2 + f['ay']**2 + f['az']**2) for f in sequence])
        
        try:
            peaks, _ = find_peaks(
                accel_mag, 
                height=np.max(accel_mag)*0.8, 
                distance=min_distance
            )
            
            if len(peaks) > 0:
                # 取最大峰值
                main_peak = peaks[np.argmax(accel_mag[peaks])]
                return main_peak
        except Exception as e:
            print(f"加速度峰值检测错误: {e}")
        
        # 方法3: 使用角速度作为最后手段
        gyro_y = np.abs([f['gy'] for f in sequence])
        return np.argmax(gyro_y)
    
    def process_sequence(self, sequence, fs=None):
        """处理时序数据并提取特征"""
        if fs is None:
            fs = self._calculate_fs(sequence)
        
        # 1. 零偏校正
        corrected = self._correct_bias(sequence, fs)
        
        # 2. 检测击球点
        impact_idx = self._detect_impact(corrected, fs)
        
        # 3. 提取击球点附近数据
        impact_frame = corrected[impact_idx]
        
        # 4. 计算有效角速度 (取主要旋转轴)
        # 羽毛球拍主要旋转轴通常是 gy (绕垂直轴旋转)
        omega_eff = abs(impact_frame['gy'])
        
        # 5. 计算信噪比 (SNR)
        # 静态段 (击球前)
        static_start = max(0, impact_idx - int(0.2*fs))
        static_end = max(0, impact_idx - int(0.05*fs))
        
        # 确保静态段有足够的点
        if static_end - static_start < 3:
            static_start = max(0, impact_idx - 10)
            static_end = max(0, impact_idx - 1)
        
        static_gy = [f['gy'] for f in corrected[static_start:static_end]]
        
        # 击球段 (击球点附近)
        impact_start = max(0, impact_idx - int(0.01*fs))
        impact_end = min(len(corrected), impact_idx + int(0.01*fs))
        impact_gy = [f['gy'] for f in corrected[impact_start:impact_end]]
        
        if len(static_gy) > 2 and len(impact_gy) > 0:
            static_std = np.std(static_gy)
            impact_peak = np.max(np.abs(impact_gy))
            snr = impact_peak / (static_std + 1e-6)  # 避免除零
        else:
            snr = 100  # 默认高信噪比
        
        # 6. 计算挥拍稳定性 (使用角速度的样本熵)
        gyro_y = [f['gy'] for f in corrected]
        
        # 简化样本熵计算
        def sample_entropy(data, m=2, r=0.2):
            n = len(data)
            if n <= m:
                return 0
                
            # 计算m维向量
            vectors = [data[i:i+m] for i in range(n - m)]
            
            # 计算距离
            count = 0
            for i, v1 in enumerate(vectors):
                for j, v2 in enumerate(vectors):
                    if i != j and max(abs(a - b) for a, b in zip(v1, v2)) < r:
                        count += 1
                        
            return count / (n * (n - 1))
        
        stability = sample_entropy(gyro_y, m=2, r=0.2)
        
        return {
            'omega_eff': omega_eff,
            'snr': snr,
            'stability': stability,
            'impact_idx': impact_idx
        }
    
    def generate_pseudo_label(self, sequence):
        """
        生成加权伪标签
        :param sequence: IMU时序数据
        :return: 伪标签球速 (km/h)
        """
        # 计算采样频率
        fs = self._calculate_fs(sequence)
        
        # 处理时序数据
        features = self.process_sequence(sequence, fs)
        
        if features is None:
            print("无法处理该序列，使用默认值")
            return self.mu_s
        
        # print(f"特征提取: ω_eff={features['omega_eff']:.1f} rad/s, SNR={features['snr']:.1f}, Stability={features['stability']:.3f}")
        
        # 3. 计算调整参数
        # 均值调整: μ_adj = μ_s + α*(ω_eff - ω_bar)
        # 注意：这里ω_bar是平均有效角速度，需要从校准数据中获取
        # 为简化，我们使用固定基准值
        omega_bar = 100.0  # 典型羽毛球杀球角速度基准值
        mu_adj = self.mu_s + self.alpha * (features['omega_eff'] - omega_bar)
        
        # 标准差调整: σ_adj = σ_s * β/(stability + γ)
        gamma = 0.1  # 防止除零
        sigma_adj = self.sigma_s * self.beta / (features['stability'] + gamma)
        
        # SNR加权: 低信噪比时扩大分布范围
        if features['snr'] < 10.0:
            sigma_adj *= 1.8
            print("低SNR警告: 扩大分布范围")
        
        # print(f"调整参数: μ_adj={mu_adj:.1f}, σ_adj={sigma_adj:.1f}")
        
        # 4. 生成伪标签
        v_pseudo = np.random.normal(mu_adj, sigma_adj)
        
        # # 5. 物理约束 (羽毛球杀球合理范围)
        # v_min, v_max = 60, 220
        # v_pseudo = np.clip(v_pseudo, v_min, v_max)
        
        return v_pseudo

    def visualize_sequence(self, sequence, title="IMU Sequence"):
        """可视化时序数据"""
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        
        # 加速度数据
        ts = [f['ts'] - sequence[0]['ts'] for f in sequence]
        axs[0].plot(ts, [f['ax'] for f in sequence], 'r-', label='ax')
        axs[0].plot(ts, [f['ay'] for f in sequence], 'g-', label='ay')
        axs[0].plot(ts, [f['az'] for f in sequence], 'b-', label='az')
        axs[0].set_ylabel('Acceleration (m/s²)')
        axs[0].legend()
        axs[0].grid(True)
        
        # 角速度数据
        axs[1].plot(ts, [f['gx'] for f in sequence], 'r-', label='gx')
        axs[1].plot(ts, [f['gy'] for f in sequence], 'g-', label='gy')
        axs[1].plot(ts, [f['gz'] for f in sequence], 'b-', label='gz')
        axs[1].set_ylabel('Gyroscope (rad/s)')
        axs[1].legend()
        axs[1].grid(True)
        
        # 麦克风信号
        axs[2].plot(ts, [f['mic_level'] for f in sequence], 'm-', label='Mic Level')
        axs[2].set_ylabel('Mic Level')
        axs[2].legend()
        axs[2].grid(True)
        
        # 加速度幅值
        accel_mag = [np.sqrt(f['ax']**2 + f['ay']**2 + f['az']**2) for f in sequence]
        axs[3].plot(ts, accel_mag, 'c-', label='Accel Magnitude')
        axs[3].set_xlabel('Time (ms)')
        axs[3].set_ylabel('Accel Magnitude')
        axs[3].legend()
        axs[3].grid(True)
        
        # 标记击球点
        fs = self._calculate_fs(sequence)
        features = self.process_sequence(sequence, fs)
        if features:
            impact_time = ts[features['impact_idx']]
            for ax in axs:
                ax.axvline(x=impact_time, color='k', linestyle='--', alpha=0.7)
                ax.annotate('Impact', 
                            xy=(impact_time, ax.get_ylim()[1]*0.9), 
                            xytext=(impact_time, ax.get_ylim()[1]*0.9),
                            arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # # 加载数据（这里使用您提供的示例数据）
    # sequence = [
    #     {'ts': 1748565104764, 'ax': -0.03660000115633011, 'ay': 1.1141040325164795, 'az': -0.045871999114751816, 'gx': 73.5, 'gy': 29.75, 'gz': -167.5800018310547, 'mic_level': 74, 'mic_peak': 26638},
    #     # ... 其他数据点 ...
    #     {'ts': 1748565106019, 'ax': 1.4044640064239502, 'ay': 0.5328959822654724, 'az': 0.28596800565719604, 'gx': 349.8599853515625, 'gy': 251.86000061035156, 'gz': 78.81999969482422, 'mic_level': 156, 'mic_peak': 26638}
    # ]
    
    # # 模拟已标记数据 (实际需真实数据)
    # labeled_sequences = [
    #     {"sequence": sequence, "speed": 305},  # 假设这是已标记数据
    #     # 添加更多已标记序列...
    # ]

    with open("./data/smash_data_real_label.json", "r") as f:
        data = json.load(f)

    labeled_sequences = [{"sequence":item['waveform'], "speed": item["speed"]} for item in data if item["speed"]]

    # 初始化生成器 (使用已标记数据校准)
    generator = IMUSequencePseudoLabelGenerator(labeled_sequences)
    
    # 为未标记数据生成伪标签
    for i, item in enumerate(data):
        if(item["speed"] == None):
            sequence = item["waveform"]
            pseudo_speed = generator.generate_pseudo_label(sequence)
            # print(f"生成的伪标签球速: {pseudo_speed:.1f} km/h\n")
            data[i]['speed'] = float(pseudo_speed)
    
    with open("./data/smash_data_pseudo_label.json", "w") as f:
        json.dump(data, f)