import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy, weibull_min
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

class IMUSequencePseudoLabelGenerator:
    def __init__(self, config):
        self.weibull_c = config.get('weibull_c', 3.5)  # 形状参数
        self.weibull_loc = config.get('weibull_loc', 50.0) # 位置参数
        self.weibull_scale = config.get('weibull_scale', 200.0) # 尺度参数
        self.alpha = config.get('alpha', 0.0) # 角速度对伪标签的影响因子
        self.beta = config.get('beta', 0.0)  # 稳定性对伪标签的影响因子
        self.min_speed = config.get('min_speed', 0.0)
        self.max_speed = config.get('max_speed', 1000.0)
        self.g = 9.81  # 重力加速度
        self.fs = 100  # 采样频率 (Hz)，根据时间戳计算
        
    def _calculate_fs(self, sequence):
        """从时间戳计算采样频率"""
        timestamps = [frame['ts'] for frame in sequence]
        if len(timestamps) < 2:
            return 100  # 默认值
        
        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        return 1000 / avg_interval  # 转换为Hz
    
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
    
    def generate_pseudo_label(self, features):
        """
        生成加权伪标签
        :param features: 提取的特征数据 (字典)
        :return: 伪标签球速 (km/h)
        """
        # 特征已在外部处理并传入，无需再次计算或处理 (Added for sync check)
        
        # print(f"特征提取: ω_eff={features['omega_eff']:.1f} rad/s, SNR={features['snr']:.1f}, Stability={features['stability']:.3f}")
        
        # 使用Weibull分布生成伪标签
        # Weibull分布的参数c (形状), loc (位置), scale (尺度)
        # 根据角速度和稳定性动态调整Weibull分布的尺度参数
        # 确保alpha和beta参数在校准后被正确设置
        # 这里的features['omega_eff']对应avg_angular_speed，features['stability']对应stability_score
        scale_adj = self.weibull_scale * (1 + self.alpha * features['omega_eff']) * (1 - self.beta * features['stability'])
        # 确保调整后的尺度参数为正值
        scale_adj = np.maximum(0.1, scale_adj) # 避免尺度参数过小或为负
        
        v_pseudo = weibull_min.rvs(self.weibull_c, loc=self.weibull_loc, scale=scale_adj, size=1)[0]
        
        # 确保伪标签不为负数 (Weibull分布自然非负，但保留此行以防万一或未来修改)
        v_pseudo = np.clip(v_pseudo, 0, None)
        
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

    def load_pseudo_labels(self, input_path):
        return np.load(input_path)

    def calibrate_weibull_params(self, true_speeds, features_data):
        # 定义一个函数，用于拟合Weibull分布的尺度参数
        def weibull_scale_func(X, alpha, beta):
            omega_eff, stability = X
            return self.weibull_scale * (1 + alpha * omega_eff) * (1 - beta * stability)

        # 提取特征数据
        omega_eff_data = np.array([f['omega_eff'] for f in features_data])
        stability_data = np.array([f['stability'] for f in features_data])

        # 假设true_speeds是与features_data对应的真实速度值
        # 我们需要从true_speeds中提取出对应的weibull_scale
        # 这里简化处理，直接使用true_speeds作为目标值，实际应用中可能需要更复杂的统计方法
        # 例如，对每个(omega_eff, stability)组合下的true_speeds进行Weibull拟合，提取其scale参数
        # 为了演示，我们暂时假设true_speeds的均值可以作为scale的近似
        # 实际校准时，需要根据true_speeds的分布来估计weibull_scale
        # 这里需要根据实际的true_speeds和features_data来构建拟合的目标值
        # 假设我们已经有了每个样本对应的“目标尺度”target_scales
        # target_scales = [weibull_min.fit(speeds_for_this_feature_combo)[2] for speeds_for_this_feature_combo in grouped_true_speeds]
        
        # 暂时使用一个简化的目标值，实际应用中需要根据真实数据的分布来确定
        # 比如，如果true_speeds是每个样本的速度，我们可以尝试拟合一个整体的weibull_scale
        # 或者，如果true_speeds是分组后的速度，可以计算每组的weibull_scale
        # 这里为了让curve_fit能运行，我们假设true_speeds的某个统计量与scale_adj相关
        # 这是一个简化的例子，实际校准需要更严谨的统计方法
        target_scales = true_speeds # 这是一个占位符，需要根据实际数据来计算

        # 使用curve_fit拟合alpha和beta
        # 初始猜测值可以根据经验设定
        initial_guess = [self.alpha, self.beta]
        try:
            # 确保X是二维数组，第一列是omega_eff，第二列是stability
            popt, pcov = curve_fit(weibull_scale_func, (omega_eff_data, stability_data), target_scales, p0=initial_guess)
            self.alpha = popt[0]
            self.beta = popt[1]
            print(f"Weibull参数校准完成：alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        except RuntimeError as e:
            print(f"Weibull参数校准失败: {e}")
            print("请检查输入数据和初始猜测值。")

    def plot_pseudo_label_distribution(self, pseudo_labels, title="伪标签速度分布"):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(pseudo_labels, kde=True, bins=30)
        plt.title(f'{title} - 直方图')
        plt.xlabel('速度 (km/h)')
        plt.ylabel('频数')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=pseudo_labels)
        plt.title(f'{title} - 箱线图')
        plt.ylabel('速度 (km/h)')
        
        plt.tight_layout()
        plt.show()

# ====================== 使用示例 ======================
if __name__ == "__main__":
    with open("./data/smash_data_real_label.json", "r") as f:
        data = json.load(f)

    labeled_sequences = [{"sequence":item['waveform'], "speed": item["speed"]} for item in data if item["speed"]]

    # 定义配置字典
    config = {
        'weibull_c': 3.5,
        'weibull_loc': 50.0,
        'weibull_scale': 200.0,
        'alpha': 0.0,
        'beta': 0.0,
        'min_speed': 0.0,
        'max_speed': 1000.0
    }

    # 初始化生成器
    generator = IMUSequencePseudoLabelGenerator(config)

    # 收集用于校准的数据
    true_speeds = []
    features_for_calibration = []

    for item in labeled_sequences:
        sequence = item["sequence"]
        true_speed = item["speed"]
        processed_features = generator.process_sequence(sequence)
        if processed_features is not None:
            true_speeds.append(true_speed)
            features_for_calibration.append(processed_features)

    # 校准Weibull分布的alpha和beta参数
    if len(true_speeds) > 0:
        generator.calibrate_weibull_params(np.array(true_speeds), features_for_calibration)

    # 为所有数据生成伪标签 (包括已标记和未标记)
    for i, item in enumerate(data):
        sequence = item["waveform"]
        processed_features = generator.process_sequence(sequence)
        
        if processed_features is not None:
            # 如果是未标记数据，或者需要重新生成伪标签
            if item["speed"] is None:
                pseudo_speed = generator.generate_pseudo_label(processed_features)
                data[i]['speed'] = float(pseudo_speed)
        else:
            print(f"警告: 无法处理序列 {i}，跳过伪标签生成。")
            # 可以选择给一个默认值或者标记为无效
            # data[i]['speed'] = -1.0 # 示例：标记为无效值
    
    with open("./data/smash_data_pseudo_label.json", "w") as f:
        json.dump(data, f)