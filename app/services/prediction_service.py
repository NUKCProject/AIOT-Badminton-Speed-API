from typing import List
import numpy as np
from fastapi import HTTPException
import logging

from app.models.schemas import SensorFrame
from app.core import config

logger = logging.getLogger(__name__)

def preprocess_sensor_data(sensor_data: List[SensorFrame]) -> np.ndarray:
    """
    預處理感測器數據
    
    Args:
        sensor_data: 30個時間幀的感測器數據
        
    Returns:
        預處理後的numpy數組，形狀為(1, 30, 6)
    """
    try:
        # 轉換為numpy數組
        data_array = []
        for frame in sensor_data:
            data_array.append([
                frame.ax, frame.ay, frame.az,
                frame.gx, frame.gy, frame.gz
            ])
        
        # 轉換為numpy數組
        X = np.array(data_array, dtype=np.float32)
        
        # 檢查數據形狀
        if X.shape != (30, 6):
            raise ValueError(f"數據形狀錯誤: 期望(30, 6)，實際{X.shape}")
        
        # 應用特徵標準化
        X_flat = X.reshape(-1, 6)
        X_scaled_flat = config.feature_scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(1, 30, 6)
        
        return X_scaled
        
    except Exception as e:
        logger.error(f"數據預處理失敗: {str(e)}")
        raise HTTPException(status_code=400, detail=f"數據預處理失敗: {str(e)}")

def validate_sensor_data_range(sensor_data: List[SensorFrame]):
    """
    驗證感測器數據是否在合理範圍內
    """
    for i, frame in enumerate(sensor_data):
        # 檢查加速度計數據 (通常在-20g到20g之間，假設單位為g)
        for axis, value in [('ax', frame.ax), ('ay', frame.ay), ('az', frame.az)]:
            if abs(value) > 50:  # 放寬範圍以適應高速運動
                logger.warning(f"幀{i}的{axis}值({value})可能異常")
        
        # 檢查陀螺儀數據 (通常在-2000到2000 dps之間)
        for axis, value in [('gx', frame.gx), ('gy', frame.gy), ('gz', frame.gz)]:
            if abs(value) > 3000:  # 放寬範圍
                logger.warning(f"幀{i}的{axis}值({value})可能異常")