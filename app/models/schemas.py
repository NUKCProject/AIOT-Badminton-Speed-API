from pydantic import BaseModel, Field, validator
from typing import List

class SensorFrame(BaseModel):
    """單個時間幀的感測器數據"""
    ax: float = Field(..., description="加速度計X軸數據")
    ay: float = Field(..., description="加速度計Y軸數據") 
    az: float = Field(..., description="加速度計Z軸數據")
    gx: float = Field(..., description="陀螺儀X軸數據")
    gy: float = Field(..., description="陀螺儀Y軸數據")
    gz: float = Field(..., description="陀螺儀Z軸數據")

class PredictionRequest(BaseModel):
    """預測請求數據模型"""
    sensor_data: List[SensorFrame] = Field(..., description="感測器數據序列")
    
    @validator('sensor_data')
    def validate_sensor_data_length(cls, v):
        if len(v) != 30:
            raise ValueError(f'sensor_data必須包含恰好30個時間幀，當前為{len(v)}個')
        return v

class PredictionResponse(BaseModel):
    """預測響應數據模型"""
    predicted_speed: float = Field(..., description="預測的球速 (km/h)")
    confidence_info: dict = Field(..., description="預測信心資訊")

class HealthResponse(BaseModel):
    """健康檢查響應"""
    status: str
    message: str
    model_loaded: bool