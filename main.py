from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
import tensorflow as tf
import joblib
import logging
from contextlib import asynccontextmanager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局變量存儲模型和預處理器
model = None
feature_scaler = None
target_scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程序生命週期管理"""
    # 啟動時加載模型
    global model, feature_scaler, target_scaler
    try:
        logger.info("正在加載模型和預處理器...")
        
        # 加載訓練好的模型
        model = tf.keras.models.load_model('badminton_speed_predictor.h5', compile=False)
        logger.info("模型加載成功")
        
        # 加載預處理器
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        logger.info("預處理器加載成功")
        
        logger.info("服務初始化完成")
        
    except Exception as e:
        logger.error(f"模型加載失敗: {str(e)}")
        raise e
    
    yield
    
    # 關閉時清理資源
    logger.info("正在關閉服務...")

# 創建FastAPI應用
app = FastAPI(
    title="羽毛球速度預測API",
    description="基於感測器數據預測羽毛球速度的API服務",
    version="1.0.0",
    lifespan=lifespan
)

# 定義數據模型
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
        X_scaled_flat = feature_scaler.transform(X_flat)
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

@app.get("/", response_model=dict)
async def root():
    """根路徑"""
    return {
        "message": "羽毛球速度預測API服務",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict_speed",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    model_loaded = model is not None and feature_scaler is not None and target_scaler is not None
    
    if model_loaded:
        return HealthResponse(
            status="healthy",
            message="服務運行正常，模型已加載",
            model_loaded=True
        )
    else:
        return HealthResponse(
            status="unhealthy", 
            message="模型尚未加載或加載失敗",
            model_loaded=False
        )

@app.post("/predict_speed", response_model=PredictionResponse)
async def predict_speed(request: PredictionRequest):
    """
    預測羽毛球速度
    
    Args:
        request: 包含30個時間幀感測器數據的請求
        
    Returns:
        預測的球速和相關信息
    """
    try:
        # 檢查模型是否已加載
        if model is None or feature_scaler is None or target_scaler is None:
            raise HTTPException(
                status_code=503, 
                detail="模型尚未加載，請稍後重試"
            )
        
        logger.info(f"收到預測請求，數據長度: {len(request.sensor_data)}")
        
        # 驗證數據範圍（可選，記錄警告但不阻止預測）
        validate_sensor_data_range(request.sensor_data)
        
        # 預處理數據
        X_processed = preprocess_sensor_data(request.sensor_data)
        
        # 進行預測
        logger.info("開始進行速度預測...")
        prediction_scaled = model.predict(X_processed, verbose=0)[0][0]
        
        # 將預測結果轉換回原始尺度
        prediction_original = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
        
        # 計算置信度信息（基於訓練數據的統計信息）
        confidence_info = {
            "prediction_in_training_range": bool(50.0 <= prediction_original <= 400.0),  # 假設訓練範圍
            "model_certainty": "medium",  # 可以根據實際需求計算更精確的置信度
            "data_quality": "normal"
        }
        
        logger.info(f"預測完成，速度: {prediction_original:.2f} km/h")
        
        return PredictionResponse(
            predicted_speed=round(prediction_original, 2),
            confidence_info=confidence_info
        )
        
    except HTTPException as e:
        # 重新拋出HTTP異常
        raise e
    except Exception as e:
        logger.error(f"預測過程中發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"預測失敗: {str(e)}"
        )

@app.post("/predict_speed_batch", response_model=List[PredictionResponse])
async def predict_speed_batch(requests: List[PredictionRequest]):
    """
    批量預測羽毛球速度
    
    Args:
        requests: 多個預測請求的列表
        
    Returns:
        對應的預測結果列表
    """
    if len(requests) > 50:  # 限制批量大小
        raise HTTPException(
            status_code=400,
            detail="批量請求數量不能超過50個"
        )
    
    results = []
    for i, request in enumerate(requests):
        try:
            result = await predict_speed(request)
            results.append(result)
        except Exception as e:
            logger.error(f"批量預測中第{i+1}個請求失敗: {str(e)}")
            # 為失敗的請求添加錯誤響應
            results.append(PredictionResponse(
                predicted_speed=-1.0,  # 使用-1表示預測失敗
                confidence_info={
                    "error": str(e),
                    "prediction_failed": True
                }
            ))
    
    return results

# 錯誤處理
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"值錯誤: {str(exc)}")
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未處理的異常: {str(exc)}")
    return HTTPException(status_code=500, detail="內部服務器錯誤")

if __name__ == "__main__":
    import uvicorn
    
    # 運行服務
    uvicorn.run(
        "main:app",  # 假設文件名為main.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開發模式
        log_level="info"
    )