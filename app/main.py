from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from typing import List

from app.models.schemas import SensorFrame, PredictionRequest, PredictionResponse, HealthResponse
from app.services.prediction_service import preprocess_sensor_data, validate_sensor_data_range
from app.core import config
from app.core.config import lifespan

# 創建FastAPI應用
app = FastAPI(
    title="羽毛球速度預測API",
    description="基於感測器數據預測羽毛球速度的API服務",
    version="1.0.0",
    lifespan=lifespan
)





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
    model_loaded = config.model is not None and config.feature_scaler is not None and config.target_scaler is not None
    
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
        if config.model is None or config.feature_scaler is None or config.target_scaler is None:
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
        prediction_scaled = config.model.predict(X_processed, verbose=0)[0][0]
        
        # 將預測結果轉換回原始尺度
        prediction_original = config.target_scaler.inverse_transform([[prediction_scaled]])[0][0]
        
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
        reload=False,  # 關閉開發模式，確保日誌正常顯示
        log_level="info"
    )