import tensorflow as tf
import os
import joblib
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# 全局變量存儲模型和預處理器
model = None
feature_scaler = None
target_scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程序生命週期管理"""
    global model, feature_scaler, target_scaler
    try:
        logger.info("正在加載模型和預處理器...")
        
        # 構建模型和預處理器的絕對路徑
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, 'ml_model', 'badminton_speed_predictor.h5')
        logger.info(f"model_path: {model_path}")
        feature_scaler_path = os.path.join(base_dir, 'ml_model', 'feature_scaler.pkl')
        target_scaler_path = os.path.join(base_dir, 'ml_model', 'target_scaler.pkl')

        # 加載訓練好的模型
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("模型加載成功")
        
        # 加載預處理器
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        logger.info("預處理器加載成功")
        
        logger.info("服務初始化完成")
        
    except Exception as e:
        logger.error(f"模型加載失敗: {str(e)}")
        raise e
    
    yield
    
    # 關閉時清理資源
    logger.info("正在關閉服務...")