import requests
import numpy as np
import json

# API基礎URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """測試健康檢查"""
    print("=== 健康檢查測試 ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"狀態碼: {response.status_code}")
        print(f"響應: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康檢查失敗: {e}")
        return False

def generate_sample_data():
    """生成示例感測器數據"""
    np.random.seed(42)
    sensor_data = []
    
    for i in range(30):
        # 生成模擬的感測器數據
        frame = {
            "ax": float(np.random.normal(0, 2)),  # 加速度計數據
            "ay": float(np.random.normal(0, 2)),
            "az": float(np.random.normal(0, 2)),
            "gx": float(np.random.normal(0, 100)),  # 陀螺儀數據
            "gy": float(np.random.normal(0, 100)),
            "gz": float(np.random.normal(0, 100))
        }
        sensor_data.append(frame)
    
    return sensor_data

def test_speed_prediction():
    """測試速度預測"""
    print("\n=== 速度預測測試 ===")
    
    # 生成測試數據
    sensor_data = generate_sample_data()
    
    # 構建請求數據
    request_data = {
        "sensor_data": sensor_data
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_speed",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"預測速度: {result['predicted_speed']} km/h")
            print(f"信心資訊: {result['confidence_info']}")
            return True
        else:
            print(f"預測失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"請求失敗: {e}")
        return False

def test_invalid_data():
    """測試無效數據處理"""
    print("\n=== 無效數據測試 ===")
    
    # 測試數據長度不正確的情況
    invalid_data = {
        "sensor_data": generate_sample_data()[:20]  # 只有20個幀，不是30個
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_speed",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"狀態碼: {response.status_code}")
        print(f"錯誤響應: {response.json()}")
        
        return response.status_code == 422  # 應該返回驗證錯誤
        
    except Exception as e:
        print(f"測試失敗: {e}")
        return False

def test_batch_prediction():
    """測試批量預測"""
    print("\n=== 批量預測測試 ===")
    
    # 生成多個測試樣本
    batch_data = []
    for i in range(3):
        batch_data.append({
            "sensor_data": generate_sample_data()
        })
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_speed_batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"批量預測結果數量: {len(results)}")
            for i, result in enumerate(results):
                print(f"  樣本{i+1}: {result['predicted_speed']} km/h")
            return True
        else:
            print(f"批量預測失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"批量請求失敗: {e}")
        return False

def test_realistic_data():
    """使用更真實的數據進行測試"""
    print("\n=== 真實數據測試 ===")
    
    # 模擬羽毛球揮拍動作的感測器數據
    sensor_data = []
    
    # 模擬揮拍過程：準備 -> 揮拍 -> 擊球 -> 跟隨
    for i in range(30):
        t = i / 29.0  # 時間進度 0-1
        
        if t < 0.3:  # 準備階段
            ax, ay, az = 0.1, 0.2, 0.1
            gx, gy, gz = 5, 10, 5
        elif t < 0.7:  # 揮拍階段
            intensity = np.sin(np.pi * (t - 0.3) / 0.4)
            ax, ay, az = -3 * intensity, 8 * intensity, -2 * intensity
            gx, gy, gz = -200 * intensity, 300 * intensity, -150 * intensity
        else:  # 跟隨階段
            ax, ay, az = -0.5, 1.0, -0.3
            gx, gy, gz = -50, 80, -30
        
        # 添加一些隨機噪聲
        frame = {
            "ax": float(ax + np.random.normal(0, 0.1)),
            "ay": float(ay + np.random.normal(0, 0.1)),
            "az": float(az + np.random.normal(0, 0.1)),
            "gx": float(gx + np.random.normal(0, 10)),
            "gy": float(gy + np.random.normal(0, 10)),
            "gz": float(gz + np.random.normal(0, 10))
        }
        sensor_data.append(frame)
    
    request_data = {"sensor_data": sensor_data}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_speed",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"真實場景預測速度: {result['predicted_speed']} km/h")
            print(f"信心資訊: {result['confidence_info']}")
            return True
        else:
            print(f"真實數據預測失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"真實數據測試失敗: {e}")
        return False

def test_batch_prediction_with_real_data():
    """測試使用真實數據進行批量預測"""
    print("\n=== 真實數據批量預測測試 ===")
    
    try:
        with open('./data/smash_data.json', 'r', encoding='utf-8') as f:
            all_smash_data = json.load(f)
        
        # 獲取前30條數據
        real_data_batch = []
        for entry in all_smash_data[:30]:
            sensor_data = []
            for frame in entry['waveform']:
                sensor_data.append({
                    "ax": frame["ax"],
                    "ay": frame["ay"],
                    "az": frame["az"],
                    "gx": frame["gx"],
                    "gy": frame["gy"],
                    "gz": frame["gz"]
                })
            real_data_batch.append({"sensor_data": sensor_data})
        
        response = requests.post(
            f"{BASE_URL}/predict_speed_batch",
            json=real_data_batch,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"真實數據批量預測結果數量: {len(results)}")
            for i, result in enumerate(results):
                print(f"  樣本{i+1}: {result['predicted_speed']} km/h")
            return True
        else:
            print(f"真實數據批量預測失敗: {response.text}")
            return False
            
    except Exception as e:
        print(f"真實數據批量請求失敗: {e}")
        return False

def main():
    """執行所有測試"""
    print("開始API測試...")
    
    tests = [
        ("健康檢查", test_health_check),
        ("速度預測", test_speed_prediction),
        ("無效數據處理", test_invalid_data),
        ("批量預測", test_batch_prediction),
        ("真實數據測試", test_realistic_data),
        ("真實數據批量預測", test_batch_prediction_with_real_data)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name}測試異常: {e}")
            results[test_name] = False
    
    # 輸出測試總結
    print("\n" + "="*50)
    print("測試總結:")
    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\n總計: {passed_tests}/{total_tests} 個測試通過")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # 確保API服務正在運行
    print("請確保API服務已在 http://localhost:8000 上運行")
    print("運行命令: python main.py")
    input("按Enter鍵開始測試...")
    
    success = main()
    if success:
        print("\n🎉 所有測試通過！API運行正常。")
    else:
        print("\n⚠️  部分測試失敗，請檢查API服務。")