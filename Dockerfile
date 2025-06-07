FROM python:3.11-slim

WORKDIR /app

# 複製所有專案文件
COPY . .

# 安裝依賴
RUN pip install -r requirements.txt

# 將專案根目錄添加到 PYTHONPATH
ENV PYTHONPATH=/app

# 設定環境變量
ENV PORT=8080

# 啟動命令
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}