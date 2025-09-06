# 1. 選擇一個包含 Python 的官方 Linux 映像
FROM python:3.10-slim

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 安裝系統級依賴 (最關鍵的一步：安裝 espeak-ng 和其他工具)
#    -y 自動回答 'yes'
#    --no-install-recommends 避免安裝不必要的建議套件，保持映像檔小巧
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    wget && \
    rm -rf /var/lib/apt/lists/*

# 4. 複製 requirements.txt 檔案到容器中並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 將專案中的所有其他檔案複製到容器中
COPY . .

# 這行是可選的，它設定了當容器直接執行時的預設命令
# CMD ["python", "your_script.py"]
