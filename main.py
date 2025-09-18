# main.py (Final Corrected Version)

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from contextlib import asynccontextmanager
import asyncio
import importlib.util
import sys
from datetime import datetime  # The required import statement

# Ngrok is optional, so we handle its potential absence
try:
    from pyngrok import ngrok, conf
    PYNGROK_INSTALLED = True
except ImportError:
    PYNGROK_INSTALLED = False

# --- Analyzer Loading Logic ---
ANALYZER_MODULES = {}
SUPPORTED_LANGUAGES = ["en_us"]

async def load_analyzers():
    print("正在預載入所有支援的分析器模型...")
    for lang in SUPPORTED_LANGUAGES:
        try:
            module_name = f"analyzer.ASR_{lang}"
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print(f"警告：找不到 {lang} 的分析器模組: {module_name}")
                continue
            
            analyzer_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = analyzer_module
            spec.loader.exec_module(analyzer_module)
            
            if hasattr(analyzer_module, 'load_model'):
                await asyncio.to_thread(analyzer_module.load_model)
                ANALYZER_MODULES[lang] = analyzer_module
                print(f"'{lang}' 分析器載入成功。")
            else:
                print(f"警告：'{lang}' 模組中沒有找到 load_model 函數。")
        except Exception as e:
            print(f"錯誤：載入 '{lang}' 分析器時失敗: {e}")

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("應用程式啟動中...")
    await load_analyzers()
    
    if PYNGROK_INSTALLED:
        NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")
        if NGROK_AUTHTOKEN:
            conf.get_default().auth_token = NGROK_AUTHTOKEN
            print("正在啟動 ngrok 通道...")
            public_url = await asyncio.to_thread(ngrok.connect, 8000, name="pronunciation-api")
            print(f"Ngrok 通道已建立，公開 URL: {public_url}")
        else:
            print("警告：未設定 NGROK_AUTHTOKEN，Ngrok 將不會啟動。")
    else:
        print("警告: pyngrok 套件未安裝，Ngrok 將不會啟動。")

    yield
    
    print("應用程式關閉中...")
    if PYNGROK_INSTALLED and ngrok.get_tunnels():
        ngrok.disconnect()
        print("Ngrok 通道已關閉。")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- API Endpoint ---
@app.post("/api/v1/recognize")
async def recognize_speech_api(
    language: str = Form(...),
    target_sentence: str = Form(...),
    file: UploadFile = File(...)
):
    if language not in ANALYZER_MODULES:
        raise HTTPException(status_code=400, detail=f"不支援的語言: '{language}'。支援的語言: {list(ANALYZER_MODULES.keys())}")

    if not file.filename or not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="檔案格式錯誤或檔名無效，請上傳 .wav 檔案。")

    safe_filename = os.path.basename(file.filename)
    temp_file_path = os.path.join(TEMP_DIR, f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{safe_filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        analyzer_module = ANALYZER_MODULES[language]
        print(f"使用 '{language}' 分析器處理檔案: {file.filename}")
        
        analysis_result = await asyncio.to_thread(
            analyzer_module.analyze, temp_file_path, target_sentence
        )

        return JSONResponse(content=analysis_result)
    except Exception as e:
        print(f"處理請求時發生未預期的錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if file:
            await file.close()

@app.get("/")
def read_root():
    return {"message": "發音分析 API 已啟動。請使用 POST /api/v1/recognize 端點。"}

# --- Server Execution ---
if __name__ == "__main__":
    print("="*60)
    if PYNGROK_INSTALLED:
        print("請確保已設定 NGROK_AUTHTOKEN 環境變數以便 ngrok 正常運作。")
    else:
        print("pyngrok 未安裝，服務僅在本地運行。")
    print("="*60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
