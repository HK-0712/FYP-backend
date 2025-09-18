# main.py (Final, Typo-Fixed Version)

import os
import importlib
import uvicorn
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter # 確保 APIRouter 被正確匯入
from fastapi.middleware.cors import CORSMiddleware

# --- 應用程式設定 ---
app = FastAPI(title="Pronunciation Analysis API")
# 【【【【【 終 極 拼 寫 修 正 在 這 裡 】】】】】
api_router = APIRouter(prefix="/api/v1") # 將 APouter 改為 APIRouter

# --- CORS 設定 (保持不變) ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全域變數和初始化邏輯 (保持不變) ---
TEMP_DIR = "./temp_audio"
ANALYZERS = {} 

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def get_analyzer_module(language: str):
    if language in ANALYZERS:
        return ANALYZERS[language]
    
    try:
        print(f"首次請求 '{language}'，正在動態載入模組...")
        analyzer_module = importlib.import_module(f"analyzer.ASR_{language}")
        
        print(f"正在呼叫 analyzer.ASR_{language}.load_model()...")
        analyzer_module.load_model()
        
        ANALYZERS[language] = analyzer_module
        print(f"'{language}' 分析器模組載入成功並快取。")
        return analyzer_module
    except ImportError:
        print(f"錯誤：找不到 '{language}' 的分析器模組 (analyzer.ASR_{language}.py)。")
        raise HTTPException(status_code=400, detail=f"不支援的語言: {language}")
    except Exception as e:
        print(f"錯誤：載入語言 '{language}' 的模型時發生嚴重錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"載入語言 '{language}' 的模型失敗。")

# --- API 端點 (保持不變) ---
@api_router.post("/recognize")
async def recognize_speech_api(
    file: UploadFile = File(...),
    target_sentence: str = Form(...),
    language: str = Form(...)
):
    analyzer_module = get_analyzer_module(language)
    
    base_filename = os.path.basename(file.filename)
    temp_file_path = os.path.join(TEMP_DIR, f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{base_filename}")
    
    print(f"使用 '{language}' 分析器處理檔案: {temp_file_path}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = analyzer_module.analyze(temp_file_path, target_sentence)
        return result
    except Exception as e:
        print(f"處理請求時發生未預期的錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

app.include_router(api_router)

# --- 主程式入口 (保持不變) ---
if __name__ == "__main__":
    print("="*60)
    print("啟動 FastAPI 伺服器 (http://localhost:8000 )...")
    print("請手動運行 ngrok: ngrok http 8000" )
    print("="*60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
