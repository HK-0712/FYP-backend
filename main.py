# main.py

# =======================================================================
# 1. 匯入區 (Imports)
#    所有需要的模組都放在檔案的最頂部。
# =======================================================================
import os
import importlib
import uvicorn
import shutil
import re  # <--- 【【新增】】為了動態掃描檔案
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware

# =======================================================================
# 2. 全域變數與配置區 (Global Variables & Config)
#    在應用程式啟動前，定義好所有全域會用到的變數。
# =======================================================================
# 【【修改】】不再硬編碼，讓它們在啟動時被動態填充
SUPPORTED_LANGUAGES = []
ANALYZERS = {}

# 【【新增】】讀取環境變數「信號旗」
APP_ENV = os.getenv('APP_ENV', 'development')

# 【【新增】】這個臨時目錄的定義也放在這裡，保持整潔
TEMP_DIR = "/tmp/temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# =======================================================================
# 3. 應用程式生命週期管理區 (Application Lifespan Management)
#    這是最核心的新增部分，它控制著應用程式啟動和關閉時的行為。
# =======================================================================
# 【【新增】】動態掃描 analyzer/ 資料夾的輔助函數
def discover_supported_languages():
    global SUPPORTED_LANGUAGES
    analyzer_dir = 'analyzer'
    pattern = re.compile(r"ASR_(.+)\.py$")
    try:
        for filename in os.listdir(analyzer_dir):
            match = pattern.match(filename)
            if match:
                language_code = match.group(1)
                SUPPORTED_LANGUAGES.append(language_code)
        if not SUPPORTED_LANGUAGES:
            print("WARNING: No language analyzer modules found in 'analyzer' directory.")
        else:
            print(f"Discovered supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    except FileNotFoundError:
        print(f"ERROR: The '{analyzer_dir}' directory was not found.")
        SUPPORTED_LANGUAGES = []

# 【【新增】】FastAPI 的 lifespan 管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 應用程式啟動時執行的程式碼 ---
    print("="*60)
    print(f"Application starting up in '{APP_ENV}' mode.")
    
    discover_supported_languages() # 首先，動態發現支援的語言
    
    if APP_ENV == "production":
        print("Production mode detected. Eager loading all models...")
        for lang in SUPPORTED_LANGUAGES:
            try:
                print(f"--- Loading model for language: {lang} ---")
                analyzer_module = importlib.import_module(f"analyzer.ASR_{lang}")
                analyzer_module.load_model()
                ANALYZERS[lang] = analyzer_module
                print(f"--- Model for {lang} loaded successfully. ---")
            except Exception as e:
                print(f"FATAL: Failed to load model for {lang}. Error: {e}")
        print("All models pre-loaded. Application is ready for requests.")
    else:
        print("Development mode detected. Models will be loaded on-demand (lazily).")
    
    print("="*60)
    yield # <--- 應用程式在這裡運行
    # --- 應用程式關閉時執行的程式碼 ---
    print("Application shutting down.")

# =======================================================================
# 4. FastAPI 應用程式實例化與中介軟體設定區
#    創建 FastAPI app 物件，並掛載所有需要的設定。
# =======================================================================
# 【【修改】】告訴 FastAPI 使用我們上面定義的 lifespan
app = FastAPI(title="Pronunciation Analysis API", lifespan=lifespan)

# CORS 中介軟體設定 (不變)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================================
# 5. 核心業務邏輯與 API 端點區
#    定義所有 API 的路由和處理函數。
# =======================================================================
api_router = APIRouter(prefix="/api/v1")

# 【【修改】】這是我們新的、更簡潔的 get_analyzer_module 函數
def get_analyzer_module(language: str):
    if language in ANALYZERS:
        return ANALYZERS[language]
    
    # 這段程式碼只會在開發模式下被執行
    print(f"'{language}' not in cache. Loading on-demand (development mode)...")
    try:
        analyzer_module = importlib.import_module(f"analyzer.ASR_{language}")
        analyzer_module.load_model()
        ANALYZERS[language] = analyzer_module
        print(f"'{language}' analyzer loaded and cached successfully.")
        return analyzer_module
    except ImportError:
        print(f"Error: Analyzer module for '{language}' not found (analyzer.ASR_{language}.py).")
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    except Exception as e:
        print(f"Error: A critical error occurred while loading the model for '{language}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model for language '{language}'.")

# API 端點定義 (不變)
@api_router.post("/recognize")
async def recognize_speech_api(
    file: UploadFile = File(...),
    target_sentence: str = Form(...),
    language: str = Form(...)
):
    analyzer_module = get_analyzer_module(language)
    base_filename = os.path.basename(file.filename)
    temp_file_path = os.path.join(TEMP_DIR, f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{base_filename}")
    print(f"Processing file: {temp_file_path} with '{language}' analyzer.")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = analyzer_module.analyze(temp_file_path, target_sentence)
        return result
    except Exception as e:
        print(f"An unexpected error occurred during request processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# 將路由掛載到主應用上
app.include_router(api_router)

# =======================================================================
# 6. 主程式入口 (Main Entry Point)
#    當您直接運行 `python main.py` 時，這部分程式碼會被執行。
# =======================================================================
if __name__ == "__main__":
    print("="*60)
    print("Starting FastAPI server in development mode (http://localhost:8000 )...")
    print("Run ngrok manually if needed: ngrok http 8000" )
    print("="*60)
    # uvicorn 會自動找到 app 物件並運行它
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
