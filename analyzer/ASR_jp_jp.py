# =======================================================================
# 1. 匯入區 (Imports)
#    - 新增了 pyopenjtalk 和 MeCab
# =======================================================================
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, HubertForCTC
import os
import pyopenjtalk
import MeCab
import numpy as np
from datetime import datetime, timezone
import re

# =======================================================================
# 2. 全域變數與配置區 (Global Variables & Config)
#    【已修改】移除了全域的 processor 和 model 變數。
# =======================================================================
# 自動檢測可用設備
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_jp_jp.py is configured to use device: {DEVICE}")

# 設定為日語 ASR 模型
MODEL_NAME = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v3"

# 初始化 MeCab 分詞器
# -Owakati 選項能直接輸出以空格分隔的單詞，非常方便
try:
    mecab_tagger = MeCab.Tagger("-Owakati")
except RuntimeError:
    print("ERROR: MeCab Tagger 初始化失敗。請確保 mecab 和 mecab-ipadic-utf8 已正確安裝。")
    mecab_tagger = None

# =======================================================================
# 3. 核心業務邏輯區 (Core Business Logic)
# =======================================================================

# -----------------------------------------------------------------------
# 3.1. 模型載入函數
#      【已刪除】舊的 load_model() 函數已被移除。
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 3.2. 日語 G2P 輔助函數 (此檔案最核心的修改)
#      【保持不變】
# -----------------------------------------------------------------------
def _get_target_phonemes_by_word(text: str) -> tuple[list[str], list[list[str]]]:
    if not mecab_tagger:
        raise RuntimeError("MeCab Tagger 未初始化，無法處理日語文本。")

    words = mecab_tagger.parse(text).strip().split()
    
    target_words_original = []
    target_ipa_by_word = []

    for word in words:
        if not word:
            continue
        
        phonemes_str = pyopenjtalk.g2p(word, kana=False)
        
        cleaned_phonemes = re.sub(r'\s+', ' ', phonemes_str).strip()
        
        phoneme_list = cleaned_phonemes.split()
        
        if word and phoneme_list:
            target_words_original.append(word)
            target_ipa_by_word.append(phoneme_list)
            
    return target_words_original, target_ipa_by_word

# -----------------------------------------------------------------------
# 3.3. 音素切分函數 (用於處理 ASR 的輸出)
#      【保持不變】
# -----------------------------------------------------------------------
def _tokenize_asr_output(phoneme_string: str) -> list:
    """
    將 ASR 模型輸出的音素字串切分為列表。
    此模型的輸出是單字元音素，以空格分隔。
    """
    return phoneme_string.split()

# -----------------------------------------------------------------------
# 3.4. 核心分析函數 (主入口)
#      【已修改】將模型載入和快取邏輯合併至此。
# -----------------------------------------------------------------------
def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    """
    接收音訊檔案路徑和目標日語句子，回傳詳細的發音分析字典。
    模型會被載入並儲存在此函數獨立的 'cache' 中，實現狀態隔離。
    """
    # 檢查快取中是否已有模型，如果沒有則載入
    if "model" not in cache:
        print(f"快取未命中 (ASR_jp_jp)。正在載入模型 '{MODEL_NAME}'...")
        try:
            # 載入模型並存入此函數的快取字典
            cache["processor"] = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            cache["model"] = HubertForCTC.from_pretrained(MODEL_NAME) # <-- 使用 HubertForCTC
            cache["model"].to(DEVICE)
            print(f"模型 '{MODEL_NAME}' 已載入並快取。")
        except Exception as e:
            print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
            raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

    # 從此函數的獨立快取中獲取模型和處理器
    processor = cache["processor"]
    model = cache["model"]

    # --- 以下為原始分析邏輯，保持不變 ---

    # 【關鍵步驟 1: G2P】
    target_words_original, target_ipa_by_word = _get_target_phonemes_by_word(target_sentence)

    if not target_words_original:
        print("警告: G2P 處理後目標句子為空。")
        return _format_to_json_structure([], target_sentence, [])

    # 【關鍵步驟 2: ASR】
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if len(speech) == 0:
            print("警告: 音訊檔案為空。")
            user_ipa_full = ""
        else:
            if sample_rate != 16000:
                speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
            
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to(DEVICE)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            user_ipa_full = processor.decode(predicted_ids[0])

    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # 【關鍵步驟 3: 對齊】
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    # 【關鍵步驟 4: 格式化】
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)

# =======================================================================
# 4. 對齊與格式化函數區 (Alignment & Formatting)
#    【保持不變】
# =======================================================================

# -----------------------------------------------------------------------
# 4.1. 對齊函數 (語言無關)
# -----------------------------------------------------------------------
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用動態規劃執行音素對齊。此函數是語言無關的。
    """
    user_phonemes = [char for word in user_phoneme_str.split() for char in word]

    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    for word_ipa_tokens in target_words_ipa_tokenized:
        flat_tokens = [char for word in word_ipa_tokens for char in word]
        target_phonemes_flat.extend(flat_tokens)
        current_idx += len(flat_tokens)
        word_boundaries_indices.append(current_idx - 1)

    if not target_phonemes_flat:
        return []

    dp = np.zeros((len(user_phonemes) + 1, len(target_phonemes_flat) + 1))
    for i in range(1, len(user_phonemes) + 1): dp[i][0] = i
    for j in range(1, len(target_phonemes_flat) + 1): dp[0][j] = j
    for i in range(1, len(user_phonemes) + 1):
        for j in range(1, len(target_phonemes_flat) + 1):
            cost = 0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    i, j = len(user_phonemes), len(target_phonemes_flat)
    user_path, target_path = [], []
    while i > 0 or j > 0:
        cost = float('inf')
        if i > 0 and j > 0:
            cost = 0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1

        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + cost:
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, target_phonemes_flat[j-1]); i -= 1; j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, '-'); i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            user_path.insert(0, '-'); target_path.insert(0, target_phonemes_flat[j-1]); j -= 1
        else:
            break
    
    alignments_by_word = []
    word_start_idx_in_path = 0
    target_phoneme_counter_in_path = 0
    word_boundary_iter = iter(word_boundaries_indices)
    current_word_boundary = next(word_boundary_iter, -1)

    for path_idx, p in enumerate(target_path):
        if p != '-':
            if target_phoneme_counter_in_path == current_word_boundary:
                target_alignment = target_path[word_start_idx_in_path : path_idx + 1]
                user_alignment = user_path[word_start_idx_in_path : path_idx + 1]
                
                alignments_by_word.append({
                    "target": target_alignment,
                    "user": user_alignment
                })
                
                word_start_idx_in_path = path_idx + 1
                current_word_boundary = next(word_boundary_iter, -1)
            
            target_phoneme_counter_in_path += 1
            
    return alignments_by_word

# -----------------------------------------------------------------------
# 4.2. 格式化函數 (語言無關)
# -----------------------------------------------------------------------
def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    """
    將對齊結果格式化為最終的 JSON 結構。此函數是語言無關的。
    """
    total_phonemes = 0
    total_errors = 0
    correct_words_count = 0
    words_data = []

    num_words_to_process = min(len(alignments), len(original_words))

    for i in range(num_words_to_process):
        alignment = alignments[i]
        word_is_correct = True
        phonemes_data = []
        
        min_len = min(len(alignment['target']), len(alignment['user']))
        for j in range(min_len):
            target_phoneme = alignment['target'][j]
            user_phoneme = alignment['user'][j]
            is_match = (user_phoneme == target_phoneme)
            
            phonemes_data.append({
                "target": target_phoneme,
                "user": user_phoneme,
                "isMatch": is_match
            })
            
            if not is_match:
                word_is_correct = False
                if not (user_phoneme == '-' and target_phoneme == '-'):
                    total_errors += 1
        
        if word_is_correct:
            correct_words_count += 1
            
        words_data.append({
            "word": original_words[i],
            "isCorrect": word_is_correct,
            "phonemes": phonemes_data
        })
        
        total_phonemes += sum(1 for p in alignment['target'] if p != '-')

    if len(alignments) < len(original_words):
        for i in range(len(alignments), len(original_words)):
            _, missed_word_ipa_list = _get_target_phonemes_by_word(original_words[i])
            
            phonemes_data = []
            if missed_word_ipa_list:
                for p_ipa in missed_word_ipa_list[0]:
                    phonemes_data.append({"target": p_ipa, "user": "-", "isMatch": False})
                    total_errors += 1
                    total_phonemes += 1

            words_data.append({
                "word": original_words[i],
                "isCorrect": False,
                "phonemes": phonemes_data
            })

    total_words = len(original_words)
    overall_score = (correct_words_count / total_words) * 100 if total_words > 0 else 0
    phoneme_error_rate = (total_errors / total_phonemes) * 100 if total_phonemes > 0 else 0

    final_result = {
        "sentence": sentence,
        "analysisTimestampUTC": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S (UTC)'),
        "summary": {
            "overallScore": round(overall_score, 1),
            "totalWords": total_words,
            "correctWords": correct_words_count,
            "phonemeErrorRate": round(phoneme_error_rate, 2),
            "total_errors": total_errors,
            "total_target_phonemes": total_phonemes
        },
        "words": words_data
    }
    
    return final_result
