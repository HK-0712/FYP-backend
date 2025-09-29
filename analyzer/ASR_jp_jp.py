# =======================================================================
# 1. 匯入區 (Imports)
#    【關鍵修改】新增了 pyopenjtalk 和 MeCab 的匯入
# =======================================================================
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import pyopenjtalk
import MeCab
import numpy as np
from datetime import datetime, timezone
import re

# =======================================================================
# 2. 全域變數與配置區 (Global Variables & Config)
# =======================================================================
# 【關鍵修改】自動檢測可用設備
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_jp_jp.py is configured to use device: {DEVICE}")

# 【關鍵修改】設定為日語 ASR 模型
MODEL_NAME = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v3"

processor = None
model = None

# 【關鍵修改】初始化 MeCab 分詞器
# 我們使用 -Owakati 選項來獲得以空格分隔的單詞列表
mecab_tagger = MeCab.Tagger("-Owakati")

# =======================================================================
# 3. 核心業務邏輯區 (Core Business Logic)
# =======================================================================

# -----------------------------------------------------------------------
# 3.1. 模型載入函數 (與其他版本邏輯相同)
# -----------------------------------------------------------------------
def load_model():
    """
    載入日語 ASR 模型和對應的處理器。
    """
    global processor, model
    if processor and model:
        print(f"模型 '{MODEL_NAME}' 已載入，跳過。")
        return True

    print(f"正在準備 ASR 模型 '{MODEL_NAME}'...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        print(f"模型 '{MODEL_NAME}' 和處理器載入成功！")
        return True
    except Exception as e:
        print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

# -----------------------------------------------------------------------
# 3.2. 日語 G2P 輔助函數 (這是此檔案最核心的新增部分)
# -----------------------------------------------------------------------
def japanese_g2p(text: str) -> list[tuple[str, str]]:
    """
    將日語句子轉換為 (單詞, 對應音素) 的元組列表。
    這是我們為日語定製的 G2P 核心。
    """
    # 1. 使用 MeCab 進行分詞
    words = mecab_tagger.parse(text).strip().split(' ')
    
    # 2. 對整個句子使用 PyOpenJTalk 獲取完整的音素序列
    #    我們直接使用 pyopenjtalk.g2p，它輸出的就是以空格分隔的音素
    full_phonemes_str = pyopenjtalk.g2p(text)
    
    # 3. 進行音素清理，以匹配 ASR 模型的輸出
    #    ASR 模型輸出的是清音，所以我們移除濁音、半濁音、長音等符號
    cleaned_phonemes = full_phonemes_str.replace('pau', ' ').replace(' ', '').replace('N', 'n').replace('cl', '')
    
    # 4. 將單詞和音素進行配對
    #    這是一個簡化的配對邏輯：我們假設音素的數量和假名的數量大致對應
    #    這在大多數情況下是有效的，因為日語是音節語言
    result = []
    phoneme_idx = 0
    for word in words:
        # 計算當前單詞大致對應多少個音素 (假名數量)
        num_mora = len(word)
        
        # 提取對應的音素片段
        word_phonemes = cleaned_phonemes[phoneme_idx : phoneme_idx + num_mora]
        
        # 檢查提取的音素是否為空，避免無效單詞的影響
        if word_phonemes:
            result.append((word, word_phonemes))
        
        phoneme_idx += num_mora
        
    return result

# -----------------------------------------------------------------------
# 3.3. 音素切分函數 (與其他版本邏輯相同，但更通用)
# -----------------------------------------------------------------------
def _tokenize_ipa(ipa_string: str) -> list:
    """
    將音素字串切分為列表。對於日語，直接按字元切分即可。
    """
    # 日語 ASR 模型的輸出是單字元音素，所以直接轉換為列表
    return list(ipa_string)

# -----------------------------------------------------------------------
# 3.4. 核心分析函數 (主入口，已修改為日語邏輯)
# -----------------------------------------------------------------------
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標日語句子，回傳詳細的發音分析字典。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # 【關鍵修改】使用我們新的日語 G2P 函數
    g2p_result = japanese_g2p(target_sentence)
    
    # 從 G2P 結果中提取原始單詞列表和按單詞劃分的音素列表
    target_words_original = [item[0] for item in g2p_result]
    target_ipa_by_word = [_tokenize_ipa(item[1]) for item in g2p_result]

    # 載入並處理音訊 (與其他版本邏輯相同)
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # 進行 ASR 推論 (與其他版本邏輯相同)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    user_ipa_full = processor.decode(predicted_ids[0])

    # 進行對齊 (與其他版本邏輯相同)
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    # 格式化輸出 (與其他版本邏輯相同)
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)

# =======================================================================
# 4. 對齊與格式化函數區 (Alignment & Formatting)
#    【注意】這些函數是語言無關的，直接從英文版複製，無需修改
# =======================================================================

# -----------------------------------------------------------------------
# 4.1. 對齊函數
# -----------------------------------------------------------------------
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    執行音素對齊。此函數是語言無關的。
    """
    user_phonemes = _tokenize_ipa(user_phoneme_str)
    
    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

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
        cost = float('inf') if i == 0 or j == 0 else (0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1)
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + cost:
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, target_phonemes_flat[j-1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, '-'); i -= 1
        else:
            user_path.insert(0, '-'); target_path.insert(0, target_phonemes_flat[j-1]); j -= 1
    
    alignments_by_word = []
    word_start_idx_in_path = 0
    target_phoneme_counter_in_path = 0

    for path_idx, p in enumerate(target_path):
        if p != '-':
            if target_phoneme_counter_in_path in word_boundaries_indices:
                target_alignment = target_path[word_start_idx_in_path : path_idx + 1]
                user_alignment = user_path[word_start_idx_in_path : path_idx + 1]
                
                alignments_by_word.append({
                    "target": target_alignment,
                    "user": user_alignment
                })
                
                word_start_idx_in_path = path_idx + 1
            
            target_phoneme_counter_in_path += 1
            
    return alignments_by_word

# -----------------------------------------------------------------------
# 4.2. 格式化函數
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
        
        for j in range(len(alignment['target'])):
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

    total_words = len(original_words)
    if len(alignments) < total_words:
        for i in range(len(alignments), total_words):
            # 處理使用者未說出的單詞
            missed_word_ipa = _tokenize_ipa(japanese_g2p(original_words[i])[0][1]) # 重新獲取音素
            phonemes_data = []
            for p_ipa in missed_word_ipa:
                phonemes_data.append({"target": p_ipa, "user": "-", "isMatch": False})
                total_errors += 1
                total_phonemes += 1

            words_data.append({
                "word": original_words[i],
                "isCorrect": False,
                "phonemes": phonemes_data
            })

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
