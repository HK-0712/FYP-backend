import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone
import unicodedata
import re
import epitran

# 【【【【【 新增程式碼 #1：自動檢測可用設備 】】】】】
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_fr_fr.py is configured to use device: {DEVICE}")

# --- 1. 全域設定與模型載入函數 (已修改為法語模型) ---
# 移除了全域的 processor 和 model 變數，只保留常數。
# 刪除了舊的 load_model() 函數。
MODEL_NAME = "Cnam-LMSSC/wav2vec2-french-phonemizer"

def _tokenize_unicode_ipa(ipa_string: str) -> list:
    """
    智能地切分包含 Unicode 組合字元的 IPA 字串。
    """
    phonemes = []
    s = ipa_string.replace(' ', '')
    
    i = 0
    while i < len(s):
        current_char = s[i]
        i += 1
        while i < len(s) and unicodedata.category(s[i]) == 'Mn':
            current_char += s[i]
            i += 1
        phonemes.append(current_char)
    return phonemes

# --- 2. 核心分析函數 (主入口) (已修改為法語邏輯) ---
# 將模型載入和快取邏輯合併至此。
def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    模型會被載入並儲存在此函數獨立的 'cache' 中，實現狀態隔離。
    """
    # 檢查快取中是否已有模型，如果沒有則載入
    if "model" not in cache:
        print(f"快取未命中 (ASR_fr_fr)。正在載入模型 '{MODEL_NAME}'...")
        try:
            # 載入模型並存入此函數的快取字典
            cache["processor"] = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            cache["model"] = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
            cache["model"].to(DEVICE)
            print(f"模型 '{MODEL_NAME}' 已載入並快取。")
        except Exception as e:
            print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
            raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

    # 從此函數的獨立快取中獲取模型和處理器
    processor = cache["processor"]
    model = cache["model"]

    # --- 以下為原始分析邏輯，保持不變 ---
    target_words_original = re.findall(r"[\w'-]+", target_sentence)
    cleaned_sentence = " ".join(target_words_original)

    epi_fr = epitran.Epitran('fra-Latn')
    target_ipa_full = epi_fr.transliterate(cleaned_sentence)
    target_ipa_by_word_str = target_ipa_full.split()

    if len(target_ipa_by_word_str) != len(target_words_original):
        target_words_original = target_words_original[:len(target_ipa_by_word_str)]

    target_ipa_by_word = [
        _tokenize_unicode_ipa(word.replace('ˈ', '').replace('ˌ', '').replace('‿', '').replace("'", ""))
        for word in target_ipa_by_word_str
    ]

    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    user_ipa_full = processor.decode(predicted_ids[0]).replace(' ', '')

    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# --- 3. 對齊函數 (已簡化切分邏輯) ---
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    執行音素對齊。對法語使用簡單的字元切分。
    """
    user_phonemes = _tokenize_unicode_ipa(user_phoneme_str)
    
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

# --- 4. 格式化函數 (語言無關，保持不變) ---
def _format_to_json_structure(alignments, sentence, original_words) -> dict:
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
            missed_word_ipa_str = phonemize(original_words[i], language='fr-fr', backend='espeak', strip=True).replace('ˈ', '').replace('ˌ', '').replace('‿', '')
            missed_word_ipa = _tokenize_unicode_ipa(missed_word_ipa_str)
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
