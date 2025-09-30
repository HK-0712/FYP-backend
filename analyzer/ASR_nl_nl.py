# =======================================================================
# analyzer/ASR_nl_nl.py
# 荷蘭語發音分析器
# 最終修正版 - 使用用戶指定的正確模型
# =======================================================================

# 1. 匯入區 (Imports)
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone
import re
import unicodedata

# =======================================================================
# 2. 全域變數與配置區
# =======================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_nl_nl.py is configured to use device: {DEVICE}")

# 【【【【【 最終的、決定性的修正 】】】】】
# 使用用戶指定的、正確的荷蘭語音素模型
MODEL_NAME = "Clementapa/wav2vec2-base-960h-phoneme-reco-dutch"

processor = None
model = None

# =======================================================================
# 3. 核心業務邏輯區
# =======================================================================

# -----------------------------------------------------------------------
# 3.1. 模型載入函數 (邏輯不變)
# -----------------------------------------------------------------------
def load_model():
    """
    載入荷蘭語 ASR 模型和對應的處理器。
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
# 3.2. 通用 IPA 切分函數 (邏輯不變)
# -----------------------------------------------------------------------
def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，可以正確處理任何語言的組合字符。
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

# -----------------------------------------------------------------------
# 3.3. 核心分析函數 (邏輯不變)
# -----------------------------------------------------------------------
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標荷蘭語句子，回傳詳細的發音分析字典。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    target_words_original = re.findall(r"[\w'-]+", target_sentence)
    cleaned_sentence = " ".join(target_words_original)
    
    target_ipa_by_word_str = phonemize(cleaned_sentence, language='nl', backend='espeak', with_stress=True, strip=True).split()
    
    if len(target_words_original) != len(target_ipa_by_word_str):
        print(f"警告: G2P 後單詞數量 ({len(target_ipa_by_word_str)}) 與原始單詞數量 ({len(target_words_original)}) 不匹配。")
        min_len = min(len(target_words_original), len(target_ipa_by_word_str))
        target_words_original = target_words_original[:min_len]
        target_ipa_by_word_str = target_ipa_by_word_str[:min_len]

    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˈ', '').replace('ˌ', '').replace('ː', ''))
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
    user_ipa_full = processor.decode(predicted_ids[0]).replace('|', '')

    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# =======================================================================
# 4. 對齊與格式化函數區 (語言無關，邏輯不變)
# =======================================================================

def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    user_phonemes = _tokenize_ipa(user_phoneme_str)
    target_phonemes_flat = [p for word in target_words_ipa_tokenized for p in word]
    word_boundaries_indices = np.cumsum([len(word) for word in target_words_ipa_tokenized]) - 1
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
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, '-'); i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            user_path.insert(0, '-'); target_path.insert(0, target_phonemes_flat[j-1]); j -= 1
        else: break
    alignments_by_word = []
    word_start_idx_in_path = 0
    target_phoneme_counter_in_path = 0
    word_boundary_iter = iter(word_boundaries_indices)
    current_word_boundary = next(word_boundary_iter, -1)
    for path_idx, p in enumerate(target_path):
        if p != '-':
            if target_phoneme_counter_in_path == current_word_boundary:
                alignments_by_word.append({
                    "target": target_path[word_start_idx_in_path : path_idx + 1],
                    "user": user_path[word_start_idx_in_path : path_idx + 1]
                })
                word_start_idx_in_path = path_idx + 1
                current_word_boundary = next(word_boundary_iter, -1)
            target_phoneme_counter_in_path += 1
    return alignments_by_word

def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    total_phonemes, total_errors, correct_words_count = 0, 0, 0
    words_data = []
    num_words_to_process = min(len(alignments), len(original_words))
    for i in range(num_words_to_process):
        alignment = alignments[i]
        word_is_correct = True
        phonemes_data = []
        min_len = min(len(alignment['target']), len(alignment['user']))
        for j in range(min_len):
            target_phoneme, user_phoneme = alignment['target'][j], alignment['user'][j]
            is_match = (user_phoneme == target_phoneme)
            phonemes_data.append({"target": target_phoneme, "user": user_phoneme, "isMatch": is_match})
            if not is_match:
                word_is_correct = False
                if not (user_phoneme == '-' and target_phoneme == '-'): total_errors += 1
        if word_is_correct: correct_words_count += 1
        words_data.append({"word": original_words[i], "isCorrect": word_is_correct, "phonemes": phonemes_data})
        total_phonemes += sum(1 for p in alignment['target'] if p != '-')
    if len(alignments) < len(original_words):
        for i in range(len(alignments), len(original_words)):
            missed_word_ipa_str = phonemize(original_words[i], language='nl', backend='espeak', strip=True).replace('ː', '')
            missed_word_ipa = _tokenize_ipa(missed_word_ipa_str)
            phonemes_data = []
            for p_ipa in missed_word_ipa:
                phonemes_data.append({"target": p_ipa, "user": "-", "isMatch": False})
                total_errors += 1
                total_phonemes += 1
            words_data.append({"word": original_words[i], "isCorrect": False, "phonemes": phonemes_data})
    total_words = len(original_words)
    overall_score = (correct_words_count / total_words) * 100 if total_words > 0 else 0
    phoneme_error_rate = (total_errors / total_phonemes) * 100 if total_phonemes > 0 else 0
    return {
        "sentence": sentence,
        "analysisTimestampUTC": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S (UTC)'),
        "summary": {
            "overallScore": round(overall_score, 1), "totalWords": total_words, "correctWords": correct_words_count,
            "phonemeErrorRate": round(phoneme_error_rate, 2), "total_errors": total_errors, "total_target_phonemes": total_phonemes
        },
        "words": words_data
    }