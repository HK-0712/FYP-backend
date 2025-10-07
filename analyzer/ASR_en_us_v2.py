# ASR_en_us_v2.py

import torch
import soundfile as sf
import librosa
# 【【【【【 關鍵修改 #1：匯入 Whisper 相關的類別 】】】】】
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone

# --- 全域設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_en_us_whisper.py is configured to use device: {DEVICE}")

# 【【【【【 關鍵修改 #2：更新模型名稱 】】】】】
MODEL_NAME = "neurlang/ipa-whisper-base"

processor = None
model = None

def load_model():
    """
    載入新的 Whisper IPA ASR 模型和對應的處理器。
    """
    global processor, model
    if processor and model:
        print(f"模型 '{MODEL_NAME}' 已載入，跳過。")
        return True

    print(f"正在準備 ASR 模型 '{MODEL_NAME}'...")
    print(f"Transformers 將自動在 HF_HOME 指定的快取中尋找或下載。")
    try:
        # 【【【【【 關鍵修改 #3：使用 Whisper 的 Processor 和 Model 類別 】】】】】
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        model.to(DEVICE)
        print(f"模型 '{MODEL_NAME}' 和處理器載入成功！")
        return True
    except Exception as e:
        print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

# --- 智能 IPA 切分函數 (與您的原版邏輯完全相同) ---
# 注意：您可能需要根據新模型的輸出，微調這個多字元音素列表
MULTI_CHAR_PHONEMES = {
    'tʃ', 'dʒ', 
    'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ', 
    'ɪə', 'eə', 'ʊə', 'ər' 
}

def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，能正確處理多字元音素。
    """
    phonemes = []
    i = 0
    # Whisper 的輸出可能包含空格，先移除以進行標準化處理
    s = ipa_string.replace(' ', '')
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in MULTI_CHAR_PHONEMES:
            phonemes.append(s[i:i+2])
            i += 2
        else:
            phonemes.append(s[i])
            i += 1
    return phonemes

# --- 核心分析函數 (主入口) (已修改以適應 Whisper 模型) ---
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    這是此模組的主要進入點。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # 步驟 1：使用 eSpeak 獲取目標 IPA (與原版邏輯相同)
    target_ipa_by_word_str = phonemize(target_sentence, language='en-us', backend='espeak', with_stress=True, strip=True).split()
    
    # 步驟 2：標準化目標 IPA (與原版邏輯相同)
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˌ', '').replace('ˈ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]
    target_words_original = target_sentence.split()

    # 步驟 3：讀取和重採樣音訊 (與原版邏輯相同)
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # 【【【【【 關鍵修改 #4：適應 Whisper 的輸入和生成方式 】】】】】
    # 步驟 4a：使用 WhisperProcessor 處理音訊，得到 input_features
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(DEVICE)

    # 步驟 4b：創建解碼提示，強制模型生成英文的轉寫結果
    # 這是 Whisper 架構的巨大優勢！
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    # 步驟 4c：使用 model.generate() 進行預測，並傳入強制解碼提示
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids
        )
    
    # 步驟 4d：使用 WhisperProcessor 解碼，並跳過特殊 token
    user_ipa_full = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # 步驟 5：使用解碼後的 IPA 進行音素對齊 (後續邏輯與原版完全相同)
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    # 步驟 6：格式化為最終的 JSON 結構 (與原版邏輯完全相同)
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# --- 4. 對齊函數 (與您的原版邏輯完全相同) ---
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
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

# --- 5. 格式化函數 (與您的原版邏輯完全相同) ---
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
            missed_word_ipa_str = phonemize(original_words[i], language='en-us', backend='espeak', strip=True).replace('ː', '')
            missed_word_ipa = _tokenize_ipa(missed_word_ipa_str)
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
