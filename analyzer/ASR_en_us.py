# analyzer/ASR_en_us.py

import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone

# --- 1. 全域設定與模型載入函數 (保持不變) ---
MODEL_NAME = "MultiBridge/wav2vec-LnNor-IPA-ft"
MODEL_SAVE_PATH = "./ASRs/MultiBridge-wav2vec-LnNor-IPA-ft-local"

processor = None
model = None

def load_model():
    """
    在應用程式啟動時載入模型和處理器。
    如果模型已載入，則跳過。
    """
    global processor, model
    if processor and model:
        print("英文模型已載入，跳過。")
        return True

    print(f"正在準備英文 (en-us) ASR 模型 '{MODEL_NAME}'...")
    try:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"本地找不到模型，正在從 Hugging Face 下載並儲存...")
            processor_to_save = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            model_to_save = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
            processor_to_save.save_pretrained(MODEL_SAVE_PATH)
            model_to_save.save_pretrained(MODEL_SAVE_PATH)
            print("模型已成功下載並儲存。")
        else:
            print(f"在 '{MODEL_SAVE_PATH}' 中找到本地模型。")
        
        processor = Wav2Vec2Processor.from_pretrained(MODEL_SAVE_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_SAVE_PATH)
        print("英文 (en-us) 模型和處理器載入成功！")
        return True
    except Exception as e:
        print(f"處理或載入 en-us 模型時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load en-us model: {e}")

# --- 2. 核心分析函數 (主入口) (保持不變) ---
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    這是此模組的主要進入點。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    target_ipa_by_word = [
        word.replace('ˌ', '').replace('ˈ', '').replace('ː', '')
        for word in phonemize(target_sentence, language='en-us', backend='espeak', with_stress=True).split()
    ]
    target_words_original = target_sentence.split()

    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    user_ipa_full = processor.decode(predicted_ids[0])

    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# --- 3. 對齊函數 (核心邏輯已修改) ---
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa):
    """
    (已修改) 執行音素對齊並確保輸出與單詞數量匹配。
    """
    user_phonemes = list(user_phoneme_str.replace(' ', ''))
    
    # --- DP 矩陣計算部分 (保持不變) ---
    target_phonemes_flat = []
    word_boundaries_indices = [] # 儲存每個單詞結束時在扁平列表中的索引
    current_idx = 0
    for word_ipa in target_words_ipa:
        phonemes = list(word_ipa)
        target_phonemes_flat.extend(phonemes)
        current_idx += len(phonemes)
        word_boundaries_indices.append(current_idx - 1) # 記錄最後一個音素的索引

    dp = np.zeros((len(user_phonemes) + 1, len(target_phonemes_flat) + 1))
    for i in range(1, len(user_phonemes) + 1): dp[i][0] = i
    for j in range(1, len(target_phonemes_flat) + 1): dp[0][j] = j
    for i in range(1, len(user_phonemes) + 1):
        for j in range(1, len(target_phonemes_flat) + 1):
            cost = 0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    # --- 回溯路徑計算 (保持不變) ---
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
    
    # --- (核心修改) 按單詞邊界分割對齊路徑 ---
    alignments_by_word = []
    word_start_idx_in_path = 0
    target_phoneme_counter_in_path = 0

    # 遍歷對齊後的路徑
    for path_idx, p in enumerate(target_path):
        # 只有當我們遇到一個非'-'的目標音素時，計數器才增加
        if p != '-':
            # 檢查這個音素是否是某個單詞的最後一個音素
            if target_phoneme_counter_in_path in word_boundaries_indices:
                # 如果是，這標誌著一個單詞的結束
                # 提取從上一個單詞結束到現在這個單詞結束的所有路徑
                target_alignment = target_path[word_start_idx_in_path : path_idx + 1]
                user_alignment = user_path[word_start_idx_in_path : path_idx + 1]
                
                alignments_by_word.append({
                    "target": target_alignment,
                    "user": user_alignment
                })
                
                # 更新下一個單詞的起始索引
                word_start_idx_in_path = path_idx + 1
            
            target_phoneme_counter_in_path += 1
            
    return alignments_by_word

# --- 4. 格式化函數 (保持不變) ---
def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    total_phonemes = 0
    total_errors = 0
    correct_words_count = 0
    words_data = []

    for i, alignment in enumerate(alignments):
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
                if user_phoneme != '-' and target_phoneme != '-': total_errors += 1
                elif user_phoneme == '-': total_errors += 1
                else: total_errors += 1
        
        if word_is_correct:
            correct_words_count += 1
            
        words_data.append({
            "word": original_words[i] if i < len(original_words) else "N/A",
            "isCorrect": word_is_correct,
            "phonemes": phonemes_data
        })
        
        total_phonemes += sum(1 for p in alignment['target'] if p != '-')

    total_words = len(alignments)
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
