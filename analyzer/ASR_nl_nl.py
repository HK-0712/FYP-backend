# =======================================================================
# analyzer/ASR_nl_nl.py
# 荷蘭語發音分析器
# 版本：v2.0 (與 en_us.py 邏輯對齊)
# 描述：此版本完全遵循 en_us.py 的程式碼結構和算法實現，
#       僅在語言特定配置（模型名稱、G2P語言）上有所不同，
#       並採用了更健壯的、基於 Unicode 的 IPA 切分方法。
# =======================================================================

# --- 1. 匯入區 (與 en_us.py 保持一致) ---
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone
import unicodedata # 【保留】這是處理多語言音素的更優方案
import re # 【保留】用於更準確地切分單詞

# --- 2. 全域設定與模型載入 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_nl_nl.py is configured to use device: {DEVICE}")

# 【關鍵修改 1：設定為荷蘭語 ASR 模型】
MODEL_NAME = "Clementapa/wav2vec2-base-960h-phoneme-reco-dutch"

processor = None
model = None

def load_model():
    """
    載入荷蘭語 ASR 模型和對應的處理器。
    (此函數邏輯與 en_us.py 完全相同)
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

# --- 3. 智能 IPA 切分函數 ---
# 【關鍵修改 2：保留更優越的通用切分邏輯】
# 雖然此函數的實現比英文版的更複雜，但它更健壯且適用於包括荷蘭語在內的多種語言。
# 這是為了「fit with Dutch」而必須保留的優化。
def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，能正確處理帶有附加符號的組合字符。
    """
    phonemes = []
    s = ipa_string.replace(' ', '')
    i = 0
    while i < len(s):
        current_char = s[i]
        i += 1
        # 檢查並組合後續的非間距標記 (例如變音符)
        while i < len(s) and unicodedata.category(s[i]) == 'Mn':
            current_char += s[i]
            i += 1
        phonemes.append(current_char)
    return phonemes

# --- 4. 核心分析函數 (主入口) ---
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標荷蘭語句子，回傳詳細的發音分析字典。
    (此函數結構與 en_us.py 完全對齊)
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # 1. 準備目標音素 (G2P)
    # 使用正則表達式準確切分單詞，這比簡單的 .split() 更穩健
    target_words_original = re.findall(r"[\w'-]+", target_sentence)
    cleaned_sentence = " ".join(target_words_original)

    # 【關鍵修改 3：設定 G2P 語言為 'nl'】
    target_ipa_by_word_str = phonemize(
        cleaned_sentence,
        language='nl',
        backend='espeak',
        with_stress=True,
        strip=True
    ).split()
    
    # 健壯性檢查：確保單詞和音素列表長度一致
    if len(target_words_original) != len(target_ipa_by_word_str):
        print(f"警告: G2P 後單詞數量 ({len(target_ipa_by_word_str)}) 與原始單詞數量 ({len(target_words_original)}) 不匹配。將進行截斷。")
        min_len = min(len(target_words_original), len(target_ipa_by_word_str))
        target_words_original = target_words_original[:min_len]
        target_ipa_by_word_str = target_ipa_by_word_str[:min_len]

    # 【關鍵修改 4：與 en_us.py 對齊，在準備目標音素時就清除所有不比較 的符號】
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˈ', '').replace('ˌ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]

    # 2. 處理音訊並進行語音辨識 (ASR)
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
    
    # 【關鍵修改 5：與 en_us.py 對齊，假設模型輸出是乾淨的，或在必要時清理】
    # 移除模型可能產生的分隔符 |，並確保也移除長音符號，以匹配目標音素的處理方式
    user_ipa_full = processor.decode(predicted_ids[0]).replace('|', '').replace('ː', '')

    # 3. 執行對齊並格式化輸出
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# --- 5. 對齊函數 (與 en_us.py 的實現邏輯完全對齊) ---
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用動態規劃執行音素對齊。
    (此函數實現與 en_us.py 完全相同)
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
        # 使用與 en_us.py 相同的、更簡潔的回溯邏輯
        cost = float('inf') if i == 0 or j == 0 else (0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1)
        
        # 優先匹配/替換
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + cost:
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, target_phonemes_flat[j-1]); i -= 1; j -= 1
        # 其次是刪除 (user 多)
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            user_path.insert(0, user_phonemes[i-1]); target_path.insert(0, '-'); i -= 1
        # 最後是插入 (target 多)
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

# --- 6. 格式化函數 (與 en_us.py 的實現邏輯完全對齊) ---
def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    """
    將對齊結果格式化為最終的 JSON 結構。
    (此函數實現與 en_us.py 完全相同，僅 G2P 語言設定不同)
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
                # 只有在不是「目標和用戶都為空」的情況下才計為錯誤
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

    # 處理使用者漏講單詞的情況
    if len(alignments) < len(original_words):
        for i in range(len(alignments), len(original_words)):
            # 【關鍵修改 6：確保此處的 G2P 語言和符號清理也保持一致】
            missed_word_ipa_str = phonemize(original_words[i], language='nl', backend='espeak', strip=True).replace('ː', '')
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
