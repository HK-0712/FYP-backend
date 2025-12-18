import torch
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModelForCTC
import os
import pycantonese
import numpy as np
from datetime import datetime, timezone
import unicodedata
import re

# =======================================================================
# 1. 全域設定與模型載入 (Global Config)
# =======================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_zh_hk.py is configured to use device: {DEVICE}")

MODEL_NAME = "HK0712/Wav2Vec2_Cantonese"

# =======================================================================
# 2. 輔助工具函數 (Helpers)
# =======================================================================

def _tokenize_unicode_ipa(ipa_string: str) -> list:
    """
    智能地切分包含 Unicode 組合字元的 IPA 字串。
    (直接沿用 ASR_fr_fr.py 的邏輯)
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

def _get_target_phonemes_by_word(text: str) -> tuple[list[str], list[list[str]]]:
    """
    使用 pycantonese 將中文文本轉換為對應的單詞列表和 IPA 音素列表。
    """
    # characters_to_jyutping 回傳 [('單詞', 'jyutping'), ...]
    jyutping_result = pycantonese.characters_to_jyutping(text)
    
    target_words_original = []
    target_ipa_by_word = []

    for segment, jp_str in jyutping_result:
        # 過濾掉標點符號或無法轉換的部分 (jp_str 為 None)
        # 也過濾掉空白 segment
        if not segment or not segment.strip() or jp_str is None:
            continue
        
        try:
            # jyutping_to_ipa 回傳一個 IPA 字串列表 (每個音節一個字串)
            ipa_list = pycantonese.jyutping_to_ipa(jp_str)
        except Exception as e:
            print(f"Warning: Failed to convert Jyutping '{jp_str}' to IPA: {e}")
            continue

        if not ipa_list:
            continue

        word_tokens = []
        for ipa_syllable in ipa_list:
            # 將每個音節的 IPA 字串再細分為音素
            word_tokens.extend(_tokenize_unicode_ipa(ipa_syllable))
        
        target_words_original.append(segment)
        target_ipa_by_word.append(word_tokens)
            
    return target_words_original, target_ipa_by_word

def _chars_to_ipa_flat(text: str) -> str:
    """
    將中文字串轉換為扁平的 IPA 字串 (用於處理 ASR 的輸出)。
    """
    jyutping_result = pycantonese.characters_to_jyutping(text)
    full_ipa_tokens = []
    
    for segment, jp_str in jyutping_result:
        if not segment or not segment.strip() or jp_str is None:
            continue
        
        try:
            ipa_list = pycantonese.jyutping_to_ipa(jp_str)
            for ipa_syllable in ipa_list:
                full_ipa_tokens.extend(_tokenize_unicode_ipa(ipa_syllable))
        except:
            pass
            
    # 回傳無空格的串接字串，或者保持 token 結構？
    # 為了配合 _get_phoneme_alignments_by_word 的輸入需求 (user_phoneme_str)，
    # 我們這裡最好回傳 token 列表，但原函數簽名通常接收 string。
    # 這裡我們為了兼容性，將其 join 起來，但這在 tokenization 時可能會混淆。
    # 更好的做法是修改 analyze 讓它直接傳遞 list。
    # 但為了保持 _get_phoneme_alignments_by_word 介面一致 (str, list[list]),
    # 我們可以使用一個特殊的分隔符，或者依賴 _tokenize_unicode_ipa 再次切分。
    # 鑑於 _tokenize_unicode_ipa 處理 unicode 很好，我們將所有音素串接。
    
    return "".join(full_ipa_tokens)

# =======================================================================
# 3. 核心分析函數 (Analyze)
# =======================================================================

def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    """
    # 1. 模型載入與快取
    if "model" not in cache:
        print(f"快取未命中 (ASR_zh_hk)。正在載入模型 '{MODEL_NAME}'...")
        try:
            cache["processor"] = AutoProcessor.from_pretrained(MODEL_NAME)
            cache["model"] = AutoModelForCTC.from_pretrained(MODEL_NAME)
            cache["model"].to(DEVICE)
            print(f"模型 '{MODEL_NAME}' 已載入並快取。")
        except Exception as e:
            print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
            raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

    processor = cache["processor"]
    model = cache["model"]

    # 2. 準備目標音素 (G2P)
    target_words_original, target_ipa_by_word = _get_target_phonemes_by_word(target_sentence)

    if not target_words_original:
        print("警告: G2P 處理後目標句子為空。")
        # 回傳空結果
        return _format_to_json_structure([], target_sentence, [])

    # 3. 執行語音辨識 (ASR)
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if len(speech) == 0:
             raise ValueError("Audio file is empty")
             
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # 模型輸出的是中文字元 (假設 Wav2Vec2_Cantonese 是 character-based)
        user_transcription_chars = processor.decode(predicted_ids[0])
        
        # 4. 將使用者轉錄的字元轉換為 IPA
        user_ipa_full = _chars_to_ipa_flat(user_transcription_chars)

    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # 5. 對齊
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    # 6. 格式化
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# =======================================================================
# 4. 對齊與格式化函數 (Alignment & Formatting)
# =======================================================================

def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用動態規劃執行音素對齊。
    """
    user_phonemes = _tokenize_unicode_ipa(user_phoneme_str)
    
    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

    # 處理空目標的情況
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

    # 修正邊界處理，確保所有路徑都被包含
    word_boundary_iter = iter(word_boundaries_indices)
    current_word_boundary = next(word_boundary_iter, -1)
    
    # 這裡的邏輯需要與 target_path 的長度匹配
    # target_phoneme_counter_in_path 只在 target_path[k] != '-' 時增加
    
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
            
    # 處理最後一個詞 (如果還沒處理完)
    # 如果最後一個詞是缺失的 (全 '-'), 上面的邏輯可能無法捕捉
    # 但通常 target_path 不會全是 '-' 除非 target 為空
    
    return alignments_by_word

def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    """
    將對齊結果格式化為最終的 JSON 結構。
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
        
        # 確保 target 和 user 長度一致 (對齊算法保證)
        length = len(alignment['target'])
        
        for j in range(length):
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

    # 處理未對齊的剩餘單詞 (Missed words)
    if len(alignments) < len(original_words):
        for i in range(len(alignments), len(original_words)):
            # 獲取遺失單詞的音標
            missed_word = original_words[i]
            # 這裡簡單調用 G2P 獲取目標音標
            _, missed_word_ipa_list = _get_target_phonemes_by_word(missed_word)
            
            phonemes_data = []
            if missed_word_ipa_list:
                for p_ipa in missed_word_ipa_list[0]:
                    phonemes_data.append({"target": p_ipa, "user": "-", "isMatch": False})
                    total_errors += 1
                    total_phonemes += 1
            
            words_data.append({
                "word": missed_word,
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
