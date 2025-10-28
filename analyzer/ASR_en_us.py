import torch
import soundfile as sf
import librosa
# 【【【【【 修改 #1：從 transformers 匯入 AutoProcessor 和 AutoModelForCTC 】】】】】
from transformers import AutoProcessor, AutoModelForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone

# --- 全域設定 (已修改) ---
# 移除了全域的 processor 和 model 變數。
# 刪除了舊的 load_model() 函數。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_en_us.py is configured to use device: {DEVICE}")

# 【【【【【 修改 #2：更新為最終選定的 KoelLabs 模型名稱 】】】】】
MODEL_NAME = "KoelLabs/xlsr-english-01"

# 【【【【【 新增程式碼 #1：為 KoelLabs 模型設計的 IPA 正規化器 】】】】】
# 【保持不變】
def normalize_koel_ipa(raw_phonemes: list) -> list:
    """
    將 KoelLabs 模型輸出的高級 IPA 序列，正規化為與 eSpeak 輸出可比的基礎 IPA 序列。
    """
    normalized_phonemes = []
    for phoneme in raw_phonemes:
        if not phoneme:
            continue
            
        base_phoneme = phoneme.replace('ʰ', '').replace('̃', '').replace('̥', '')
        
        if base_phoneme == 'β':
            base_phoneme = 'v'
        elif base_phoneme in ['x', 'ɣ', 'ɦ']:
            base_phoneme = 'h'
            
        normalized_phonemes.append(base_phoneme)
        
    return normalized_phonemes

# --- 2. 智能 IPA 切分函數 (與您的原版邏輯完全相同) ---
# 【保持不變】
MULTI_CHAR_PHONEMES = {
    'tʃ', 'dʒ',
    'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ',
    'ɪə', 'eə', 'ʊə', 'ər'
}

def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，能正確處理多字元音素。
    """
    s = ipa_string.replace(' ', '').replace('ˌ', '').replace('ˈ', '').replace('ː', '')
    phonemes = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in MULTI_CHAR_PHONEMES:
            phonemes.append(s[i:i+2])
            i += 2
        else:
            phonemes.append(s[i])
            i += 1
    return phonemes

# 【【【【【 全新函式：智慧 G2P 歸屬邏輯 - 方案 B 版本 】】】】】
def _get_target_ipa_by_word(sentence: str) -> (list, list):
    """
    使用「啟發式拆分」方法（方案B），將句子級 G2P 結果智慧地歸屬到每個單字。
    """
    original_words = sentence.strip().split()
    
    # 1. 獲取句子級別的 G2P 結果
    sentence_ipa_groups_raw = [s.strip('[]') for s in phonemize(sentence, language='en-us', backend='espeak', with_stress=True, strip=True).split()]
    sentence_ipa_groups = [_tokenize_ipa(group) for group in sentence_ipa_groups_raw]

    # 如果數量剛好匹配，直接返回，這是最理想的情況
    if len(original_words) == len(sentence_ipa_groups):
        print("G2P alignment perfect match. No heuristic needed.")
        return original_words, sentence_ipa_groups

    # 2. 數量不匹配，啟用啟發式歸屬邏輯
    print(f"G2P Mismatch Detected: {len(original_words)} words vs {len(sentence_ipa_groups)} IPA groups. Applying heuristic splitting.")
    
    # 獲取單字級別的 G2P 結果作為參考
    word_ipas_reference = [_tokenize_ipa(phonemize(word, language='en-us', backend='espeak', strip=True)) for word in original_words]

    final_ipa_by_word = []
    word_idx = 0
    ipa_group_idx = 0

    while word_idx < len(original_words):
        # 邊界檢查：如果句子級音標已經用完
        if ipa_group_idx >= len(sentence_ipa_groups):
            print(f"Warning: Ran out of sentence IPA groups. Appending reference IPA for '{original_words[word_idx]}'.")
            final_ipa_by_word.append(word_ipas_reference[word_idx])
            word_idx += 1
            continue

        current_word = original_words[word_idx]
        current_ipa_group = sentence_ipa_groups[ipa_group_idx]
        ref_ipa_len = len(word_ipas_reference[word_idx])
        
        # 啟發式核心：如果當前句子級音標組比參考音標長，且這不是最後一個詞
        if len(current_ipa_group) > ref_ipa_len and word_idx + 1 < len(original_words):
            # 假設多出來的部分屬於下一個詞
            print(f"Heuristic Split: Splitting IPA group for '{current_word}' and '{original_words[word_idx+1]}'.")
            
            # 切分！
            ipa_for_current_word = current_ipa_group[:ref_ipa_len]
            ipa_for_next_word = current_ipa_group[ref_ipa_len:]
            
            final_ipa_by_word.append(ipa_for_current_word)
            final_ipa_by_word.append(ipa_for_next_word)
            
            # 一次處理了兩個詞，所以索引都要加 2
            word_idx += 2
            ipa_group_idx += 1
        else:
            # 正常情況：長度匹配或無法應用啟發式規則
            final_ipa_by_word.append(current_ipa_group)
            word_idx += 1
            ipa_group_idx += 1

    # 最後的長度校驗，如果不匹配，證明啟發式失敗，執行最終回退
    if len(final_ipa_by_word) != len(original_words):
        print(f"Heuristic splitting failed (final count: {len(final_ipa_by_word)} vs {len(original_words)}). Falling back to word-by-word G2P for safety.")
        return original_words, word_ipas_reference

    print("Heuristic splitting successful.")
    return original_words, final_ipa_by_word


# --- 3. 核心分析函數 (主入口) (已修改以整合正規化器和快取邏輯) ---
def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    模型會被載入並儲存在此函數獨立的 'cache' 中，實現狀態隔離。
    """
    # 檢查快取中是否已有模型，如果沒有則載入
    if "model" not in cache:
        print(f"快取未命中 (ASR_en_us)。正在載入模型 '{MODEL_NAME}'...")
        try:
            cache["processor"] = AutoProcessor.from_pretrained(MODEL_NAME)
            cache["model"] = AutoModelForCTC.from_pretrained(MODEL_NAME)
            cache["model"].to(DEVICE)
            print(f"模型 '{MODEL_NAME}' 已載入並快取。")
        except Exception as e:
            print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
            raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

    # 從此函數的獨立快取中獲取模型和處理器
    processor = cache["processor"]
    model = cache["model"]

    # --- 【【【【【 主要修改點：使用新的智慧 G2P 函式 】】】】】 ---
    target_words_original, target_ipa_by_word = _get_target_ipa_by_word(target_sentence)
    
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
    
    raw_user_ipa_str = processor.decode(predicted_ids[0])
    raw_user_phonemes = raw_user_ipa_str.split(' ')
    normalized_user_phonemes = normalize_koel_ipa(raw_user_phonemes)
    user_ipa_full = "".join(normalized_user_phonemes)

    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)


# --- 4. 對齊函數 (與您的原版邏輯完全相同) ---
# 【保持不變】
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    (已修改) 使用新的切分邏輯執行音素對齊。
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

    num_words_to_align = len(target_words_ipa_tokenized)
    current_word_idx = 0
    
    if not target_path:
        return []

    for path_idx, p in enumerate(target_path):
        if p != '-':
            if target_phoneme_counter_in_path in word_boundaries_indices:
                if current_word_idx < num_words_to_align:
                    target_alignment = target_path[word_start_idx_in_path : path_idx + 1]
                    user_alignment = user_path[word_start_idx_in_path : path_idx + 1]
                    
                    alignments_by_word.append({
                        "target": target_alignment,
                        "user": user_alignment
                    })
                    
                    word_start_idx_in_path = path_idx + 1
                    current_word_idx += 1

            target_phoneme_counter_in_path += 1
    
    if word_start_idx_in_path < len(target_path) and current_word_idx < num_words_to_align:
        target_alignment = target_path[word_start_idx_in_path:]
        user_alignment = user_path[word_start_idx_in_path:]
        alignments_by_word.append({
            "target": target_alignment,
            "user": user_alignment
        })

    return alignments_by_word


# --- 5. 格式化函數 (與您的原版邏輯完全相同) ---
# 【保持不變】
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
        
        if not alignment or not alignment.get('target'):
            word_is_correct = False
        else:
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
            total_phonemes += sum(1 for p in alignment['target'] if p != '-')

        if word_is_correct and phonemes_data:
            correct_words_count += 1
            
        words_data.append({
            "word": original_words[i],
            "isCorrect": word_is_correct,
            "phonemes": phonemes_data
        })
        
    total_words = len(original_words)
    if len(words_data) < total_words:
        for i in range(len(words_data), total_words):
            missed_word_ipa_str = phonemize(original_words[i], language='en-us', backend='espeak', strip=True)
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
