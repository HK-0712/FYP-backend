import torch
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModelForCTC
import os
import re
import pycantonese
import numpy as np
from datetime import datetime, timezone

# --- 全域設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_zh_hk.py is configured to use device: {DEVICE}")

MODEL_NAME = "HK0712/Wav2Vec2_Cantonese"

# --- 1. 輔助函數：粵拼智慧切分器 ---
def _tokenize_jyutping_smart(jyutping_str: str) -> list:
    """
    將單個粵拼音節 (如 'gwong2') 根據聲韻學結構切分為 token。
    Target: 'gwong2' -> ['gw', 'o', 'ng', '2']
    """
    try:
        parsed = pycantonese.parse_jyutping(jyutping_str)
        tokens = []
        for jp in parsed:
            if jp.onset: tokens.append(jp.onset)
            if jp.nucleus: tokens.append(jp.nucleus)
            if jp.coda: tokens.append(jp.coda)
            if jp.tone: tokens.append(jp.tone)
        return tokens
    except:
        return re.findall(r'[a-z]+|[0-9]', jyutping_str)

# --- 2. 智慧 G2P 歸屬邏輯 ---
def _get_target_jyutping_by_char(sentence: str) -> (list, list):
    """
    將中文句子轉換為「字」級別的粵拼目標。
    """
    segmented_result = pycantonese.characters_to_jyutping(sentence)
    
    original_chars_flat = []
    target_jyutping_groups = []
    jyutping_syllable_pattern = re.compile(r'([a-z]+[1-6])')

    for word_segment, jyutping_segment in segmented_result:
        if not jyutping_segment: continue

        syllables = jyutping_syllable_pattern.findall(jyutping_segment)
        
        if len(word_segment) == len(syllables):
            for char, syl in zip(word_segment, syllables):
                original_chars_flat.append(char)
                target_jyutping_groups.append(_tokenize_jyutping_smart(syl))
        else:
            print(f"WARNING: Mismatch length for {word_segment}. Fallback to char-by-char G2P.")
            for char in word_segment:
                original_chars_flat.append(char)
                single_res = pycantonese.characters_to_jyutping(char)
                if single_res and single_res[0][1]:
                    target_jyutping_groups.append(_tokenize_jyutping_smart(single_res[0][1]))
                else:
                    target_jyutping_groups.append([])

    return original_chars_flat, target_jyutping_groups

# --- 3. 核心分析函數 ---
def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    if "model" not in cache:
        print(f"Cache miss (ASR_zh_hk). Loading model '{MODEL_NAME}'...")
        try:
            cache["processor"] = AutoProcessor.from_pretrained(MODEL_NAME)
            model = AutoModelForCTC.from_pretrained(MODEL_NAME)
            
            if DEVICE == "cpu":
                print("⚠️ CPU detected. Quantizing model...")
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            
            model.to(DEVICE)
            cache["model"] = model
            print(f"Model '{MODEL_NAME}' loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    processor = cache["processor"]
    model = cache["model"]

    # 1. 準備目標 (Target)
    target_chars, target_jyutping_by_char = _get_target_jyutping_by_char(target_sentence)
    
    # 2. 推理 (Inference)
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"Audio error: {e}")
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    if DEVICE == "cuda": input_values = input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # 3. 獲取使用者輸出
    raw_output_str = processor.decode(predicted_ids[0])
    
    # 處理 User Tokens
    # 嘗試抓取標準音節，如果失敗則退化為 smart parse
    user_tokens = []
    user_syllables = re.findall(r'[a-z]+[0-9]', raw_output_str)
    
    if user_syllables:
        for syl in user_syllables:
            user_tokens.extend(_tokenize_jyutping_smart(syl))
    else:
        # 如果用戶完全沒讀出聲調，或者是亂碼
        user_tokens = _tokenize_jyutping_smart(raw_output_str)

    # 4. 對齊 (Alignment)
    word_alignments = _get_phoneme_alignments_by_word(user_tokens, target_jyutping_by_char)

    return _format_to_json_structure(word_alignments, target_sentence, target_chars)


# --- 4. 對齊函數 (已強化：類型感知 Type-Aware) ---
def _get_phoneme_alignments_by_word(user_phonemes, target_words_ipa_tokenized):
    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

    # DP Initialization
    dp = np.zeros((len(user_phonemes) + 1, len(target_phonemes_flat) + 1))
    for i in range(1, len(user_phonemes) + 1): dp[i][0] = i
    for j in range(1, len(target_phonemes_flat) + 1): dp[0][j] = j
    
    # 【【【 Type-Aware Cost Calculation 】】】
    for i in range(1, len(user_phonemes) + 1):
        for j in range(1, len(target_phonemes_flat) + 1):
            u_char = user_phonemes[i-1]
            t_char = target_phonemes_flat[j-1]
            
            # 判斷是否為數字 (聲調)
            u_is_digit = u_char.isdigit()
            t_is_digit = t_char.isdigit()
            
            if u_char == t_char:
                cost = 0
            elif u_is_digit != t_is_digit:
                # 💥 關鍵修改：如果類型不同 (數字 vs 字母)，給予超大懲罰
                # 這會強制算法選擇 Insertion 或 Deletion，而不是 Substitution
                cost = 100 
            else:
                # 類型相同但字符不同 (e.g. '2' vs '3', 'a' vs 'o') -> 一般錯誤
                cost = 1
                
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    # Backtracking (需要保持一致的 cost 邏輯)
    i, j = len(user_phonemes), len(target_phonemes_flat)
    user_path, target_path = [], []
    while i > 0 or j > 0:
        # 重算當前格子的 cost 以決定路徑
        if i > 0 and j > 0:
            u_char = user_phonemes[i-1]
            t_char = target_phonemes_flat[j-1]
            u_is_digit = u_char.isdigit()
            t_is_digit = t_char.isdigit()
            
            if u_char == t_char:
                match_cost = 0
            elif u_is_digit != t_is_digit:
                match_cost = 100
            else:
                match_cost = 1
        else:
            match_cost = float('inf') # 邊界情況

        # 檢查是否來自對角線 (Substitution/Match)
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + match_cost:
            user_path.insert(0, user_phonemes[i-1])
            target_path.insert(0, target_phonemes_flat[j-1])
            i -= 1; j -= 1
        # 檢查是否來自上方 (Deletion / Missing in User)
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            user_path.insert(0, user_phonemes[i-1])
            target_path.insert(0, '-')
            i -= 1
        # 檢查是否來自左方 (Insertion / Extra in User)
        else:
            user_path.insert(0, '-')
            target_path.insert(0, target_phonemes_flat[j-1])
            j -= 1
    
    # --- 下面的切分邏輯保持不變 ---
    alignments_by_word = []
    word_start_idx_in_path = 0
    target_phoneme_counter_in_path = 0
    num_words_to_align = len(target_words_ipa_tokenized)
    current_word_idx = 0
    
    if not target_path: return []

    for path_idx, p in enumerate(target_path):
        if p != '-':
            if target_phoneme_counter_in_path in word_boundaries_indices:
                if current_word_idx < num_words_to_align:
                    alignments_by_word.append({
                        "target": target_path[word_start_idx_in_path : path_idx + 1],
                        "user": user_path[word_start_idx_in_path : path_idx + 1]
                    })
                    word_start_idx_in_path = path_idx + 1
                    current_word_idx += 1
            target_phoneme_counter_in_path += 1
    
    if word_start_idx_in_path < len(target_path) and current_word_idx < num_words_to_align:
        alignments_by_word.append({
            "target": target_path[word_start_idx_in_path:],
            "user": user_path[word_start_idx_in_path:]
        })

    return alignments_by_word

# --- 5. 格式化函數 (保持與英文版一致) ---
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
                target_p = alignment['target'][j]
                user_p = alignment['user'][j]
                is_match = (user_p == target_p)
                phonemes_data.append({"target": target_p, "user": user_p, "isMatch": is_match})
                if not is_match:
                    word_is_correct = False
                    if not (user_p == '-' and target_p == '-'): total_errors += 1
            total_phonemes += sum(1 for p in alignment['target'] if p != '-')

        if word_is_correct and phonemes_data: correct_words_count += 1
        words_data.append({"word": original_words[i], "isCorrect": word_is_correct, "phonemes": phonemes_data})
        
    total_words = len(original_words)
    if len(words_data) < total_words:
        _, remaining_targets = _get_target_jyutping_by_char("".join(original_words[len(words_data):]))
        for i, target_group in enumerate(remaining_targets):
            phonemes_data = [{"target": p, "user": "-", "isMatch": False} for p in target_group]
            for _ in target_group: total_errors += 1; total_phonemes += 1
            words_data.append({"word": original_words[len(words_data)], "isCorrect": False, "phonemes": phonemes_data})

    score = (correct_words_count / total_words) * 100 if total_words > 0 else 0
    per = (total_errors / total_phonemes) * 100 if total_phonemes > 0 else 0

    return {
        "sentence": sentence,
        "analysisTimestampUTC": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S (UTC)'),
        "summary": {
            "overallScore": round(score, 1),
            "totalWords": total_words,
            "correctWords": correct_words_count,
            "phonemeErrorRate": round(per, 2),
            "total_errors": total_errors,
            "total_target_phonemes": total_phonemes
        },
        "words": words_data
    }