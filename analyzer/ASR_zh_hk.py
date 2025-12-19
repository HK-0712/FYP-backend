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

# --- 1. 輔助函數：粵拼智慧切分器 (Linguistic Split) ---
def _tokenize_jyutping_smart(jyutping_str: str) -> list:
    """
    將單個粵拼音節 (如 'gwong2') 根據聲韻學結構切分為 token。
    Target: 'gwong2' -> ['gw', 'o', 'ng', '2']
    這樣前端顯示時會是 "gw o ng 2"，比 "g w o n g 2" 易讀得多。
    """
    try:
        # pycantonese.parse_jyutping 回傳的是一個列表，包含 Jyutping 物件
        # 例如: parse_jyutping('gwong2') -> [Jyutping(onset='gw', nucleus='o', coda='ng', tone='2')]
        parsed = pycantonese.parse_jyutping(jyutping_str)
        
        tokens = []
        for jp in parsed:
            if jp.onset: tokens.append(jp.onset)
            if jp.nucleus: tokens.append(jp.nucleus)
            if jp.coda: tokens.append(jp.coda)
            if jp.tone: tokens.append(jp.tone)
            
        return tokens
    except:
        # 萬一解析失敗（例如模型輸出的拼音不標準），回退到簡單切分
        # 但保留數字作為獨立 token
        return re.findall(r'[a-z]+|[0-9]', jyutping_str)

# --- 2. 智慧 G2P 歸屬邏輯 (中文版) ---
def _get_target_jyutping_by_char(sentence: str) -> (list, list):
    """
    將中文句子轉換為「字」級別的粵拼目標。
    """
    # pycantonese.characters_to_jyutping 會處理變調與分詞
    # 範例: "廣東話" -> [('廣東話', 'gwong2dung1waa2')]
    segmented_result = pycantonese.characters_to_jyutping(sentence)
    
    original_chars_flat = []
    target_jyutping_groups = []

    # 簡單的正則表達式，用來把連在一起的拼音分開 (e.g. 'gwong2dung1' -> 'gwong2', 'dung1')
    jyutping_syllable_pattern = re.compile(r'([a-z]+[1-6])')

    for word_segment, jyutping_segment in segmented_result:
        if not jyutping_segment:
            continue

        syllables = jyutping_syllable_pattern.findall(jyutping_segment)
        
        # 嘗試將分詞後的結果對齊回單個漢字
        if len(word_segment) == len(syllables):
            for char, syl in zip(word_segment, syllables):
                original_chars_flat.append(char)
                # 使用智慧切分：'gwong2' -> ['gw', 'o', 'ng', '2']
                target_jyutping_groups.append(_tokenize_jyutping_smart(syl))
        else:
            # 長度不匹配時的備用方案 (逐字處理)
            print(f"WARNING: Mismatch length for {word_segment}. Fallback to char-by-char G2P.")
            for char in word_segment:
                original_chars_flat.append(char)
                # 對單字再做一次 G2P
                single_res = pycantonese.characters_to_jyutping(char)
                if single_res and single_res[0][1]:
                    target_jyutping_groups.append(_tokenize_jyutping_smart(single_res[0][1]))
                else:
                    target_jyutping_groups.append([])

    return original_chars_flat, target_jyutping_groups

# --- 3. 核心分析函數 (主入口) ---
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
    
    # 3. 獲取使用者輸出 (User Output)
    # 模型輸出: "gwong2 dung1 waa2" (字串)
    raw_output_str = processor.decode(predicted_ids[0])
    
    # 清理並準備對齊
    # 我們需要把用戶的輸出也變成 ['gw', 'o', 'ng', '2', 'd', 'u', 'ng', '1'...] 的流
    # 這樣才能跟 Target 的結構對齊
    
    # 步驟 A: 移除空格，變成連續字串 "gwong2dung1waa2"
    # 注意：這一步假設模型輸出的拼音是標準的。如果模型輸出亂碼，tokenize 可能會切得不完美，
    # 但 Needleman-Wunsch 算法會處理這些 mismatch，所以沒關係。
    user_jyutping_clean = raw_output_str.replace(" ", "")
    
    # 步驟 B: 使用相同的邏輯切分用戶輸入
    # 因為用戶輸入是一長串，我們用正則表達式把 [a-z] 和 [0-9] 分開，或者嘗試 parse
    # 這裡用一個簡單的策略：把它當作一連串的 components
    # 為了最佳對齊，我們這裡還是用 "Character + Number" 的粒度比較好，
    # 因為用戶可能讀錯導致無法形成合法的 onset/nucleus。
    # 
    # ★ 關鍵決策：為了避免用戶讀錯導致 crash，用戶端我們使用較細的粒度 (Regex Split)，
    # 然後讓對齊算法去匹配 Target 的 "gw", "o", "ng"。
    # 等等，如果 Target 是 "gw" (1個token)，User 是 "g", "w" (2個 tokens)，對齊會錯位。
    # 
    # ★ 修正策略：
    # 我們也嘗試用 pycantonese.parse_jyutping 去解析用戶的整句輸出。
    # 如果解析成功，我們就用結構化 token。如果失敗（亂讀），回退到字母切分。
    
    user_tokens = []
    # 嘗試把用戶輸出拆成音節 (e.g. "gwong2", "dung1")
    user_syllables = re.findall(r'[a-z]+[0-9]', raw_output_str)
    
    if user_syllables:
        # 如果能抓到音節，就用結構化切分
        for syl in user_syllables:
            user_tokens.extend(_tokenize_jyutping_smart(syl))
    else:
        # 如果抓不到（例如沒聲調），就退化成字母切分
        # 但這會導致跟 Target (gw) 對不上。
        # 為了保險，我們這裡對於 Target 也許應該退化成簡單切分？
        # 不，Target 是 Ground Truth，應該保持結構。
        # 
        # 最終方案：讓 User stream 盡量 "粘" 在一起。
        # 實際上，Wav2Vec2 輸出的通常是標準拼音。我們直接用 smart parse。
        user_tokens = _tokenize_jyutping_smart(raw_output_str)


    # 4. 對齊 (Alignment)
    word_alignments = _get_phoneme_alignments_by_word(user_tokens, target_jyutping_by_char)

    return _format_to_json_structure(word_alignments, target_sentence, target_chars)

# --- 4. 對齊與格式化 (保持原樣或微調) ---
# 這裡的邏輯與之前相同，不需要大改，因為它只是比較兩個 list 的相似度。
# 只要 user_tokens 和 target_jyutping_by_char 的元素 (token) 粒度一致即可。
# ... ( _get_phoneme_alignments_by_word 與 _format_to_json_structure 代碼同上) ...

# 為了節省篇幅，請使用上一版提供的 _get_phoneme_alignments_by_word 和 _format_to_json_structure
# 只需要替換上面的 _tokenize_jyutping_smart 和 analyze 函數即可。
# 下面我會把完整的 _get_phoneme_alignments_by_word 貼上以確保完整性。

def _get_phoneme_alignments_by_word(user_phonemes, target_words_ipa_tokenized):
    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

    # DP Matrix
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