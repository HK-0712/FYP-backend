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

# --- 1. 輔助函數：粵拼切分器 ---
def _tokenize_jyutping_components(jyutping_str: str) -> list:
    """
    將單個粵拼音節（如 'gwong2'）切分為字元列表（如 ['g', 'w', 'o', 'n', 'g', '2']）。
    這是為了模擬 IPA 的 'phoneme' 粒度，讓評分能指出具體是聲母、韻母還是聲調錯了。
    """
    # 移除可能多餘的空格
    s = jyutping_str.strip()
    # 直接拆分為單個字符，這是最簡單且有效的 "音素" 級別對比
    return list(s)

# --- 2. 智慧 G2P 歸屬邏輯 (中文版) ---
def _get_target_jyutping_by_char(sentence: str) -> (list, list):
    """
    將中文句子轉換為「字」級別的粵拼目標。
    邏輯：
    1. 使用 pycantonese 進行分詞與標音 (考慮變調)。
    2. 將分詞結果（如 '蛋糕' -> 'daan6gou1'）拆解回單字（'蛋'->'daan6', '糕'->'gou1'）。
    3. 回傳 (原始字列表, 每個字的粵拼 component 列表)。
    """
    # pycantonese 回傳格式範例: [('廣東話', 'gwong2dung1waa2'), ('好', 'hou2'), ('難', 'naan4')]
    segmented_result = pycantonese.characters_to_jyutping(sentence)
    
    original_chars_flat = []
    target_jyutping_groups = []

    jyutping_pattern = re.compile(r'([a-z]+[1-6])') # 匹配標準粵拼 (字母+數字)

    for word_segment, jyutping_segment in segmented_result:
        # 如果是標點符號或無讀音字符，pycantonese 可能回傳 None 或原字符
        if not jyutping_segment:
             # 對於標點，我們暫時忽略或保留，這裡選擇忽略以專注於發音
            continue

        # 找出該詞段中包含的所有粵拼音節
        syllables = jyutping_pattern.findall(jyutping_segment)
        
        # 簡單驗證：音節數應該等於漢字數
        # 注意：這在極少數多音字或特殊情況下可能不完美，但對絕大多數情況適用
        if len(word_segment) == len(syllables):
            for char, syl in zip(word_segment, syllables):
                original_chars_flat.append(char)
                # 將該字的粵拼拆成 components (e.g. "d", "o", "n", "g", "2")
                target_jyutping_groups.append(_tokenize_jyutping_components(syl))
        else:
            # 發生長度不匹配（罕見），回退策略：直接把整個詞當作一個單位，或跳過
            print(f"WARNING: Word segment '{word_segment}' length does not match Jyutping syllables '{jyutping_segment}'. alignment might be off.")
            # 盡力嘗試逐字對應
            for i, char in enumerate(word_segment):
                original_chars_flat.append(char)
                if i < len(syllables):
                    target_jyutping_groups.append(_tokenize_jyutping_components(syllables[i]))
                else:
                    target_jyutping_groups.append([]) # 無法對應

    return original_chars_flat, target_jyutping_groups

# --- 3. 核心分析函數 (主入口) ---
def analyze(audio_file_path: str, target_sentence: str, cache: dict = {}) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    """
    # 檢查快取中是否已有模型，如果沒有則載入
    if "model" not in cache:
        print(f"Cache miss (ASR_zh_hk). Loading model '{MODEL_NAME}'...")
        try:
            # 不需要顯式傳遞 token，依賴環境變數或 Hugging Face Space 登入狀態
            cache["processor"] = AutoProcessor.from_pretrained(MODEL_NAME)
            model = AutoModelForCTC.from_pretrained(MODEL_NAME)
            
            # 【【【 CPU 加速優化 】】】
            if DEVICE == "cpu":
                print("⚠️ CPU environment detected. Applying dynamic quantization to boost speed...")
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            model.to(DEVICE)
            cache["model"] = model
            print(f"Model '{MODEL_NAME}' loaded and cached.")
        except Exception as e:
            print(f"Error loading model '{MODEL_NAME}': {e}")
            raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

    processor = cache["processor"]
    model = cache["model"]

    # 1. 準備目標發音 (Ground Truth)
    # target_chars: ['檔', '案']
    # target_jyutping_by_char: [['d','o','n','g','2'], ['o','n','3']]
    target_chars, target_jyutping_by_char = _get_target_jyutping_by_char(target_sentence)
    
    # 2. 處理音訊與模型推理
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"Error processing audio: {e}")
    
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    # 如果是量化模型(CPU)，不需要 input.to(device)
    if DEVICE == "cuda":
        input_values = input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # 3. 獲取使用者發音 (Model Output)
    # 模型輸出通常是 "dong2 aan3" 這樣的字串
    raw_output_str = processor.decode(predicted_ids[0])
    
    # 將輸出字串清理並轉為連續的 components 列表
    # 例如 "dong2 aan3" -> "dong2aan3" -> ['d','o','n','g','2','a','a','n','3']
    # 這樣做是為了讓對齊算法能在整個句子層面上找到最佳匹配
    user_jyutping_full_str = raw_output_str.replace(" ", "")
    
    # 4. 執行對齊
    word_alignments = _get_phoneme_alignments_by_word(user_jyutping_full_str, target_jyutping_by_char)

    # 5. 格式化輸出
    return _format_to_json_structure(word_alignments, target_sentence, target_chars)


# --- 4. 對齊函數 (通用邏輯，適配 Jyutping components) ---
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用動態規劃 (Needleman-Wunsch) 對齊使用者發音與目標發音。
    這裡的 "phoneme" 對於粵語來說就是 Jyutping 的單個字符 (字母或數字)。
    """
    # 將使用者字串轉為列表: "dong2" -> ['d','o','n','g','2']
    user_phonemes = list(user_phoneme_str)
    
    target_phonemes_flat = []
    word_boundaries_indices = [] 
    current_idx = 0
    
    # 展平目標發音以便進行全局對齊
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

    # DP 初始化
    dp = np.zeros((len(user_phonemes) + 1, len(target_phonemes_flat) + 1))
    for i in range(1, len(user_phonemes) + 1): dp[i][0] = i
    for j in range(1, len(target_phonemes_flat) + 1): dp[0][j] = j
    
    # 填充 DP 表
    for i in range(1, len(user_phonemes) + 1):
        for j in range(1, len(target_phonemes_flat) + 1):
            cost = 0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    # 回溯 (Backtracking) 找最佳路徑
    i, j = len(user_phonemes), len(target_phonemes_flat)
    user_path, target_path = [], []
    while i > 0 or j > 0:
        cost = float('inf') if i == 0 or j == 0 else (0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1)
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + cost:
            user_path.insert(0, user_phonemes[i-1])
            target_path.insert(0, target_phonemes_flat[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            user_path.insert(0, user_phonemes[i-1])
            target_path.insert(0, '-')
            i -= 1
        else:
            user_path.insert(0, '-')
            target_path.insert(0, target_phonemes_flat[j-1])
            j -= 1
    
    # 根據單字邊界切分對齊結果
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
    
    # 處理最後一個字（如果有的話）
    if word_start_idx_in_path < len(target_path) and current_word_idx < num_words_to_align:
        target_alignment = target_path[word_start_idx_in_path:]
        user_alignment = user_path[word_start_idx_in_path:]
        alignments_by_word.append({
            "target": target_alignment,
            "user": user_alignment
        })

    return alignments_by_word


# --- 5. 格式化函數 (JSON Output) ---
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
                
                phonemes_data.append({
                    "target": target_p,
                    "user": user_p,
                    "isMatch": is_match
                })
                
                if not is_match:
                    word_is_correct = False
                    if not (user_p == '-' and target_p == '-'):
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
    # 處理漏讀的字
    if len(words_data) < total_words:
        # 需要計算剩餘字的預期 phonemes
        _, remaining_targets = _get_target_jyutping_by_char("".join(original_words[len(words_data):]))
        
        for i, target_group in enumerate(remaining_targets):
            current_word_idx = len(words_data)
            phonemes_data = []
            for p_char in target_group:
                phonemes_data.append({"target": p_char, "user": "-", "isMatch": False})
                total_errors += 1
                total_phonemes += 1
            
            words_data.append({
                "word": original_words[current_word_idx],
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