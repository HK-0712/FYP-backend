# =======================================================================
# 1. 匯入區 (Imports)
#    - 與英文版完全相同，因為我們使用相同的工具鏈。
# =======================================================================
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
# 2. 全域變數與配置區 (Global Variables & Config)
# =======================================================================
# 自動檢測可用設備
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_pt_br.py is configured to use device: {DEVICE}")

# 【【【【【 關鍵修改 1：設定為葡萄牙語 ASR 模型 】】】】】
MODEL_NAME = "caiocrocha/wav2vec2-large-xlsr-53-phoneme-portuguese"

processor = None
model = None

# =======================================================================
# 3. 核心業務邏輯區 (Core Business Logic)
# =======================================================================

# -----------------------------------------------------------------------
# 3.1. 模型載入函數
#      - 與英文版邏輯完全相同，僅替換模型名稱。
# -----------------------------------------------------------------------
def load_model():
    """
    載入葡萄牙語 ASR 模型和對應的處理器。
    """
    global processor, model
    if processor and model:
        print(f"模型 '{MODEL_NAME}' 已載入，跳過。")
        return True

    print(f"正在準備 ASR 模型 '{MODEL_NAME}'...")
    try:
        # 這些模型通常使用標準的 Wav2Vec2Processor 和 Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        print(f"模型 '{MODEL_NAME}' 和處理器載入成功！")
        return True
    except Exception as e:
        print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

# -----------------------------------------------------------------------
# 3.2. 智能 IPA 切分函數
#      - 【關鍵修改 2】針對葡萄牙語的 IPA 特性進行調整。
# -----------------------------------------------------------------------
def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表。
    這個版本能處理葡萄牙語中常見的多字元音素和帶有附加符號的音素。
    """
    phonemes = []
    # 移除所有由 phonemizer 產生的多餘空格
    s = ipa_string.replace(' ', '')
    i = 0
    while i < len(s):
        # 檢查葡萄牙語中常見的雙字元塞擦音
        if i + 1 < len(s) and s[i:i+2] in {'dʒ', 'tʃ'}:
            phonemes.append(s[i:i+2])
            i += 2
            continue

        # 處理帶有鼻化符 (波浪號) 的元音
        # unicodedata.category(char) == 'Mn' 用於檢測非間距標記 (例如波浪號)
        current_char = s[i]
        i += 1
        while i < len(s) and unicodedata.category(s[i]) == 'Mn':
            current_char += s[i]
            i += 1
        phonemes.append(current_char)
        
    return phonemes

# -----------------------------------------------------------------------
# 3.3. 核心分析函數 (主入口)
#      - 【關鍵修改 3】將 G2P 語言設定為 'pt-br'。
# -----------------------------------------------------------------------
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標葡萄牙語句子，回傳詳細的發音分析字典。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # --- G2P 步驟 ---
    # 1. 使用正則表達式來準確地分割單詞，並自動忽略標點符號
    target_words_original = re.findall(r"[\w'-]+", target_sentence)
    # 2. 將分割好的、乾淨的單詞重新組合，再傳給 phonemizer
    cleaned_sentence = " ".join(target_words_original)
    
    # 3. 呼叫 phonemizer，並將語言設定為 'pt-br' (巴西葡萄牙語)
    target_ipa_by_word_str = phonemize(
        cleaned_sentence,
        language='pt-br',
        backend='espeak',
        with_stress=True, # 保留重音符號以便後續處理
        strip=True
    ).split()

    # 4. 確保單詞列表和音素列表的長度一致，以防 G2P 工具出錯
    if len(target_words_original) != len(target_ipa_by_word_str):
        print(f"警告：單詞數量 ({len(target_words_original)}) 與 G2P 結果數量 ({len(target_ipa_by_word_str)}) 不匹配。將進行截斷處理。")
        min_len = min(len(target_words_original), len(target_ipa_by_word_str))
        target_words_original = target_words_original[:min_len]
        target_ipa_by_word_str = target_ipa_by_word_str[:min_len]

    # 5. 清理 G2P 輸出的音素，並使用我們為葡萄牙語定製的切分函數
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˈ', '').replace('ˌ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]

    # --- ASR 步驟 ---
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if len(speech) == 0:
            print("警告: 音訊檔案為空。")
            user_ipa_full = ""
        else:
            if sample_rate != 16000:
                speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
            
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to(DEVICE)
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            # 解碼後，移除模型可能產生的分隔符 '|'
            user_ipa_full = processor.decode(predicted_ids[0]).replace('|', '')

    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # --- 對齊與格式化步驟 (與英文版邏輯完全相同) ---
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)
    return _format_to_json_structure(word_alignments, target_sentence, target_words_original)

# =======================================================================
# 4. 對齊與格式化函數區 (Alignment & Formatting)
#    - 【注意】這些函數是語言無關的，直接從英文版複製而來，無需修改。
# =======================================================================

# -----------------------------------------------------------------------
# 4.1. 對齊函數 (語言無關)
# -----------------------------------------------------------------------
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用動態規劃執行音素對齊。此函數是語言無關的。
    """
    # 對於 ASR 的輸出，我們也使用相同的、更通用的切分函數
    user_phonemes = _tokenize_ipa(user_phoneme_str)
    
    target_phonemes_flat = [p for word in target_words_ipa_tokenized for p in word]
    
    # 如果目標音素為空 (例如，輸入句子只有標點符號)，返回空對齊
    if not target_phonemes_flat:
        return []
        
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

# -----------------------------------------------------------------------
# 4.2. 格式化函數 (語言無關)
# -----------------------------------------------------------------------
def _format_to_json_structure(alignments, sentence, original_words) -> dict:
    """
    將對齊結果格式化為最終的 JSON 結構。此函數是語言無關的。
    """
    total_phonemes, total_errors, correct_words_count = 0, 0, 0
    words_data = []
    num_words_to_process = min(len(alignments), len(original_words))

    for i in range(num_words_to_process):
        alignment = alignments[i]
        word_is_correct = True
        phonemes_data = []
        
        # 增加一個健壯性檢查，以防對齊演算法返回長度不一的列表
        min_len = min(len(alignment.get('target', [])), len(alignment.get('user', [])))
        for j in range(min_len):
            target_phoneme, user_phoneme = alignment['target'][j], alignment['user'][j]
            is_match = (user_phoneme == target_phoneme)
            phonemes_data.append({"target": target_phoneme, "user": user_phoneme, "isMatch": is_match})
            if not is_match:
                word_is_correct = False
                if not (user_phoneme == '-' and target_phoneme == '-'): total_errors += 1
        
        if word_is_correct and min_len > 0: correct_words_count += 1
        
        words_data.append({"word": original_words[i], "isCorrect": word_is_correct, "phonemes": phonemes_data})
        total_phonemes += sum(1 for p in alignment.get('target', []) if p != '-')

    # 【Fuse Logic】處理使用者漏講了單詞的情況
    if len(alignments) < len(original_words):
        for i in range(len(alignments), len(original_words)):
            # 【關鍵修改 4】確保這裡也使用 'pt-br'
            missed_word_ipa_str = phonemize(original_words[i], language='pt-br', backend='espeak', strip=True).replace('ː', '')
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
            "overallScore": round(overall_score, 1),
            "totalWords": total_words,
            "correctWords": correct_words_count,
            "phonemeErrorRate": round(phoneme_error_rate, 2),
            "total_errors": total_errors,
            "total_target_phonemes": total_phonemes
        },
        "words": words_data
    }
