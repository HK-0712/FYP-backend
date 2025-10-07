# ASR_en_us_v3.py

import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone

# --- 全域設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_en_us_v3.py is configured to use device: {DEVICE}")

# 【【【【【 關鍵修改 #1：更新為最終選定的模型名稱 】】】】】
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

processor = None
model = None

# 【【【【【 新增程式碼 #1：IPA 淨化器相關的字典 】】】】】

# 步驟 1a：定義一個權威的、我們認可的「標準美式英語 IPA 符號集」
# 這個集合是我們的「白名單」
VALID_ENGLISH_IPA = {
    # 元音 (Vowels)
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɐ', 'ᵻ',
    # R音化元音 (R-colored Vowels)
    'ɚ', 'ɝ',
    # 雙元音 (Diphthongs)
    'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'iə', 'eə', 'ʊə', 'ɛɹ', 'ɪɹ', 'ʊɹ', 'aɪɚ', 'aɪə',
    # 輔音 (Consonants)
    'p', 'b', 't', 'd', 'k', 'ɡ', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'ɹ', 'w', 'j',
    # 塞擦音 (Affricates)
    'tʃ', 'dʒ',
    # 其他常見變體
    'ɾ', 'ʔ', 'ɫ', 'n̩', 'l̩', 'r̩'
}

# 步驟 1b：建立「外語到英語」的映射規則字典
# 這是我們的「重點觀察名單」或「黑名單轉換規則」
NON_ENGLISH_TO_ENGLISH_MAP = {
    # 歐洲語言常見變體
    'ʁ': 'ɹ', 'r': 'ɹ', 'β': 'v', 'x': 'h', 'ɣ': 'ɡ', 'ç': 'h', 'y': 'i', 'ø': 'e', 'œ': 'ɛ', 'ɒ': 'ɑ', 'əʊ': 'oʊ',
    # 鼻化元音 (去掉鼻化)
    'ɑ̃': 'ɑ', 'ɔ̃': 'ɔ', 'ɛ̃': 'ɛ', 'œ̃': 'ɛ', 'ɐ̃': 'ɐ', 'õ': 'o', 'ĩ': 'i', 'ũ': 'u',
    # 亞洲/斯拉夫語系常見音 (映射到最接近的英語音)
    'ɕ': 'ʃ', 'tɕ': 'tʃ', 'ʂ': 'ʃ', 'ʐ': 'ʒ', 'dʑ': 'dʒ',
    # 印地語捲舌音 (去掉捲舌特徵)
    'ʈ': 't', 'ɖ': 'd', 'ɳ': 'n', 'ɭ': 'l', 'ɽ': 'ɾ',
    # 阿拉伯語系音
    'ʕ': 'ʔ', 'ħ': 'h', 'q': 'k',
    # 其他...
    'ʎ': 'j', 'ɲ': 'n', 'ʋ': 'v', 'c': 'k', 'ɟ': 'ɡ', 'ɸ': 'f', 'χ': 'h',
}

def load_model():
    """
    載入 Facebook 的 Wav2Vec2 espeak ASR 模型。
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

# 【【【【【 新增程式碼 #2：IPA 淨化器函式 】】】】】
def purify_ipa_sequence(raw_phonemes: list) -> list:
    """
    將一個可能包含外語 IPA 的音素序列，淨化為只包含標準英語 IPA 的序列。
    """
    purified_phonemes = []
    for phoneme in raw_phonemes:
        if not phoneme:  # 跳過空字串
            continue

        # 1. 如果音素本身就是合法的英語 IPA，直接接受
        if phoneme in VALID_ENGLISH_IPA:
            purified_phonemes.append(phoneme)
            continue
            
        # 2. 如果音素在我們的映射字典中，進行替換
        if phoneme in NON_ENGLISH_TO_ENGLISH_MAP:
            replacement = NON_ENGLISH_TO_ENGLISH_MAP[phoneme]
            purified_phonemes.append(replacement)
            # print(f"INFO: Replaced non-English IPA '{phoneme}' with '{replacement}'.") # 可選的日誌
            continue

        # 3. 處理帶有附加符號的音素 (例如長音 'ː', 顎化 'ʲ')
        # 簡化處理：直接去掉附加符號，看剩下的部分是否合法
        base_phoneme = phoneme.replace('ː', '').replace('ʲ', '').replace('ʰ', '')
        if base_phoneme in VALID_ENGLISH_IPA:
            purified_phonemes.append(base_phoneme)
            # print(f"INFO: Stripped diacritics from '{phoneme}' to '{base_phoneme}'.") # 可選的日誌
            continue

        # 4. 如果經過以上所有步驟仍然無法識別，作為最後手段，忽略該音素
        # print(f"WARNING: Unknown IPA phoneme '{phoneme}' encountered and was ignored.") # 可選的日誌
        
    return purified_phonemes

# --- 2. 智能 IPA 切分函數 (與您的原版邏輯完全相同) ---
MULTI_CHAR_PHONEMES = {
    'tʃ', 'dʒ', 'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ', 'ɪə', 'eə', 'ʊə', 'ər',
    # 為 Facebook 模型輸出新增的組合
    'ɑː', 'iː', 'uː', 'ɔː', 'ɜː', 'oː', 'eː', 'yː', 'øː', 'œː', 'ɛː', 'æː',
    'ɑːɹ', 'ɔːɹ', 'oːɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ', 'aɪɚ', 'aɪə'
}

def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，能正確處理多字元音素。
    """
    phonemes = []
    i = 0
    s = ipa_string.replace(' ', '')
    while i < len(s):
        # 優先檢查三個字符的組合 (例如 ɑːɹ)
        if i + 2 < len(s) and s[i:i+3] in MULTI_CHAR_PHONEMES:
            phonemes.append(s[i:i+3])
            i += 3
        # 再檢查兩個字符的組合
        elif i + 1 < len(s) and s[i:i+2] in MULTI_CHAR_PHONEMES:
            phonemes.append(s[i:i+2])
            i += 2
        else:
            phonemes.append(s[i])
            i += 1
    return phonemes

# --- 3. 核心分析函數 (主入口) (已修改以整合淨化器) ---
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # 步驟 1：獲取目標 IPA (與原版邏輯相同)
    target_ipa_by_word_str = phonemize(target_sentence, language='en-us', backend='espeak', with_stress=True, strip=True).split()
    
    # 【【【【【 關鍵修改 #2：完全遵循您對目標 IPA 的清理邏輯 】】】】】
    # 在切分前，移除所有重音和長音符號
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˌ', '').replace('ˈ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]
    target_words_original = target_sentence.split()

    # 步驟 2：讀取和重採樣音訊 (與原版邏輯相同)
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")
    
    # 步驟 3：使用 Wav2Vec2 模型進行預測
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # 步驟 4：解碼得到原始的、可能混雜的音素序列
    raw_user_ipa_str = processor.batch_decode(predicted_ids[0])[0]
    raw_user_phonemes = raw_user_ipa_str.split(' ')

    # 【【【【【 關鍵修改 #3：在此處插入淨化步驟 】】】】】
    purified_user_phonemes = purify_ipa_sequence(raw_user_phonemes)
    user_ipa_full = "".join(purified_user_phonemes)

    # 步驟 5：使用淨化後的 IPA 進行音素對齊 (後續邏輯與原版完全相同)
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
            # 【【【【【 關鍵修改 #4：完全遵循您對遺漏單詞的清理邏輯 】】】】】
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
