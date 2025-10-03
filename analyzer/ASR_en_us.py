# ASR_en_us.py

import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime, timezone

# 【【【【【 新增程式碼 #1：自動檢測可用設備 】】】】】
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_fr_fr.py is configured to use device: {DEVICE}")

# --- 1. 全域設定與模型載入函數 (保持不變) ---
MODEL_NAME = "MultiBridge/wav2vec-LnNor-IPA-ft"

processor = None
model = None

def load_model():
    """
    (方案 A) 讓 transformers 自動處理模型的下載、快取和加載。
    它會自動使用 Dockerfile 中設定的 HF_HOME 環境變數。
    """
    global processor, model
    if processor and model:
        print(f"模型 '{MODEL_NAME}' 已載入，跳過。")
        return True

    print(f"正在準備 ASR 模型 '{MODEL_NAME}'...")
    print(f"Transformers 將自動在 HF_HOME 指定的快取中尋找或下載。")
    try:
        # 直接使用模型的線上名稱調用 from_pretrained
        # 這就是魔法發生的地方！
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        
        model.to(DEVICE)
        print(f"模型 '{MODEL_NAME}' 和處理器載入成功！")
        return True
    except Exception as e:
        print(f"處理或載入模型 '{MODEL_NAME}' 時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

# --- 2. 智能 IPA 切分函數 (已更新) ---
# 移除了包含 'ː' 的組合，因為我們將在源頭移除它
MULTI_CHAR_PHONEMES = {
    'tʃ', 'dʒ', # 輔音 (Affricates)
    'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ', # 雙元音 (Diphthongs)
    'ɪə', 'eə', 'ʊə', 'ər' # R-controlled 和其他組合
}

def _tokenize_ipa(ipa_string: str) -> list:
    """
    將 IPA 字串智能地切分為音素列表，能正確處理多字元音素。
    """
    phonemes = []
    i = 0
    s = ipa_string.replace(' ', '')
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in MULTI_CHAR_PHONEMES:
            phonemes.append(s[i:i+2])
            i += 2
        else:
            phonemes.append(s[i])
            i += 1
    return phonemes

# --- 3. 核心分析函數 (主入口) (已修改) ---
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    這是此模組的主要進入點。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請確保在呼叫 analyze 之前已成功執行 load_model()。")

    # =========================================================================
    #  第一步：執行您現有的、未經修改的完整分析流程
    # =========================================================================
    target_ipa_by_word_str = phonemize(target_sentence, language='en-us', backend='espeak', with_stress=True, strip=True).split()
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˌ', '').replace('ˈ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]
    target_words_original = target_sentence.split()

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
    
    # 這是您原始程式碼的流程，我們先獲取不帶時間戳的辨識結果
    user_ipa_full = processor.decode(predicted_ids[0])

    # 使用您原始的對齊函數
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

    # 使用您原始的格式化函數，產生不含時間戳的初始結果
    initial_result = _format_to_json_structure(word_alignments, target_sentence, target_words_original)


    # =========================================================================
    #  第二步：獲取帶時間戳的辨識結果，用於後續的「注入」
    # =========================================================================
    transcription_with_offsets = processor.batch_decode(
        predicted_ids, 
        output_char_offsets=True
    )
    
    char_offsets = transcription_with_offsets.char_offsets[0]
    
    # 建立一個從辨識出的音素到其時間戳的映射字典
    # 鍵是音素，值是時間戳物件列表（因為同一個音素可能出現多次）
    phoneme_to_ts_map = {}
    for offset in char_offsets:
        phoneme = offset['char']
        if phoneme not in phoneme_to_ts_map:
            phoneme_to_ts_map[phoneme] = []
        
        phoneme_to_ts_map[phoneme].append({
            "start_time": round(offset['start_offset'] * model.config.inputs_to_logits_ratio / 16000.0, 2),
            "end_time": round(offset['end_offset'] * model.config.inputs_to_logits_ratio / 16000.0, 2)
        })

    # =========================================================================
    #  第三步：將時間戳「注入」到初始結果中，完成最終輸出
    # =========================================================================
    # 複製映射字典，以便在遍歷時可以安全地從中移除已使用的時間戳
    ts_map_copy = {k: v[:] for k, v in phoneme_to_ts_map.items()}

    for word_data in initial_result["words"]:
        word_start_time = None
        word_end_time = None
        
        for phoneme_data in word_data["phonemes"]:
            user_phoneme = phoneme_data["user"]
            
            # 預設時間戳為 null
            phoneme_data["startTime"] = None
            phoneme_data["endTime"] = None

            # 如果使用者發音不是'-'，且在時間戳映射中能找到
            if user_phoneme != '-' and user_phoneme in ts_map_copy and ts_map_copy[user_phoneme]:
                # 取出並移除第一個可用的時間戳（先進先出）
                ts = ts_map_copy[user_phoneme].pop(0)
                
                # 為音素注入時間戳
                phoneme_data["startTime"] = ts["start_time"]
                phoneme_data["endTime"] = ts["end_time"]
                
                # 更新單字的時間戳
                if word_start_time is None:
                    word_start_time = ts["start_time"]
                word_end_time = ts["end_time"] # 不斷更新為最後一個音素的結束時間
        
        # 為單字注入時間戳
        word_data["startTime"] = word_start_time
        word_data["endTime"] = word_end_time

    return initial_result



# --- 4. 對齊函數 (與上一版相同) ---
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

# --- 5. 格式化函數 (與上一版相同) ---
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
            # 確保這裡也移除 'ː'
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
