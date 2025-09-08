import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from phonemizer import phonemize
import numpy as np
from datetime import datetime
from colorama import init, Fore, Style

# 初始化 colorama
init(autoreset=True)

# --- 1. 全域設定 ---
TARGET_SENTENCE = "how was your day"
AUDIO_FILE_PATH = "./TestAudio/hello.wav"
MODEL_NAME = "MultiBridge/wav2vec-LnNor-IPA-ft"
MODEL_SAVE_PATH = "./ASRs/MultiBridge-wav2vec-LnNor-IPA-ft-local"

# --- 2. 載入模型和處理器 (保持不變) ---
print(f"正在準備模型 '{MODEL_NAME}'...")
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
    print("模型和處理器載入成功！")
except Exception as e:
    print(f"處理或載入模型時發生錯誤: {e}")
    exit()

# --- 3. 準備目標音標 (Target) - (已修改) ---
print("正在準備目標音標...")
# 在這一步就徹底移除重音符號，得到最乾淨的目標音標列表
target_ipa_by_word = [
    word.replace('ˌ', '').replace('ˈ', '').replace('ː', '')
    for word in phonemize(TARGET_SENTENCE, language='en-us', backend='espeak', with_stress=True).split()
]

# --- 4. 讀取音訊並進行辨識 (保持不變) ---
print(f"正在讀取音訊檔案: {AUDIO_FILE_PATH}...")
try:
    speech, sample_rate = sf.read(AUDIO_FILE_PATH)
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
except Exception as e:
    print(f"讀取或處理音訊時發生錯誤: {e}")
    exit()
print("正在辨識用戶的實際發音...")
input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
with torch.no_grad():
  logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
user_ipa_full = processor.decode(predicted_ids[0])


# --- 5. 核心函式：現在處理的都是乾淨的音標，邏輯保持不變 ---
def get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa):
    user_phonemes = list(user_phoneme_str.replace(' ', ''))
    target_phonemes_flat = []
    word_boundaries = []
    current_idx = 0
    for word_ipa in target_words_ipa:
        phonemes = list(word_ipa) # 已經是乾淨的音標
        target_phonemes_flat.extend(phonemes)
        current_idx += len(phonemes)
        word_boundaries.append(current_idx)

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
    word_start_idx = 0
    target_phoneme_count = 0
    
    for i, phoneme in enumerate(target_path):
        if phoneme != '-':
            target_phoneme_count += 1
        
        if target_phoneme_count in word_boundaries:
            target_alignment = target_path[word_start_idx:i+1]
            user_alignment = user_path[word_start_idx:i+1]
            alignments_by_word.append({
                "target": target_alignment,
                "user": user_alignment
            })
            word_start_idx = i + 1
            
    return alignments_by_word

# --- 6. 格式化輸出函式 (已簡化) ---
def format_and_print_final_report(alignments):
    total_phonemes = 0
    total_errors = 0
    correct_words = 0
    
    target_line_parts = []
    user_line_parts = []

    for alignment in alignments:
        word_is_correct = True
        
        max_lens = [max(len(t), len(u)) for t, u in zip(alignment['target'], alignment['user'])]
        
        target_word_parts = [p.ljust(max_lens[j]) for j, p in enumerate(alignment['target'])]
        target_line_parts.append(f"[ {' '.join(target_word_parts)} ]")
        
        user_word_parts = []
        for j, user_phoneme in enumerate(alignment['user']):
            target_phoneme = alignment['target'][j]
            is_match = (user_phoneme == target_phoneme)
            
            if not is_match:
                word_is_correct = False
                if user_phoneme != '-' and target_phoneme != '-': # 替換
                    total_errors += 1
                elif user_phoneme == '-': # 省略
                    total_errors += 1
                else: # 插入
                    total_errors += 1
            
            color = Fore.GREEN if is_match else Fore.RED
            user_word_parts.append(f"{color}{user_phoneme.ljust(max_lens[j])}{Style.RESET_ALL}")
        
        user_line_parts.append(f"[ {' '.join(user_word_parts)} ]")
        
        if word_is_correct:
            correct_words += 1
        
        total_phonemes += sum(1 for p in alignment['target'] if p != '-')

    # --- 計算統計資料 ---
    total_words = len(alignments)
    incorrect_words = total_words - correct_words
    overall_score = (correct_words / total_words) * 100 if total_words > 0 else 0
    phoneme_error_rate = (total_errors / total_phonemes) * 100 if total_phonemes > 0 else 0

    # --- 列印報告 ---
    separator = "="*70
    print("\n" + separator)
    print("Pronunciation Analysis".center(70))
    print(separator + "\n")

    print(f"Sentence: {TARGET_SENTENCE}\n")
    print(f"Target  : {' '.join(target_line_parts)}")
    print(f"User    : {' '.join(user_line_parts)}")

    print("\n" + "-" * 70)
    print("[ Summary ]")
    print("-" * 70)
    print(f"- Overall Score:         {overall_score:.1f}%")
    print(f"- Total Words:           {total_words}")
    print(f"- Correct Words:         {correct_words}")
    print(f"- Incorrect Words:       {incorrect_words}")
    print(f"- Phoneme Error Rate:    {phoneme_error_rate:.2f}% ({total_errors} errors in {total_phonemes} target phonemes)")
    # (已修改) 使用 UTC 時間
    print(f"- Analysis Timestamp:    {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
    
    print("\n" + separator)


# --- 主流程 ---
print("正在進行音素級對齊...")
word_alignments = get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)

format_and_print_final_report(word_alignments)
