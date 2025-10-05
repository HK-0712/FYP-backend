# ASR_en_us.py (fixed & replace-with)
import torch
import soundfile as sf
import librosa
import os
import numpy as np
from datetime import datetime, timezone

from phonemizer import phonemize
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Optional: LM-assisted decoder (preferred for robust offsets)
try:
    from transformers import Wav2Vec2ProcessorWithLM
    HAS_WITH_LM = True
except Exception:
    HAS_WITH_LM = False

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: ASR_en_us.py is configured to use device: {DEVICE}")

# ---------- Global & model ----------
MODEL_NAME = "MultiBridge/wav2vec-LnNor-IPA-ft"
processor = None
processor_lm = None
model = None

def load_model():
    """
    載入模型與處理器：
    - 先載標準 Processor + 模型
    - 若可用，再載 LM Processor 以取得更穩定的 offsets
    """
    global processor, processor_lm, model

    if processor and model:
        print(f"模型 '{MODEL_NAME}' 已載入，跳過。")
        return True

    print(f"正在準備 ASR 模型 '{MODEL_NAME}'...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        model.to(DEVICE)

        if HAS_WITH_LM:
            try:
                processor_lm = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_NAME)
                print("LM 解碼器載入成功：將優先使用 logits + LM 取得 offsets。")
            except Exception as e:
                processor_lm = None
                print(f"LM 解碼器不可用（{e}），回退到標準解碼。")

        print(f"模型 '{MODEL_NAME}' 和處理器載入成功！")
        return True
    except Exception as e:
        print(f"載入模型/處理器時發生錯誤: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

# ---------- IPA multi-char handling ----------
MULTI_CHAR_PHONEMES = {
    'tʃ', 'dʒ',                 # Affricates
    'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ',  # Diphthongs
    'ɪə', 'eə', 'ʊə', 'ər'      # R-controlled & others
}

def _tokenize_ipa(ipa_string: str) -> list:
    """
    智能切分 IPA 字串為音素列表，處理多字元音素。
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

# ---------- Core analyze ----------
def analyze(audio_file_path: str, target_sentence: str) -> dict:
    """
    接收音訊檔案路徑和目標句子，回傳詳細的發音分析字典。
    修正：以 logits 取得 offsets，保留 CTC 時序；順序注入；多字元音素聚合；詞級時間回寫。
    """
    if not processor or not model:
        raise RuntimeError("模型尚未載入。請先呼叫 load_model()。")

    # 1) 目標 IPA 解析
    target_ipa_by_word_str = phonemize(
        target_sentence,
        language='en-us',
        backend='espeak',
        with_stress=True,
        strip=True
    ).split()

    # 去掉重音與長度符號
    target_ipa_by_word = [
        _tokenize_ipa(word.replace('ˌ', '').replace('ˈ', '').replace('ː', ''))
        for word in target_ipa_by_word_str
    ]
    target_words_original = target_sentence.split()

    # 2) 讀取與重取樣
    try:
        speech, sample_rate = sf.read(audio_file_path)
        if sample_rate != 16000:
            speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
    except Exception as e:
        raise IOError(f"讀取或處理音訊時發生錯誤: {e}")

    # 3) 前處理 & 模型推論
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)

    # 使用者 IPA（不含時間戳） + 對齊
    user_ipa_full = processor.decode(pred_ids[0])
    word_alignments = _get_phoneme_alignments_by_word(user_ipa_full, target_ipa_by_word)
    result = _format_to_json_structure(word_alignments, target_sentence, target_words_original)

    # 4) 取得 offsets（優先 logits+LM，否則回退）
    char_offsets = None
    if processor_lm is not None:
        try:
            lm_out = processor_lm.batch_decode(logits.cpu().numpy())
            if hasattr(lm_out, "char_offsets") and lm_out.char_offsets:
                char_offsets = lm_out.char_offsets[0]
        except Exception as e:
            print(f"LM 解碼 offsets 失敗，回退到標準。原因: {e}")

    if char_offsets is None:
        transcription_with_offsets = processor.batch_decode(
            pred_ids,
            output_char_offsets=True
        )
        char_offsets = transcription_with_offsets.char_offsets[0] if hasattr(transcription_with_offsets, "char_offsets") else []

    # 5) offsets 轉秒並按順序注入
    step_sec = (model.config.inputs_to_logits_ratio / float(sample_rate))  # 例如 320/16000=0.02s
    ts_seq = []
    for off in char_offsets:
        s = round(off.get('start_offset', None) * step_sec, 3) if off.get('start_offset', None) is not None else None
        e = round(off.get('end_offset', None) * step_sec, 3) if off.get('end_offset', None) is not None else None
        ts_seq.append({
            "char": off.get('char', ''),
            "start": s,
            "end": e
        })

    _inject_timestamps_in_order(result, ts_seq)

    # 6) 補上分析時間戳
    result["analysisTimestampUTC"] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S (UTC)')
    return result

# ---------- Alignment ----------
def _get_phoneme_alignments_by_word(user_phoneme_str, target_words_ipa_tokenized):
    """
    使用新的切分邏輯執行音素對齊：輸出 by-word 的 user/target 對齊路徑。
    """
    user_phonemes = _tokenize_ipa(user_phoneme_str)
    target_phonemes_flat = []
    word_boundaries_indices = []
    current_idx = 0
    for word_ipa_tokens in target_words_ipa_tokenized:
        target_phonemes_flat.extend(word_ipa_tokens)
        current_idx += len(word_ipa_tokens)
        word_boundaries_indices.append(current_idx - 1)

    # DP for edit distance
    dp = np.zeros((len(user_phonemes) + 1, len(target_phonemes_flat) + 1))
    for i in range(1, len(user_phonemes) + 1): dp[i][0] = i
    for j in range(1, len(target_phonemes_flat) + 1): dp[0][j] = j

    for i in range(1, len(user_phonemes) + 1):
        for j in range(1, len(target_phonemes_flat) + 1):
            cost = 0 if user_phonemes[i-1] == target_phonemes_flat[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

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
                user_alignment   = user_path[word_start_idx_in_path   : path_idx + 1]
                alignments_by_word.append({
                    "target": target_alignment,
                    "user": user_alignment
                })
                word_start_idx_in_path = path_idx + 1
            target_phoneme_counter_in_path += 1
    return alignments_by_word

# ---------- Formatting ----------
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

# ---------- Timestamp injection (new core) ----------
def _inject_timestamps_in_order(result_dict: dict, ts_seq: list):
    """
    以「順序」把時間戳注入到每個音素與詞：
    - 不用字串鍵映射，避免同符號多次出現造成錯位
    - 多字元 IPA 音素以相鄰 char 聚合其時間邊界
    - 寫回詞級 start/end；做基本數學一致性檢查
    """
    # 依序消耗 char offsets
    k = 0  # 指向 ts_seq
    total_ts = len(ts_seq)

    for word in result_dict["words"]:
        word_start = None
        word_end = None

        for p in word["phonemes"]:
            p_user = p.get("user", "-")
            # 預設
            p["startTime"] = None
            p["endTime"] = None

            if p_user == "-" or k >= total_ts:
                continue

            # 可能存在空白、分隔符等：跳過無效 char
            while k < total_ts and (ts_seq[k]["char"] is None or ts_seq[k]["char"] == ""):
                k += 1
                if k >= total_ts:
                    break
            if k >= total_ts:
                break

            # 精確匹配：下一個 char 等於整個音素
            if ts_seq[k]["char"] == p_user:
                s = ts_seq[k]["start"]; e = ts_seq[k]["end"]
                if _valid_ts_pair(s, e):
                    p["startTime"] = s; p["endTime"] = e
                    word_start = s if word_start is None else word_start
                    word_end = e
                k += 1
                continue

            # 多字元音素：嘗試聚合相鄰 char
            if len(p_user) > 1:
                agg_start = None
                agg_end = None
                consumed = 0
                buffer = ""

                while (k + consumed) < total_ts and len(buffer) < len(p_user):
                    cur_char = ts_seq[k + consumed]["char"] or ""
                    buffer += cur_char
                    ts_s = ts_seq[k + consumed]["start"]
                    ts_e = ts_seq[k + consumed]["end"]
                    if ts_s is not None:
                        agg_start = ts_s if agg_start is None else min(agg_start, ts_s)
                    if ts_e is not None:
                        agg_end = ts_e if agg_end is None else max(agg_end, ts_e)
                    consumed += 1
                    if buffer == p_user:
                        if _valid_ts_pair(agg_start, agg_end):
                            p["startTime"] = agg_start
                            p["endTime"] = agg_end
                            word_start = agg_start if word_start is None else word_start
                            word_end = agg_end
                        k += consumed
                        break
                # 若聚合失敗，不消耗 ts_seq，保留 None

            # 單字元但不相等：避免錯位，不消耗 ts_seq；保留 None

        # 詞級時間回寫（以該詞第一/最後一個有時間的音素為邊界）
        word["startTime"] = word_start
        word["endTime"] = word_end

    # 事後基本檢查：全局時間單調 & 音素不重疊
    _sanitize_monotonic_and_nonoverlap(result_dict)

def _valid_ts_pair(s, e):
    return (s is not None) and (e is not None) and (s <= e)

def _sanitize_monotonic_and_nonoverlap(result_dict: dict):
    """
    保證列表中各音素時間不回退、不重疊（允許等邊界接觸），
    並限制到非負與合理的浮點小數三位。
    """
    last_end = None
    for w in result_dict.get("words", []):
        w_start = None
        w_end = None
        for p in w.get("phonemes", []):
            s = p.get("startTime", None)
            e = p.get("endTime", None)
            if s is None or e is None:
                continue
            # 不重疊：若 s < last_end，則把 s 夾到 last_end
            if last_end is not None and s < last_end:
                s = last_end
            # 非負與單調
            if s < 0:
                s = 0.0
            if e < s:
                e = s
            # 四捨五入到 3 位
            p["startTime"] = round(float(s), 3)
            p["endTime"] = round(float(e), 3)
            last_end = p["endTime"]

            # 詞級邊界更新
            w_start = p["startTime"] if w_start is None else w_start
            w_end = p["endTime"]

        # 回寫詞級
        w["startTime"] = w_start if w_start is not None else w.get("startTime", None)
        w["endTime"] = w_end if w_end is not None else w.get("endTime", None)