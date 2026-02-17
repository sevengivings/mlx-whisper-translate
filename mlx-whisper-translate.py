import os
import shutil
import argparse
import mlx_whisper
from mlx_lm import load, generate
import datetime
import time
import re
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ==========================================
# 1. 설정 (이 부분만 수정하면 됩니다)
# ==========================================
INPUT_PATH = "./target_files"  # 변환할 대상(폴더 또는 파일 1개)
EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".mkv") # 처리할 확장자

WHISPER_MODEL = "mlx-community/whisper-medium"
TRANSLATE_MODEL = "mlx-community/translategemma-4b-it-4bit"
TRANSLATE_MAX_TOKENS = 100
DEFAULT_WHISPER_LANGUAGE = "ja"
DEFAULT_TARGET_LANGUAGE = "ko"
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_BATCH_SIZE = 1
REPETITIVE_FILTER_MIN_CHARS = 40
REPETITIVE_FILTER_MIN_REPEATS = 12

# Whisper 옵션 (속도 우선 프리셋)
WHISPER_OPTIONS_FAST = {
    "word_timestamps": False,                 # 단어 타임스탬프 비활성화로 속도 개선
    "condition_on_previous_text": False,      # 문맥 연결 비활성화로 속도 개선
    "temperature": 0.0,                       # 단일 디코딩 패스
    "no_speech_threshold": 0.45,
}

# Whisper 옵션 (정확도 우선 프리셋)
WHISPER_OPTIONS_ACCURATE = {
    "word_timestamps": True,                  # 단어 단위 타임스탬프
    "condition_on_previous_text": True,       # 이전 문맥 유지
    "temperature": (0.0, 0.2, 0.4),           # 저온도부터 단계적으로 fallback
    "compression_ratio_threshold": 2.2,       # 반복/깨짐 텍스트 억제
    "logprob_threshold": -1.0,                # 저신뢰 구간 재시도 기준
    "no_speech_threshold": 0.45,              # 무음/배경음 구간 판정 민감도
    "hallucination_silence_threshold": 1.0    # 긴 무음에서 환각 텍스트 억제
}

# ==========================================
# 2. 보조 함수들
# ==========================================
def format_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_timecode(timecode):
    hhmmss, ms = timecode.split(",")
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def load_segments_from_srt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    segments = []
    blocks = content.split("\n\n")
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip() != ""]
        if len(lines) < 2:
            continue

        # 표준 SRT: 번호 / 시간 / 텍스트...
        # 번호가 없더라도 첫 줄이 시간 라인이면 허용
        time_line_index = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_index]:
            continue

        time_line = lines[time_line_index]
        text_lines = lines[time_line_index + 1 :]
        start_str, end_str = [part.strip() for part in time_line.split("-->", 1)]
        segments.append(
            {
                "start": parse_timecode(start_str),
                "end": parse_timecode(end_str),
                "text": "\n".join(text_lines).strip(),
            }
        )
    return segments

def is_repetitive_noise_text(text):
    normalized = re.sub(r"\s+", "", text)
    normalized = re.sub(r"[、。,.!?！？…・「」『』（）()\[\]{}\"'`~^_|\\/:\-+=*#@$%&;]+", "", normalized)

    if len(normalized) < REPETITIVE_FILTER_MIN_CHARS:
        return False

    unique_chars = set(normalized)
    if len(unique_chars) == 1:
        return True

    if len(unique_chars) <= 2:
        max_ratio = max(normalized.count(ch) / len(normalized) for ch in unique_chars)
        if max_ratio >= 0.8:
            return True

    max_unit_len = min(6, len(normalized) // 2)
    for unit_len in range(1, max_unit_len + 1):
        repeats = len(normalized) // unit_len
        if repeats < REPETITIVE_FILTER_MIN_REPEATS:
            continue

        unit = normalized[:unit_len]
        rebuilt = unit * repeats + unit[: len(normalized) % unit_len]
        match_count = sum(1 for a, b in zip(normalized, rebuilt) if a == b)
        match_ratio = match_count / len(normalized)
        if match_ratio >= 0.9:
            return True

    return False

def sanitize_segments(segments, min_duration=0.1):
    cleaned = []
    dropped_empty = 0
    dropped_invalid = 0
    fixed_duration = 0
    dropped_duplicate = 0
    dropped_repetitive = 0
    seen_texts = set()

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()

        if not text:
            dropped_empty += 1
            continue

        if is_repetitive_noise_text(text):
            dropped_repetitive += 1
            continue

        # 전역 중복 제거: 이미 한 번 나온 문장은 이후 모두 제거
        if text in seen_texts:
            dropped_duplicate += 1
            continue
        seen_texts.add(text)

        if end < start:
            dropped_invalid += 1
            continue

        if end == start:
            end = start + min_duration
            fixed_duration += 1

        cleaned.append({"start": start, "end": end, "text": text})

    stats = {
        "dropped_empty": dropped_empty,
        "dropped_invalid": dropped_invalid,
        "fixed_duration": fixed_duration,
        "dropped_duplicate": dropped_duplicate,
        "dropped_repetitive": dropped_repetitive,
    }
    return cleaned, stats

def sanitize_translated_text(text):
    if text is None:
        return ""

    cleaned = text.strip()

    # 모델이 turn 특수 토큰을 반복 출력하는 경우 앞부분 번역만 유지
    cleaned = cleaned.split("<end_of_turn>", 1)[0]
    cleaned = cleaned.split("<start_of_turn>", 1)[0]

    for token in ("<bos>", "<eos>", "<pad>", "</s>"):
        cleaned = cleaned.replace(token, "")

    # 남아있는 단순 특수 토큰 제거 (예: <end_of_turn>)
    cleaned = re.sub(r"<[a-zA-Z_][a-zA-Z0-9_]*>", "", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines()).strip()

    # 괄호로만 감싼 메타 설명문 제거
    # 예: "(No text provided...)", "(음식을 묘사하는 단어...)"
    if re.fullmatch(r"\s*\([^()]+\)\s*", cleaned):
        return ""

    # 문장 내부 괄호 구절 제거
    # 예: "(이것이) 뜨겁네요." -> "뜨겁네요."
    cleaned = re.sub(r"\([^()]*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned

def confirm_overwrite(path):
    answer = input(f"  - 기존 SRT가 있습니다. 덮어쓸까요? ({path}) [y/N]: ").strip().lower()
    return answer in ("y", "yes")

def build_translate_messages(text, source_lang, target_lang):
    return [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
            "image": None,
        }],
    }]

def generate_translation_text(model, tokenizer, source_lang, target_lang, input_text):
    messages = build_translate_messages(input_text, source_lang, target_lang)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=TRANSLATE_MAX_TOKENS,
        verbose=False,
    )
    return sanitize_translated_text(raw_text)

def translate_one_with_retry(model, tokenizer, source_lang, target_lang, text, max_retries, retry_delay, label):
    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            translated = generate_translation_text(model, tokenizer, source_lang, target_lang, text)
            return translated or text
        except Exception as e:
            if attempt < total_attempts:
                print(
                    f"    · 경고: {label} 번역 실패(시도 {attempt}/{total_attempts}) - 재시도합니다. "
                    f"오류: {e}"
                )
                if retry_delay > 0:
                    time.sleep(retry_delay)
            else:
                print(
                    f"    · 경고: {label} 번역 최종 실패(시도 {attempt}/{total_attempts}). "
                    "원문을 그대로 기록합니다."
                )
                return text

def translate_batch_with_retry(
    model,
    tokenizer,
    source_lang,
    target_lang,
    texts,
    max_retries,
    retry_delay,
    label,
):
    separator = "|||SEG_SPLIT|||"
    batch_input = (
        f"Translate each segment to the target language and keep order. "
        f"Output only translations separated by {separator}. "
        f"Total segments: {len(texts)}.\n\n"
        + separator.join(texts)
    )
    total_attempts = max(1, max_retries + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            translated = generate_translation_text(model, tokenizer, source_lang, target_lang, batch_input)
            parts = [part.strip() for part in translated.split(separator)]
            if len(parts) == len(texts) and all(parts):
                return parts
            raise ValueError(f"배치 분할 실패: expected={len(texts)}, got={len(parts)}")
        except Exception as e:
            if attempt < total_attempts:
                print(
                    f"    · 경고: {label} 배치 번역 실패(시도 {attempt}/{total_attempts}) - 재시도합니다. "
                    f"오류: {e}"
                )
                if retry_delay > 0:
                    time.sleep(retry_delay)
            else:
                return None

def validate_language_pair(tokenizer, source_lang, target_lang):
    try:
        messages = build_translate_messages("test", source_lang, target_lang)
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return True, None
    except Exception as e:
        return False, str(e)

# ==========================================
# 3. 메인 로직
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="오디오/영상 파일에서 자막(SRT) 생성")
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH, help="처리할 파일/폴더 경로")
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisper 모델 (기본: {WHISPER_MODEL})",
    )
    parser.add_argument("--lang", default=DEFAULT_WHISPER_LANGUAGE, help="Whisper 입력 언어 코드 (기본: ja)")
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANGUAGE, help="번역 대상 언어 코드 (기본: ko)")
    parser.add_argument(
        "--whisper-accurate",
        action="store_true",
        help="Whisper 정확도 우선 옵션 사용 (기본은 속도 우선)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=f"번역 진행 로그 출력 간격(구간 단위, 기본: {DEFAULT_PROGRESS_EVERY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"구간 번역 실패 시 재시도 횟수(기본: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"재시도 대기 시간(초, 기본: {DEFAULT_RETRY_DELAY})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"번역 배치 크기(기본: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        print("경고: HF_TOKEN이 설정되지 않았습니다. 다운로드 속도/요청 한도가 낮을 수 있습니다.")

    # ffmpeg 사전 점검 (mlx_whisper.transcribe 내부에서 필요)
    if shutil.which("ffmpeg") is None:
        print("오류: 'ffmpeg'를 찾을 수 없습니다.")
        print("설치 후 다시 실행하세요. (macOS: brew install ffmpeg)")
        return

    input_path = args.input_path
    whisper_options = dict(WHISPER_OPTIONS_ACCURATE if args.whisper_accurate else WHISPER_OPTIONS_FAST)
    whisper_options["language"] = args.lang
    # mlx_whisper.transcribe 내부 tqdm 진행바 활성화 (False일 때 진행바 표시)
    whisper_options["verbose"] = False

    # 입력 경로(폴더 또는 파일 1개) 탐색
    if not os.path.exists(input_path):
        print(f"오류: '{input_path}' 경로가 존재하지 않습니다.")
        return

    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(EXTENSIONS)
        ]
        print(f"총 {len(files)}개의 파일을 찾았습니다.\n")
    elif os.path.isfile(input_path):
        if not input_path.lower().endswith(EXTENSIONS):
            print(f"오류: 지원하지 않는 파일 형식입니다. ({input_path})")
            return
        files = [input_path]
        print("단일 파일 1개를 처리합니다.\n")
    else:
        print(f"오류: 유효한 파일 또는 폴더 경로가 아닙니다. ({input_path})")
        return

    if not files:
        print("처리할 파일이 없습니다.")
        return

    # 모델 로드 (한 번만 로드하여 속도 향상)
    print(f"--- 모델 로드 중: {TRANSLATE_MODEL} ---")
    model, tokenizer = load(TRANSLATE_MODEL)

    is_valid_lang_pair, lang_error = validate_language_pair(tokenizer, args.lang, args.target_lang)
    if not is_valid_lang_pair:
        print(f"오류: 유효하지 않은 언어 코드 조합입니다. source='{args.lang}', target='{args.target_lang}'")
        print(f"상세: {lang_error}")
        return

    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        base_path = os.path.splitext(file_path)[0]
        original_output_path = base_path + "-original.srt"
        translated_output_path = base_path + ".srt"
        
        print(f"[{idx+1}/{len(files)}] 작업 시작: {filename}")
        write_original = True
        write_translated = True

        if os.path.exists(original_output_path) and not confirm_overwrite(original_output_path):
            write_original = False
        if os.path.exists(translated_output_path) and not confirm_overwrite(translated_output_path):
            write_translated = False

        if not write_original and not write_translated:
            print("  - 건너뜀: 기존 SRT 파일 유지\n")
            continue

        segments = None

        # Step A: 음성 인식 (Whisper) 또는 기존 원문 SRT 재사용
        if (not write_original) and write_translated and os.path.exists(original_output_path):
            print(f"  - 기존 원문 SRT 재사용: {original_output_path}")
            segments = load_segments_from_srt(original_output_path)
            if not segments:
                print("  - 경고: 기존 원문 SRT를 읽지 못해 Whisper를 다시 실행합니다.")

        if segments is None or not segments:
            print(
                f"  - 음성 인식 및 추출 중... "
                f"(model={args.whisper_model}, language={args.lang}, accurate={args.whisper_accurate})"
            )
            result = mlx_whisper.transcribe(file_path, path_or_hf_repo=args.whisper_model, **whisper_options)
            segments = result['segments']

        segments, segment_stats = sanitize_segments(segments)
        if (
            segment_stats["dropped_empty"] > 0
            or segment_stats["dropped_invalid"] > 0
            or segment_stats["fixed_duration"] > 0
            or segment_stats["dropped_duplicate"] > 0
            or segment_stats["dropped_repetitive"] > 0
        ):
            print(
                "  - 구간 정리: "
                f"빈 텍스트 제거 {segment_stats['dropped_empty']}개, "
                f"역전 구간 제거 {segment_stats['dropped_invalid']}개, "
                f"0초 구간 보정 {segment_stats['fixed_duration']}개, "
                f"중복 문장 제거 {segment_stats['dropped_duplicate']}개, "
                f"반복 노이즈 제거 {segment_stats['dropped_repetitive']}개"
            )

        # Step B: 원문 SRT 저장
        if write_original:
            with open(original_output_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments):
                    start_time = format_time(seg['start'])
                    end_time = format_time(seg['end'])
                    original_text = seg['text'].strip()
                    f.write(f"{i + 1}\n{start_time} --> {end_time}\n{original_text}\n\n")
            print(f"  - 원문 SRT 완료: {original_output_path}")
        else:
            print(f"  - 원문 SRT 건너뜀: {original_output_path}")

        if not segments:
            print("  - 자막 구간이 없어 번역을 건너뜁니다.\n")
            continue

        # Step C: 번역 SRT 저장
        if not write_translated:
            print(f"  - 번역 SRT 건너뜀: {translated_output_path}\n")
            continue

        print(f"  - 번역 및 SRT 생성 중 (구간: {len(segments)}개)...")
        translate_start = time.time()
        progress_every = max(1, args.progress_every)
        batch_size = max(1, args.batch_size)
        translation_cache = {}
        cache_hits = 0
        batch_fallbacks = 0
        use_tqdm = tqdm is not None
        progress_bar = None
        if use_tqdm:
            progress_bar = tqdm(
                total=len(segments),
                desc="  - 번역 진행",
                unit="seg",
            )
        with open(translated_output_path, "w", encoding="utf-8") as f:
            for batch_start in range(0, len(segments), batch_size):
                batch = segments[batch_start : batch_start + batch_size]
                translated_batch = [""] * len(batch)
                uncached_positions = []
                uncached_texts = []

                for pos, seg in enumerate(batch):
                    original_text = seg["text"].strip()
                    if not original_text:
                        translated_batch[pos] = ""
                    elif original_text in translation_cache:
                        translated_batch[pos] = translation_cache[original_text]
                        cache_hits += 1
                    else:
                        uncached_positions.append(pos)
                        uncached_texts.append(original_text)

                if uncached_texts:
                    if len(uncached_texts) == 1:
                        only_pos = uncached_positions[0]
                        original_text = uncached_texts[0]
                        label = f"{batch_start + only_pos + 1}번 구간"
                        translated = translate_one_with_retry(
                            model,
                            tokenizer,
                            args.lang,
                            args.target_lang,
                            original_text,
                            args.max_retries,
                            args.retry_delay,
                            label,
                        )
                        translated_batch[only_pos] = translated
                        translation_cache[original_text] = translated
                    else:
                        label = f"{batch_start + 1}~{batch_start + len(batch)}번 구간"
                        batch_result = translate_batch_with_retry(
                            model,
                            tokenizer,
                            args.lang,
                            args.target_lang,
                            uncached_texts,
                            args.max_retries,
                            args.retry_delay,
                            label,
                        )
                        if batch_result is None:
                            batch_fallbacks += 1
                            for pos, original_text in zip(uncached_positions, uncached_texts):
                                item_label = f"{batch_start + pos + 1}번 구간"
                                translated = translate_one_with_retry(
                                    model,
                                    tokenizer,
                                    args.lang,
                                    args.target_lang,
                                    original_text,
                                    args.max_retries,
                                    args.retry_delay,
                                    item_label,
                                )
                                translated_batch[pos] = translated
                                translation_cache[original_text] = translated
                        else:
                            for pos, original_text, translated in zip(
                                uncached_positions, uncached_texts, batch_result
                            ):
                                translated = translated or original_text
                                translated_batch[pos] = translated
                                translation_cache[original_text] = translated

                for pos, seg in enumerate(batch):
                    i = batch_start + pos
                    start_time = format_time(seg["start"])
                    end_time = format_time(seg["end"])
                    translated_text = translated_batch[pos]
                    f.write(f"{i + 1}\n{start_time} --> {end_time}\n{translated_text}\n\n")

                current = batch_start + len(batch)
                if progress_bar is not None:
                    progress_bar.update(len(batch))
                elif current % progress_every == 0 or current == len(segments):
                    elapsed = time.time() - translate_start
                    print(
                        f"    · 번역 진행: {current}/{len(segments)} ({elapsed:.1f}s, "
                        f"cache hit {cache_hits})"
                    )

        if progress_bar is not None:
            progress_bar.close()
        print(f"  - 번역 캐시 히트: {cache_hits}개")
        print(f"  - 배치 폴백 횟수: {batch_fallbacks}회")
        print(f"  - 번역 SRT 완료: {translated_output_path}\n")

    print("==========================================")
    print("모든 파일의 자막 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
