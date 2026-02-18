import argparse
import datetime
import inspect
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Optional
from subtitle_cleanup import sanitize_segments as sanitize_segments_shared
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from mlx_audio.transcribe import generate_transcription as _transcribe_fn
    TRANSCRIBE_BACKEND = "mlx_audio.transcribe"
except ImportError:
    _transcribe_fn = None
    TRANSCRIBE_BACKEND = None

if _transcribe_fn is None:
    try:
        from mlx_audio.stt.generate import generate_transcription as _stt_generate_fn
        TRANSCRIBE_BACKEND = "mlx_audio.stt.generate"
    except ImportError:
        _stt_generate_fn = None
else:
    _stt_generate_fn = None

try:
    from mlx_audio.stt.utils import get_available_models as _get_available_stt_models
except ImportError:
    _get_available_stt_models = None

try:
    from mlx_audio.stt.utils import load_model as _load_stt_model
except ImportError:
    _load_stt_model = None


INPUT_PATH = "./target_files"
EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".mkv")
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"
DEFAULT_LANGUAGE = "ja"
DEFAULT_CHUNK_DURATION = 30.0
DEFAULT_MIN_CHUNK_DURATION = 1.0
DEFAULT_VAD_NOISE = "-35dB"
DEFAULT_VAD_MIN_SILENCE = 0.5
DEFAULT_VAD_MIN_SEGMENT = 1.0
DEFAULT_VAD_MERGE_GAP = 0.3
DEFAULT_MAX_SUB_DURATION = 8.0
DEFAULT_MAX_SUB_CHARS = 28
DEFAULT_MIN_SUB_DURATION = 0.8
DEFAULT_MAX_SECONDS = 0.0

_STT_MODEL_CACHE: dict[str, Any] = {}
_USE_TQDM_WRITE = False


def log(message: str = "") -> None:
    if _USE_TQDM_WRITE and tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def format_time(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def confirm_overwrite(path: str) -> bool:
    answer = input(f"  - 기존 SRT가 있습니다. 덮어쓸까요? ({path}) [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def sanitize_segments(segments: list[dict[str, Any]], min_duration: float = 0.1) -> list[dict[str, Any]]:
    cleaned, _ = sanitize_segments_shared(
        segments,
        min_duration=min_duration,
        drop_repetitive=True,
        drop_duplicate=True,
    )
    return cleaned


def write_srt(path: str, segments: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")


def convert_media_to_temp_wav(input_path: str) -> str:
    fd, temp_wav = tempfile.mkstemp(prefix="qwen_asr_", suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        temp_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise RuntimeError(f"ffmpeg 변환 실패: {proc.stderr.strip()}")
    return temp_wav


def extract_head_to_temp_wav(input_path: str, max_seconds: float) -> str:
    fd, temp_wav = tempfile.mkstemp(prefix="qwen_head_", suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-t",
        f"{max_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        temp_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise RuntimeError(f"ffmpeg 앞부분 추출 실패: {proc.stderr.strip()}")
    return temp_wav


def parse_timecode(timecode: str) -> float:
    hhmmss, ms = timecode.split(",")
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def load_segments_from_srt(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    segments = []
    blocks = content.split("\n\n")
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line_index = 1 if len(lines) > 1 and "-->" in lines[1] else 0
        if "-->" not in lines[time_line_index]:
            continue
        start_str, end_str = [p.strip() for p in lines[time_line_index].split("-->", 1)]
        text = "\n".join(lines[time_line_index + 1 :]).strip()
        if not text:
            continue
        segments.append(
            {"start": parse_timecode(start_str), "end": parse_timecode(end_str), "text": text}
        )
    return segments


def get_media_duration(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe duration 조회 실패: {proc.stderr.strip()}")
    return max(0.0, float(proc.stdout.strip()))


def detect_silence_intervals(input_path: str, noise: str, min_silence: float) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-af",
        f"silencedetect=noise={noise}:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logs = proc.stderr

    silence_start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    silence_end_re = re.compile(r"silence_end:\s*([0-9.]+)")

    silences: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    for line in logs.splitlines():
        m_start = silence_start_re.search(line)
        if m_start:
            current_start = float(m_start.group(1))
            continue

        m_end = silence_end_re.search(line)
        if m_end:
            end = float(m_end.group(1))
            start = 0.0 if current_start is None else current_start
            if end > start:
                silences.append((start, end))
            current_start = None

    return silences


def merge_intervals(intervals: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_voiced_intervals(
    duration: float,
    silences: list[tuple[float, float]],
    min_segment: float,
    merge_gap: float,
) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    if not silences:
        return [(0.0, duration)]

    silences = sorted(silences, key=lambda x: x[0])
    voiced: list[tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences:
        s_start = max(0.0, min(duration, s_start))
        s_end = max(0.0, min(duration, s_end))
        if s_start > cursor:
            voiced.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if cursor < duration:
        voiced.append((cursor, duration))

    voiced = merge_intervals(voiced, merge_gap)
    return [(s, e) for s, e in voiced if (e - s) >= min_segment]


def extract_wav_segment(input_wav: str, start: float, end: float, output_wav: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        input_wav,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_wav,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 구간 추출 실패({start:.2f}-{end:.2f}s): {proc.stderr.strip()}")


def run_vad_preprocessed_transcription(
    source_path: str,
    source_label: str,
    output_path: str,
    model_id: str,
    language: Optional[str],
    chunk_duration: float,
    min_chunk_duration: float,
    vad_noise: str,
    vad_min_silence: float,
    vad_min_segment: float,
    vad_merge_gap: float,
    use_progress: bool,
    input_is_wav: bool = False,
) -> None:
    temp_vad_wav = source_path if input_is_wav else convert_media_to_temp_wav(source_path)
    try:
        duration = get_media_duration(temp_vad_wav)
        silences = detect_silence_intervals(
            temp_vad_wav, noise=vad_noise, min_silence=vad_min_silence
        )
        voiced_intervals = build_voiced_intervals(
            duration=duration,
            silences=silences,
            min_segment=vad_min_segment,
            merge_gap=vad_merge_gap,
        )
        if not voiced_intervals:
            log("  - VAD 결과 음성 구간이 없어 전체 파일로 대체 인식합니다.")
            voiced_intervals = [(0.0, duration)]
        log(f"  - VAD 음성 구간: {len(voiced_intervals)}개")

        merged_segments: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="qwen_vad_") as tmpdir:
            seg_iter = voiced_intervals
            if tqdm is not None and use_progress:
                seg_iter = tqdm(voiced_intervals, desc=f"ASR chunks ({source_label})", unit="seg")

            for seg_idx, (seg_start, seg_end) in enumerate(seg_iter, start=1):
                chunk_wav = os.path.join(tmpdir, f"chunk_{seg_idx:04d}.wav")
                chunk_srt = os.path.join(tmpdir, f"chunk_{seg_idx:04d}.srt")
                extract_wav_segment(temp_vad_wav, seg_start, seg_end, chunk_wav)
                generate_srt_with_qwen(
                    audio_path=chunk_wav,
                    output_path=chunk_srt,
                    model_id=model_id,
                    language=language,
                    chunk_duration=chunk_duration,
                    min_chunk_duration=min_chunk_duration,
                )
                chunk_segments = load_segments_from_srt(chunk_srt)
                for s in chunk_segments:
                    merged_segments.append(
                        {
                            "start": s["start"] + seg_start,
                            "end": s["end"] + seg_start,
                            "text": s["text"],
                        }
                    )

        merged_segments = sanitize_segments(sorted(merged_segments, key=lambda x: x["start"]))
        if not merged_segments:
            raise RuntimeError("VAD 전처리 후 생성된 자막 구간이 없습니다.")
        write_srt(output_path, merged_segments)
    finally:
        if (not input_is_wav) and temp_vad_wav and os.path.exists(temp_vad_wav):
            os.remove(temp_vad_wav)


def split_text_units(text: str) -> list[str]:
    units = []
    for part in re.split(r"([。！？!?…]+)", text):
        if not part:
            continue
        if re.fullmatch(r"[。！？!?…]+", part):
            if units:
                units[-1] = units[-1] + part
            continue
        units.extend([p for p in re.split(r"[、,]\s*", part) if p])
    return [u.strip() for u in units if u.strip()]


def split_long_segments(
    segments: list[dict[str, Any]],
    max_duration: float,
    max_chars: int,
    min_duration: float,
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    split_added = 0

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg["text"]).strip()
        duration = max(0.1, end - start)

        if duration <= max_duration and len(text) <= max_chars:
            out.append(seg)
            continue

        units = split_text_units(text)
        if len(units) <= 1:
            out.append(seg)
            continue

        groups: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for u in units:
            ul = len(u)
            if cur and (cur_len + ul > max_chars):
                groups.append(" ".join(cur).strip())
                cur = [u]
                cur_len = ul
            else:
                cur.append(u)
                cur_len += ul
        if cur:
            groups.append(" ".join(cur).strip())

        if len(groups) <= 1:
            out.append(seg)
            continue

        weights = [max(1, len(re.sub(r"\s+", "", g))) for g in groups]
        total_weight = sum(weights)
        cursor = start
        for i, g in enumerate(groups):
            if i == len(groups) - 1:
                part_start, part_end = cursor, end
            else:
                part_dur = max(min_duration, duration * (weights[i] / total_weight))
                part_start, part_end = cursor, min(end, cursor + part_dur)
                cursor = part_end
            out.append({"start": part_start, "end": part_end, "text": g})

        split_added += len(groups) - 1

    return out, split_added


def postprocess_srt(
    path: str,
    max_sub_duration: float,
    max_sub_chars: int,
    min_sub_duration: float,
) -> int:
    segments = load_segments_from_srt(path)
    if not segments:
        return 0

    new_segments, split_added = split_long_segments(
        segments=segments,
        max_duration=max_sub_duration,
        max_chars=max_sub_chars,
        min_duration=min_sub_duration,
    )
    if split_added > 0:
        write_srt(path, new_segments)
    return split_added


def extract_segments(result: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    if isinstance(result, dict) and isinstance(result.get("segments"), list):
        segments = result["segments"]
    elif hasattr(result, "segments") and isinstance(getattr(result, "segments"), list):
        segments = getattr(result, "segments")
    elif isinstance(result, list):
        segments = result
    elif isinstance(result, dict) and isinstance(result.get("chunks"), list):
        # 일부 ASR 구현은 chunks + timestamp((start, end)) 형태를 반환
        for ch in result["chunks"]:
            text = str(ch.get("text", "")).strip()
            timestamp = ch.get("timestamp")
            if (
                isinstance(timestamp, (tuple, list))
                and len(timestamp) == 2
                and text
            ):
                segments.append({"start": timestamp[0], "end": timestamp[1], "text": text})

    return sanitize_segments(segments)


def get_cached_stt_model(model_id: str) -> Any:
    if model_id in _STT_MODEL_CACHE:
        return _STT_MODEL_CACHE[model_id]

    if _load_stt_model is None:
        return model_id

    log(f"  - STT 모델 로드(1회): {model_id}")
    model = _load_stt_model(model_id)
    _STT_MODEL_CACHE[model_id] = model
    return model


def generate_srt_with_qwen(
    audio_path: str,
    output_path: str,
    model_id: str,
    language: Optional[str],
    chunk_duration: float,
    min_chunk_duration: float,
) -> None:
    if _transcribe_fn is None and _stt_generate_fn is None:
        raise RuntimeError(
            "mlx-audio에서 ASR 함수를 찾지 못했습니다. `pip install -U mlx-audio` 후 다시 실행하세요."
        )

    result: Any = None
    if _transcribe_fn is not None:
        # 구버전/중간 버전 호환: transcribe API가 있는 경우
        kwargs: dict[str, Any] = {}
        kwargs["repo_id"] = model_id
        if language:
            kwargs["language"] = language
        kwargs["format"] = "srt"
        kwargs["output_path"] = output_path
        kwargs["chunk_duration"] = chunk_duration
        kwargs["min_chunk_duration"] = min_chunk_duration
        result = _transcribe_fn(audio_path, **kwargs)
    else:
        # mlx-audio==0.2.7 계열: stt.generate.generate 사용
        if _get_available_stt_models is not None:
            available = set(_get_available_stt_models())
            # Qwen3-ASR 모델을 쓰려면 qwen3_asr 타입이 필요
            if "qwen3_asr" in model_id.lower() and "qwen3_asr" not in available:
                supported = ", ".join(sorted(available)) if available else "(알 수 없음)"
                raise RuntimeError(
                    "현재 mlx-audio 버전은 Qwen3-ASR STT를 지원하지 않습니다. "
                    f"지원 모델 타입: {supported}. "
                    "`.venv/bin/python -m pip install -U 'mlx-audio>=0.3.1'` 후 다시 실행하세요."
                )

        output_base = output_path[:-4] if output_path.endswith(".srt") else output_path
        sig = inspect.signature(_stt_generate_fn)
        params = sig.parameters
        kwargs = {
            "model": get_cached_stt_model(model_id),
            "output_path": output_base,
            "format": "srt",
            "verbose": False,
        }
        # 버전별 인자명 호환: audio(0.3.x) / audio_path(구버전)
        if "audio" in params:
            kwargs["audio"] = audio_path
        elif "audio_path" in params:
            kwargs["audio_path"] = audio_path
        else:
            # 예상 밖 시그니처일 때도 기본 동작 시도
            kwargs["audio"] = audio_path

        if language:
            kwargs["language"] = language
        kwargs["chunk_duration"] = chunk_duration
        kwargs["min_chunk_duration"] = min_chunk_duration
        try:
            result = _stt_generate_fn(**kwargs)
        except TypeError:
            # 일부 모델/버전은 language 파라미터를 받지 않음
            kwargs.pop("language", None)
            result = _stt_generate_fn(**kwargs)
        except ValueError as e:
            if "Model type None not supported" in str(e):
                raise RuntimeError(
                    "mlx-audio가 모델 타입을 인식하지 못했습니다. "
                    "Qwen3-ASR 사용 시 `mlx-audio>=0.3.1`이 필요합니다."
                ) from e
            raise

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    segments = extract_segments(result)
    if not segments:
        raise RuntimeError(
            "SRT 저장에 실패했습니다. 현재 mlx-audio 버전의 반환 형식에서 타임스탬프를 찾지 못했습니다."
        )
    write_srt(output_path, segments)


def main() -> None:
    global _USE_TQDM_WRITE
    parser = argparse.ArgumentParser(description="Qwen3-ASR(MLX)로 원문 SRT(-original.srt) 생성")
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH, help="처리할 파일/폴더 경로")
    parser.add_argument(
        "--asr-model",
        default=ASR_MODEL,
        help=f"ASR 모델 (기본: {ASR_MODEL})",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANGUAGE,
        help=f"입력 언어 코드 (기본: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 -original.srt가 있어도 확인 없이 덮어쓰기",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Qwen ASR 청크 길이(초, 기본: {DEFAULT_CHUNK_DURATION})",
    )
    parser.add_argument(
        "--min-chunk-duration",
        type=float,
        default=DEFAULT_MIN_CHUNK_DURATION,
        help=f"Qwen ASR 최소 청크 길이(초, 기본: {DEFAULT_MIN_CHUNK_DURATION})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm 진행바 비활성화",
    )
    parser.add_argument(
        "--vad-preprocess",
        action="store_true",
        help="ffmpeg silencedetect 기반 VAD 전처리 사용",
    )
    parser.add_argument(
        "--vad-noise",
        default=DEFAULT_VAD_NOISE,
        help=f"silencedetect noise 임계값 (기본: {DEFAULT_VAD_NOISE})",
    )
    parser.add_argument(
        "--vad-min-silence",
        type=float,
        default=DEFAULT_VAD_MIN_SILENCE,
        help=f"silencedetect 최소 무음 길이(초, 기본: {DEFAULT_VAD_MIN_SILENCE})",
    )
    parser.add_argument(
        "--vad-min-segment",
        type=float,
        default=DEFAULT_VAD_MIN_SEGMENT,
        help=f"VAD 후 유지할 최소 음성 구간 길이(초, 기본: {DEFAULT_VAD_MIN_SEGMENT})",
    )
    parser.add_argument(
        "--vad-merge-gap",
        type=float,
        default=DEFAULT_VAD_MERGE_GAP,
        help=f"인접 음성 구간 병합 간격(초, 기본: {DEFAULT_VAD_MERGE_GAP})",
    )
    parser.add_argument(
        "--max-sub-duration",
        type=float,
        default=DEFAULT_MAX_SUB_DURATION,
        help=f"SRT 후처리 분할 최대 길이(초, 기본: {DEFAULT_MAX_SUB_DURATION})",
    )
    parser.add_argument(
        "--max-sub-chars",
        type=int,
        default=DEFAULT_MAX_SUB_CHARS,
        help=f"SRT 후처리 분할 최대 글자 수(기본: {DEFAULT_MAX_SUB_CHARS})",
    )
    parser.add_argument(
        "--min-sub-duration",
        type=float,
        default=DEFAULT_MIN_SUB_DURATION,
        help=f"SRT 후처리 최소 구간 길이(초, 기본: {DEFAULT_MIN_SUB_DURATION})",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="테스트용 최대 처리 길이(초). 0이면 전체 처리",
    )
    parser.add_argument(
        "--show-hf-progress",
        action="store_true",
        help="Hugging Face 다운로드 진행바 표시(기본: 비표시)",
    )
    parser.add_argument(
        "--original-output",
        default=None,
        help="원문 SRT 출력 파일 경로(단일 파일 입력일 때만 사용)",
    )
    args = parser.parse_args()
    _USE_TQDM_WRITE = (tqdm is not None and not args.no_progress)

    if not args.show_hf_progress:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    if shutil.which("ffmpeg") is None:
        log("오류: 'ffmpeg'를 찾을 수 없습니다.")
        log("설치 후 다시 실행하세요. (macOS: brew install ffmpeg)")
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        log(f"오류: '{input_path}' 경로가 존재하지 않습니다.")
        return

    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(EXTENSIONS)
        ]
        log(f"총 {len(files)}개의 파일을 찾았습니다.\n")
    elif os.path.isfile(input_path):
        if not input_path.lower().endswith(EXTENSIONS):
            log(f"오류: 지원하지 않는 파일 형식입니다. ({input_path})")
            return
        files = [input_path]
        log("단일 파일 1개를 처리합니다.\n")
    else:
        log(f"오류: 유효한 파일 또는 폴더 경로가 아닙니다. ({input_path})")
        return

    if not files:
        log("처리할 파일이 없습니다.")
        return

    if args.original_output and len(files) != 1:
        log("오류: --original-output은 단일 파일 입력에서만 사용할 수 있습니다.")
        return

    if not os.getenv("HF_TOKEN"):
        log("경고: HF_TOKEN이 설정되지 않았습니다. 다운로드 속도/요청 한도가 낮을 수 있습니다.")

    file_iter = files
    if tqdm is not None and not args.no_progress:
        file_iter = tqdm(files, desc="Qwen ASR", unit="file")

    for idx, file_path in enumerate(file_iter, start=1):
        filename = os.path.basename(file_path)
        base_path = os.path.splitext(file_path)[0]
        original_output_path = args.original_output or (base_path + "-original.srt")

        if tqdm is not None and not args.no_progress and hasattr(file_iter, "set_postfix_str"):
            file_iter.set_postfix_str(filename)

        log(f"[{idx}/{len(files)}] 작업 시작: {filename}")
        if os.path.exists(original_output_path) and not args.force:
            if not confirm_overwrite(original_output_path):
                log(f"  - 건너뜀: 기존 파일 유지 ({original_output_path})\n")
                continue

        temp_audio_path = None
        temp_head_wav = None
        processing_input_path = file_path
        try:
            if args.max_seconds and args.max_seconds > 0:
                log(f"  - 테스트 모드: 앞 {args.max_seconds:.1f}초만 처리합니다.")
                temp_head_wav = extract_head_to_temp_wav(file_path, args.max_seconds)
                processing_input_path = temp_head_wav

            log(f"  - 음성 인식 및 SRT 생성 중... (model={args.asr_model}, language={args.lang})")
            if args.vad_preprocess:
                log(
                    f"  - VAD 전처리 중... (noise={args.vad_noise}, "
                    f"min_silence={args.vad_min_silence}s)"
                )
                run_vad_preprocessed_transcription(
                    source_path=processing_input_path,
                    source_label=filename,
                    output_path=original_output_path,
                    model_id=args.asr_model,
                    language=args.lang,
                    chunk_duration=args.chunk_duration,
                    min_chunk_duration=args.min_chunk_duration,
                    vad_noise=args.vad_noise,
                    vad_min_silence=args.vad_min_silence,
                    vad_min_segment=args.vad_min_segment,
                    vad_merge_gap=args.vad_merge_gap,
                    use_progress=not args.no_progress,
                )
            else:
                generate_srt_with_qwen(
                    audio_path=processing_input_path,
                    output_path=original_output_path,
                    model_id=args.asr_model,
                    language=args.lang,
                    chunk_duration=args.chunk_duration,
                    min_chunk_duration=args.min_chunk_duration,
                )
            split_added = postprocess_srt(
                original_output_path,
                max_sub_duration=max(1.0, args.max_sub_duration),
                max_sub_chars=max(10, args.max_sub_chars),
                min_sub_duration=max(0.2, args.min_sub_duration),
            )
            if split_added > 0:
                log(
                    "  - SRT 후처리 분할: "
                    f"{split_added}개 추가 (max_duration={args.max_sub_duration}s, "
                    f"max_chars={args.max_sub_chars})"
                )
            log(f"  - 완료: {original_output_path}\n")
        except Exception as e:
            err = str(e)
            if "unsupported file format" in err.lower():
                try:
                    log("  - 입력 포맷을 직접 읽지 못해 WAV(16k mono)로 변환 후 재시도합니다...")
                    temp_audio_path = convert_media_to_temp_wav(processing_input_path)
                    log("  - 재시도는 VAD 구간 분할 모드로 진행합니다...")
                    run_vad_preprocessed_transcription(
                        source_path=temp_audio_path,
                        source_label=filename,
                        output_path=original_output_path,
                        model_id=args.asr_model,
                        language=args.lang,
                        chunk_duration=args.chunk_duration,
                        min_chunk_duration=args.min_chunk_duration,
                        vad_noise=args.vad_noise,
                        vad_min_silence=args.vad_min_silence,
                        vad_min_segment=args.vad_min_segment,
                        vad_merge_gap=args.vad_merge_gap,
                        use_progress=not args.no_progress,
                        input_is_wav=True,
                    )
                    split_added = postprocess_srt(
                        original_output_path,
                        max_sub_duration=max(1.0, args.max_sub_duration),
                        max_sub_chars=max(10, args.max_sub_chars),
                        min_sub_duration=max(0.2, args.min_sub_duration),
                    )
                    if split_added > 0:
                        log(
                            "  - SRT 후처리 분할: "
                            f"{split_added}개 추가 (max_duration={args.max_sub_duration}s, "
                            f"max_chars={args.max_sub_chars})"
                        )
                    log(f"  - 완료: {original_output_path}\n")
                    continue
                except Exception as fallback_error:
                    err = str(fallback_error)
            if "401" in err and "Repository Not Found" in err:
                log("  - 오류: 모델 저장소 접근에 실패했습니다 (401/Repository Not Found).")
                log(f"    요청 모델: {args.asr_model}")
                log("    확인된 공개 대체 모델 예시:")
                log("    - mlx-community/Qwen3-ASR-0.6B-4bit")
                log("    - mlx-community/Qwen3-ASR-1.7B-4bit")
                log("    필요하면 HF_TOKEN 로그인 상태도 확인하세요.\n")
            else:
                log(f"  - 오류: {e}\n")
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if temp_head_wav and os.path.exists(temp_head_wav):
                os.remove(temp_head_wav)


if __name__ == "__main__":
    main()
