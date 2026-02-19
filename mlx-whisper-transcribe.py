import argparse
import datetime
import os
import shutil
import subprocess
import tempfile

import mlx_whisper

from model_picker import choose_model_online
from subtitle_cleanup import sanitize_segments

INPUT_PATH = "./target_files"
EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".mkv")
WHISPER_MODEL = "mlx-community/whisper-medium"
DEFAULT_WHISPER_LANGUAGE = "ja"
DEFAULT_MAX_SECONDS = 0.0

WHISPER_OPTIONS_FAST = {
    "word_timestamps": False,
    "condition_on_previous_text": False,
    "temperature": 0.0,
    "no_speech_threshold": 0.45,
}

WHISPER_OPTIONS_ACCURATE = {
    "word_timestamps": True,
    "condition_on_previous_text": True,
    "temperature": (0.0, 0.2, 0.4),
    "compression_ratio_threshold": 2.2,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.45,
    "hallucination_silence_threshold": 1.0,
}


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


def write_srt(path: str, segments: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            text = str(seg["text"]).strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")


def extract_head_to_temp_wav(input_path: str, max_seconds: float) -> str:
    fd, temp_wav = tempfile.mkstemp(prefix="whisper_head_", suffix=".wav")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper(MLX)로 원문 SRT(-original.srt) 생성")
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH, help="처리할 파일/폴더 경로")
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisper 모델 (기본: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--choose-model",
        action="store_true",
        help="온라인에서 mlx-community Whisper 모델 목록을 조회해 번호로 선택",
    )
    parser.add_argument("--lang", default=DEFAULT_WHISPER_LANGUAGE, help="Whisper 입력 언어 코드 (기본: ja)")
    parser.add_argument(
        "--whisper-accurate",
        action="store_true",
        help="Whisper 정확도 우선 옵션 사용 (기본은 속도 우선)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 -original.srt가 있어도 확인 없이 덮어쓰기",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="테스트용 최대 처리 길이(초). 0이면 전체 처리",
    )
    parser.add_argument(
        "--original-output",
        default=None,
        help="원문 SRT 출력 파일 경로(단일 파일 입력일 때만 사용)",
    )
    args = parser.parse_args()

    if args.choose_model:
        args.whisper_model = choose_model_online("whisper", args.whisper_model)

    if not os.getenv("HF_TOKEN"):
        print("경고: HF_TOKEN이 설정되지 않았습니다. 다운로드 속도/요청 한도가 낮을 수 있습니다.")

    if shutil.which("ffmpeg") is None:
        print("오류: 'ffmpeg'를 찾을 수 없습니다.")
        print("설치 후 다시 실행하세요. (macOS: brew install ffmpeg)")
        return

    input_path = args.input_path
    whisper_options = dict(WHISPER_OPTIONS_ACCURATE if args.whisper_accurate else WHISPER_OPTIONS_FAST)
    whisper_options["language"] = args.lang
    whisper_options["verbose"] = False

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

    if args.original_output and len(files) != 1:
        print("오류: --original-output은 단일 파일 입력에서만 사용할 수 있습니다.")
        return

    for idx, file_path in enumerate(files, start=1):
        filename = os.path.basename(file_path)
        original_output_path = args.original_output or (os.path.splitext(file_path)[0] + "-original.srt")
        print(f"[{idx}/{len(files)}] 작업 시작: {filename}")

        if os.path.exists(original_output_path) and not args.force:
            if not confirm_overwrite(original_output_path):
                print(f"  - 건너뜀: 기존 파일 유지 ({original_output_path})\n")
                continue

        transcribe_input = file_path
        temp_head_wav = None
        if args.max_seconds and args.max_seconds > 0:
            print(f"  - 테스트 모드: 앞 {args.max_seconds:.1f}초만 처리합니다.")
            temp_head_wav = extract_head_to_temp_wav(file_path, args.max_seconds)
            transcribe_input = temp_head_wav

        print(
            f"  - 음성 인식 및 추출 중... "
            f"(model={args.whisper_model}, language={args.lang}, accurate={args.whisper_accurate})"
        )
        try:
            result = mlx_whisper.transcribe(transcribe_input, path_or_hf_repo=args.whisper_model, **whisper_options)
        finally:
            if temp_head_wav and os.path.exists(temp_head_wav):
                os.remove(temp_head_wav)
        segments, stats = sanitize_segments(result.get("segments", []), min_duration=0.1)

        if any(stats.values()):
            print(
                "  - 구간 정리: "
                f"빈 텍스트 제거 {stats['dropped_empty']}개, "
                f"역전 구간 제거 {stats['dropped_invalid']}개, "
                f"0초 구간 보정 {stats['fixed_duration']}개, "
                f"중복 문장 제거 {stats['dropped_duplicate']}개, "
                f"반복 노이즈 제거 {stats['dropped_repetitive']}개"
            )

        write_srt(original_output_path, segments)
        print(f"  - 원문 SRT 완료: {original_output_path}\n")

    print("==========================================")
    print("Whisper 원문 자막 생성이 완료되었습니다!")


if __name__ == "__main__":
    main()
