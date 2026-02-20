import argparse
import subprocess
import sys

INPUT_PATH = "./target_files"
WHISPER_MODEL = "mlx-community/whisper-medium"
DEFAULT_WHISPER_LANGUAGE = "ja"
DEFAULT_TARGET_LANGUAGE = "ko"
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_SECONDS = 0.0
DEFAULT_THROTTLE_MS = 0
DEFAULT_DEVICE = "gpu"
TRANSLATE_MODEL = "mlx-community/translategemma-4b-it-4bit"


def run_step(cmd: list[str], label: str) -> int:
    print(f"--- {label} 시작 ---")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"오류: {label} 단계가 실패했습니다. (exit={proc.returncode})")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="(호환 래퍼) Whisper 추출 + TranslateGemma 번역을 순차 실행"
    )
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH, help="처리할 파일/폴더 경로")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"Whisper 모델 (기본: {WHISPER_MODEL})")
    parser.add_argument("--lang", default=DEFAULT_WHISPER_LANGUAGE, help="Whisper 입력 언어 코드 (기본: ja)")
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANGUAGE, help="번역 대상 언어 코드 (기본: ko)")
    parser.add_argument(
        "--translate-model",
        default=TRANSLATE_MODEL,
        help=f"TranslateGemma 모델 (기본: {TRANSLATE_MODEL})",
    )
    parser.add_argument(
        "--choose-model",
        action="store_true",
        help="Whisper/TranslateGemma 모델을 온라인 목록에서 번호로 선택",
    )
    parser.add_argument("--whisper-accurate", action="store_true", help="Whisper 정확도 우선 옵션 사용")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="번역 진행 로그 간격")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="구간 번역 재시도 횟수")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="재시도 대기 시간(초)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="번역 배치 크기")
    parser.add_argument(
        "--throttle-ms",
        type=int,
        default=DEFAULT_THROTTLE_MS,
        help=f"파일/배치 처리 간 대기 시간(ms, 기본: {DEFAULT_THROTTLE_MS})",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"MLX 실행 디바이스 선택 (기본: {DEFAULT_DEVICE})",
    )
    parser.add_argument("--force", action="store_true", help="기존 SRT가 있어도 확인 없이 덮어쓰기")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="테스트용 최대 처리 길이(초)")
    args = parser.parse_args()
    args.throttle_ms = max(0, args.throttle_ms)

    print("안내: 이 파일은 호환 래퍼입니다.")
    print("  - 원문 추출: mlx-whisper-transcribe.py")
    print("  - 번역 변환: mlx-translategemma.py")

    whisper_cmd = [
        sys.executable,
        "mlx-whisper-transcribe.py",
        args.input_path,
        "--whisper-model",
        args.whisper_model,
        "--lang",
        args.lang,
        "--throttle-ms",
        str(args.throttle_ms),
        "--device",
        args.device,
    ]
    if args.whisper_accurate:
        whisper_cmd.append("--whisper-accurate")
    if args.choose_model:
        whisper_cmd.append("--choose-model")
    if args.force:
        whisper_cmd.append("--force")
    if args.max_seconds and args.max_seconds > 0:
        whisper_cmd.extend(["--max-seconds", str(args.max_seconds)])

    rc = run_step(whisper_cmd, "Whisper 원문 추출")
    if rc != 0:
        raise SystemExit(rc)

    translate_cmd = [
        sys.executable,
        "mlx-translategemma.py",
        args.input_path,
        "--lang",
        args.lang,
        "--target-lang",
        args.target_lang,
        "--translate-model",
        args.translate_model,
        "--progress-every",
        str(args.progress_every),
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
        "--batch-size",
        str(args.batch_size),
        "--throttle-ms",
        str(args.throttle_ms),
        "--device",
        args.device,
    ]
    if args.choose_model:
        translate_cmd.append("--choose-model")
    if args.force:
        translate_cmd.append("--force")
    rc = run_step(translate_cmd, "TranslateGemma 번역")
    if rc != 0:
        raise SystemExit(rc)

    print("==========================================")
    print("모든 파일의 자막 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()
