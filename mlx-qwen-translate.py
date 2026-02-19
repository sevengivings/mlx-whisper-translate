import argparse
import subprocess
import sys

INPUT_PATH = "./target_files"
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"
DEFAULT_SOURCE_LANGUAGE = "ja"
DEFAULT_TARGET_LANGUAGE = "ko"
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_BATCH_SIZE = 1
DEFAULT_CHUNK_DURATION = 30.0
DEFAULT_MIN_CHUNK_DURATION = 1.0
DEFAULT_MAX_SECONDS = 0.0
TRANSLATE_MODEL = "mlx-community/translategemma-4b-it-4bit"


def run_step(cmd: list[str], label: str) -> int:
    print(f"--- {label} 시작 ---")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"오류: {label} 단계가 실패했습니다. (exit={proc.returncode})")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="(호환 래퍼) Qwen ASR 추출 + TranslateGemma 번역을 순차 실행"
    )
    parser.add_argument("input_path", nargs="?", default=INPUT_PATH, help="처리할 파일/폴더 경로")
    parser.add_argument("--asr-model", default=ASR_MODEL, help=f"ASR 모델 (기본: {ASR_MODEL})")
    parser.add_argument("--lang", default=DEFAULT_SOURCE_LANGUAGE, help="원문 언어 코드 (기본: ja)")
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANGUAGE, help="번역 대상 언어 코드 (기본: ko)")
    parser.add_argument(
        "--translate-model",
        default=TRANSLATE_MODEL,
        help=f"TranslateGemma 모델 (기본: {TRANSLATE_MODEL})",
    )
    parser.add_argument(
        "--choose-model",
        action="store_true",
        help="Qwen3-ASR/TranslateGemma 모델을 온라인 목록에서 번호로 선택",
    )
    parser.add_argument("--force", action="store_true", help="기존 SRT가 있어도 확인 없이 덮어쓰기")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="테스트용 최대 처리 길이(초)")
    parser.add_argument("--chunk-duration", type=float, default=DEFAULT_CHUNK_DURATION, help="Qwen ASR 청크 길이(초)")
    parser.add_argument("--min-chunk-duration", type=float, default=DEFAULT_MIN_CHUNK_DURATION, help="Qwen ASR 최소 청크 길이(초)")
    parser.add_argument("--vad-preprocess", action="store_true", help="ffmpeg silencedetect 기반 VAD 전처리 사용")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="번역 진행 로그 간격")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="구간 번역 재시도 횟수")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="재시도 대기 시간(초)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="번역 배치 크기")
    args = parser.parse_args()

    print("안내: 이 파일은 호환 래퍼입니다.")
    print("  - 원문 추출: mlx-qwen-transcribe.py")
    print("  - 번역 변환: mlx-translategemma.py")

    qwen_cmd = [
        sys.executable,
        "mlx-qwen-transcribe.py",
        args.input_path,
        "--asr-model",
        args.asr_model,
        "--lang",
        args.lang,
        "--chunk-duration",
        str(args.chunk_duration),
        "--min-chunk-duration",
        str(args.min_chunk_duration),
    ]
    if args.vad_preprocess:
        qwen_cmd.append("--vad-preprocess")
    if args.choose_model:
        qwen_cmd.append("--choose-model")
    if args.force:
        qwen_cmd.append("--force")
    if args.max_seconds and args.max_seconds > 0:
        qwen_cmd.extend(["--max-seconds", str(args.max_seconds)])

    rc = run_step(qwen_cmd, "Qwen 원문 추출")
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
