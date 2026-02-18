# mlx-whisper-translate

로컬(Apple Silicon + MLX)에서 오디오/영상 파일의 자막을 생성하고 번역하는 스크립트 모음입니다.

현재는 기능이 분리되어 있습니다.

- ASR(Whisper): `mlx-whisper-transcribe.py`
- ASR(Qwen): `mlx-qwen-transcribe.py`
- 번역(TranslateGemma): `mlx-translategemma.py`
- 호환 래퍼(기존 흐름 유지):
  - Whisper + TranslateGemma: `mlx-whisper-translate.py`
  - Qwen + TranslateGemma: `mlx-qwen-translate.py`

공통 자막 정리(반복 노이즈/중복/구간 보정)는 `subtitle_cleanup.py`를 공유합니다.

## 1) 출력 파일 규칙

기본 출력:

- 원문 자막: `파일명-original.srt`
- 번역 자막: `파일명.srt`

직접 지정 옵션:

- 원문 출력 파일 지정: `--original-output` (Whisper/Qwen transcribe, 단일 파일 입력 전용)
- 번역 입력/출력 지정:
  - `--original-srt`
  - `--output-srt`

## 2) 설치

필수:

- Python 3.10 이상 (권장: 3.12)
- Apple Silicon + MLX 환경

참고:

- 시스템 `python3`가 3.9인 경우 일부 문법(`str | None`) 때문에 실행되지 않습니다.
- 반드시 `.venv` 인터프리터로 실행하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`ffmpeg` 설치:

```bash
brew install ffmpeg
```

권장: Hugging Face 토큰 설정

```bash
export HF_TOKEN="hf_xxx"
```

## 3) 스크립트별 사용법

### A. Whisper로 원문 SRT만 생성

```bash
.venv/bin/python mlx-whisper-transcribe.py "./target_files" --lang ja
```

자주 쓰는 옵션:

- `--whisper-model`
- `--whisper-accurate`
- `--force`
- `--max-seconds` (테스트용 앞부분만 처리)
- `--original-output` (단일 파일 입력 시 출력 파일명 지정)

예시(3분 테스트 + 출력명 지정):

```bash
.venv/bin/python mlx-whisper-transcribe.py "/path/to/input.mp4" \
  --max-seconds 180 \
  --force \
  --original-output "/path/to/whisper-original-3m.srt"
```

### B. Qwen으로 원문 SRT만 생성

```bash
.venv/bin/python mlx-qwen-transcribe.py "./target_files" --lang ja
```

자주 쓰는 옵션:

- `--asr-model` (기본: `mlx-community/Qwen3-ASR-1.7B-4bit`)
- `--vad-preprocess` (VAD 전처리)
- `--chunk-duration`, `--min-chunk-duration`
- `--max-seconds` (테스트용 앞부분만 처리)
- `--max-sub-duration`, `--max-sub-chars`, `--min-sub-duration` (SRT 후처리 분할)
- `--force`
- `--original-output` (단일 파일 입력 시 출력 파일명 지정)
- `--show-hf-progress` (기본은 HF 진행바 비표시)

예시(3분 테스트 + VAD + 출력명 지정):

```bash
.venv/bin/python mlx-qwen-transcribe.py "/path/to/input.mp4" \
  --max-seconds 180 \
  --vad-preprocess \
  --force \
  --original-output "/path/to/qwen-original-3m.srt"
```

### C. TranslateGemma로 번역만 수행

기본(파일/폴더 입력 시 `-original.srt`를 찾아 번역):

```bash
.venv/bin/python mlx-translategemma.py "./target_files" --lang ja --target-lang ko
```

직접 SRT 지정:

```bash
.venv/bin/python mlx-translategemma.py \
  --original-srt "/path/to/input-original.srt" \
  --output-srt "/path/to/output-ko.srt" \
  --lang ja \
  --target-lang ko \
  --force
```

자주 쓰는 옵션:

- `--batch-size`
- `--max-retries`
- `--retry-delay`
- `--force`

## 4) 호환 래퍼 사용법

### 기존 Whisper 전체 파이프라인 (호환)

```bash
.venv/bin/python mlx-whisper-translate.py "/path/to/input.mp4" \
  --lang ja \
  --target-lang ko \
  --max-seconds 180 \
  --force
```

내부적으로:

1. `mlx-whisper-transcribe.py`
2. `mlx-translategemma.py`

순서로 실행됩니다.

### Qwen 전체 파이프라인 (신규 래퍼)

```bash
.venv/bin/python mlx-qwen-translate.py "/path/to/input.mp4" \
  --lang ja \
  --target-lang ko \
  --vad-preprocess \
  --max-seconds 180 \
  --force
```

내부적으로:

1. `mlx-qwen-transcribe.py`
2. `mlx-translategemma.py`

순서로 실행됩니다.

## 5) 자막 정리(공통)

`subtitle_cleanup.py`에서 Whisper/Qwen 공통으로 적용:

- 빈 텍스트 제거
- `end < start` 제거
- `end == start` 구간 최소 길이 보정
- 반복 노이즈 제거
- 중복 문장 제거

## 6) 트러블슈팅

- `ffmpeg` 오류:
  - 설치 확인: `brew install ffmpeg`
- `mlx_*` import 오류:
  - 반드시 `.venv` 인터프리터 사용
  - 예: `.venv/bin/python ...`
- Qwen 모델 401/Repo 오류:
  - 모델 ID 확인
  - HF 인증 확인 (`HF_TOKEN`)
- 번역에 원문이 남는 경우:
  - `mlx-translategemma.py`는 미번역/원문 유지 출력 감지 후 재시도하도록 되어 있음
  - 반복 발생 시 `--max-retries` 증가 고려
