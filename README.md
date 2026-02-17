# mlx-whisper-translate

오디오/영상 파일에서 자막을 만들고, 한국어(또는 원하는 언어)로 번역하는 스크립트입니다.

- 원문 자막: `파일명-original.srt`
- 번역 자막: `파일명.srt`

## 1) 용도

- Whisper(MLX)로 음성 인식 후 SRT 생성
- TranslateGemma(MLX)로 구간별 번역 SRT 생성
- 기존 `-original.srt`가 있으면 Whisper를 생략하고 번역만 수행 가능

## 2) 동작 조건 / 특징

- 지원 입력 확장자: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`
- 입력은 파일 1개 또는 폴더 전체
- `ffmpeg`가 반드시 필요
- `HF_TOKEN`이 없으면 경고만 출력(동작은 가능)
- 자막 정리 로직:
  - 빈 텍스트 구간 제거
  - `end < start` 구간 제거
  - `end == start` 구간은 0.1초 보정
  - 긴 반복 패턴 노이즈 자막 제거 (예: `あ、あ、あ...`, `うっうっうっ...`)
  - 동일 문장은 파일 전체에서 1회만 유지, 이후 중복 제거
- 정리 후 자막 구간이 0개면 번역 단계 자동 생략
- 번역 실패 시 재시도 후 최종 실패하면 원문으로 대체
- 번역 결과의 특수 토큰(`<end_of_turn>` 등) 자동 정리

## 3) 설치

Apple Silicon + Python 가상환경 기준 예시입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlx-whisper mlx-lm tqdm
```

`ffmpeg` 설치:

```bash
brew install ffmpeg
```

권장: Hugging Face 토큰 설정

```bash
export HF_TOKEN="hf_xxx"
```

영구 적용(zsh):

```bash
echo 'export HF_TOKEN="hf_xxx"' >> ~/.zshrc
source ~/.zshrc
```

## 4) 사용 방법

기본 실행:

```bash
python mlx-whisper-translate.py
```

특정 파일 1개:

```bash
python mlx-whisper-translate.py "/path/to/input.mp4"
```

특정 폴더 전체:

```bash
python mlx-whisper-translate.py "/path/to/folder"
```

언어 지정 예시(일본어 음성 -> 한국어 번역):

```bash
python mlx-whisper-translate.py ./target_files --lang ja --target-lang ko
```

Whisper 모델 지정 예시(더 빠른 모델 사용):

```bash
python mlx-whisper-translate.py ./target_files --whisper-model mlx-community/whisper-small
```

Whisper 정확도 우선 모드:

```bash
python mlx-whisper-translate.py ./target_files --whisper-accurate
```

## 5) 주요 옵션

```bash
python mlx-whisper-translate.py --help
```

- `input_path`: 처리할 파일/폴더 경로 (기본 `./target_files`)
- `--whisper-model`: Whisper 모델 (기본 `mlx-community/whisper-medium`)
- `--whisper-accurate`: Whisper 정확도 우선 옵션 사용 (기본은 속도 우선)
- `--lang`: Whisper 입력 언어 코드 (기본 `ja`)
- `--target-lang`: 번역 대상 언어 코드 (기본 `ko`)
- `--progress-every`: 진행 로그 간격 (기본 `10`)
- `--max-retries`: 번역 재시도 횟수 (기본 `2`)
- `--retry-delay`: 재시도 간 대기(초) (기본 `0.5`)
- `--batch-size`: 번역 배치 크기 (기본 `1`, 즉 기본은 배치 번역 비활성)

## 6) 출력 파일 및 덮어쓰기 규칙

입력 파일이 `movie.mp4`라면:

- 원문 자막: `movie-original.srt`
- 번역 자막: `movie.srt`

이미 파일이 있으면 각각 덮어쓰기 여부를 묻습니다.

- `y`/`yes`: 덮어쓰기
- 엔터 포함 그 외 입력: 덮어쓰지 않음

둘 다 덮어쓰지 않으면 해당 파일 처리를 건너뜁니다.

## 7) 성능 팁

- Whisper 기본값은 속도 우선 프리셋 + `mlx-community/whisper-medium`
- 더 빠르게 처리하려면 `--whisper-model mlx-community/whisper-small` 사용
- 정확도를 높이고 싶으면 `--whisper-accurate` 사용(대신 느려질 수 있음)
- 가장 큰 영향: 번역 모델 크기
  - 현재 기본: `mlx-community/translategemma-4b-it-4bit`
- `--max-retries 0`으로 재시도 비활성화 시 속도 개선 가능
- 자막에 반복 문장이 많을수록 번역 캐시로 시간 절약
- 배치 번역은 모델 출력 형식에 따라 분할 실패가 날 수 있으므로 기본 비활성(`--batch-size 1`)

## 8) 문제 해결

- `ffmpeg` 오류: 설치 확인 후 재실행
- 언어 코드 오류: `--lang`, `--target-lang` 값 확인
- HF 경고: `HF_TOKEN` 설정 시 다운로드 속도/요청 한도 개선
