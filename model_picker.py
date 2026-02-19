from __future__ import annotations

from dataclasses import dataclass

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


@dataclass(frozen=True)
class ModelQuery:
    search: str
    must_contain: str
    title: str


MODEL_QUERIES: dict[str, ModelQuery] = {
    "whisper": ModelQuery(search="whisper", must_contain="whisper", title="Whisper"),
    "qwen3_asr": ModelQuery(search="Qwen3-ASR", must_contain="qwen3-asr", title="Qwen3-ASR"),
    "translategemma": ModelQuery(
        search="translategemma", must_contain="translategemma", title="TranslateGemma"
    ),
}


WHISPER_REPRESENTATIVE_ORDER = [
    # Balanced defaults by size (multilingual)
    "mlx-community/whisper-tiny-4bit",
    "mlx-community/whisper-base-4bit",
    "mlx-community/whisper-small-4bit",
    "mlx-community/whisper-medium-4bit",
    "mlx-community/whisper-medium",
    # Higher quality / faster large variants
    "mlx-community/whisper-large-v3-turbo-4bit",
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-mlx-4bit",
    "mlx-community/whisper-large-v3-mlx",
    # Distilled variants
    "mlx-community/distil-whisper-large-v3",
    "mlx-community/distil-whisper-medium.en",
]


def fetch_mlx_models(kind: str, limit: int = 100) -> list[str]:
    if HfApi is None:
        raise RuntimeError("huggingface_hub가 설치되어 있지 않습니다.")
    if kind not in MODEL_QUERIES:
        raise ValueError(f"지원하지 않는 모델 카테고리입니다: {kind}")

    query = MODEL_QUERIES[kind]
    api = HfApi()
    results = api.list_models(author="mlx-community", search=query.search, limit=limit)

    model_ids = {
        model.id
        for model in results
        if getattr(model, "id", None)
        and query.must_contain in str(model.id).lower()
    }
    return sorted(model_ids, key=str.lower)


def filter_representative_models(kind: str, models: list[str]) -> list[str]:
    if kind != "whisper":
        return models

    available = set(models)
    filtered = [model_id for model_id in WHISPER_REPRESENTATIVE_ORDER if model_id in available]
    return filtered or models


def choose_model_from_list(title: str, default_model: str, models: list[str]) -> str:
    print(f"\n--- {title} 모델 선택 ---")
    for i, model_id in enumerate(models, start=1):
        suffix = " (기본값)" if model_id == default_model else ""
        print(f"{i}. {model_id}{suffix}")

    while True:
        choice = input(
            f"번호를 선택하세요 (1-{len(models)}, Enter=기본값 '{default_model}', q=건너뛰기): "
        ).strip()
        if not choice:
            return default_model if default_model in models else models[0]
        if choice.lower() in ("q", "quit", "skip"):
            return default_model
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        print("잘못된 입력입니다. 번호를 다시 입력하세요.")


def choose_model_online(kind: str, default_model: str) -> str:
    if kind not in MODEL_QUERIES:
        return default_model

    query = MODEL_QUERIES[kind]
    try:
        all_models = fetch_mlx_models(kind=kind)
        models = filter_representative_models(kind=kind, models=all_models)
    except Exception as e:
        print(f"경고: 온라인 모델 목록 조회에 실패했습니다. 기본 모델을 사용합니다. ({e})")
        return default_model

    if not models:
        print("경고: 조회된 모델 목록이 없습니다. 기본 모델을 사용합니다.")
        return default_model

    if default_model and default_model not in models:
        models = [default_model] + models

    if kind == "whisper" and len(models) < len(all_models):
        print(f"안내: Whisper 대표 모델 {len(models)}개만 표시합니다. (전체 {len(all_models)}개)")

    selected = choose_model_from_list(query.title, default_model=default_model, models=models)
    print(f"선택된 모델: {selected}\n")
    return selected
