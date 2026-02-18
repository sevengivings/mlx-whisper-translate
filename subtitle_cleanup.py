import re
from typing import Any

REPETITIVE_FILTER_MIN_CHARS = 40
REPETITIVE_FILTER_MIN_REPEATS = 12
REPETITIVE_FILTER_SUFFIX_MIN_CHARS = 24
REPETITIVE_FILTER_SUFFIX_MAX_UNIT_LEN = 3
REPETITIVE_FILTER_TOKEN_MAX_LEN = 4


def is_repetitive_noise_text(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    normalized = re.sub(r"[、。,.!?！？…・「」『』（）()\[\]{}\"'`~^_|\\/:\-+=*#@$%&;]+", "", normalized)

    if len(normalized) < REPETITIVE_FILTER_MIN_CHARS:
        return False

    if re.search(r"(.)\1{23,}$", normalized):
        return True

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

    for unit_len in range(1, REPETITIVE_FILTER_SUFFIX_MAX_UNIT_LEN + 1):
        if len(normalized) < unit_len * REPETITIVE_FILTER_MIN_REPEATS:
            continue
        unit = normalized[-unit_len:]
        repeat_chars = 0
        idx = len(normalized)
        while idx >= unit_len and normalized[idx - unit_len : idx] == unit:
            repeat_chars += unit_len
            idx -= unit_len
        if repeat_chars >= REPETITIVE_FILTER_SUFFIX_MIN_CHARS:
            return True

    tokens = re.findall(r"[^\s、。,.!?！？…・「」『』（）()\[\]{}\"'`~^_|\\/:\-+=*#@$%&;]+", text)
    if len(tokens) >= REPETITIVE_FILTER_MIN_REPEATS:
        last = tokens[-1]
        if len(last) <= REPETITIVE_FILTER_TOKEN_MAX_LEN:
            run = 1
            idx = len(tokens) - 2
            while idx >= 0 and tokens[idx] == last:
                run += 1
                idx -= 1
            if run >= REPETITIVE_FILTER_MIN_REPEATS:
                return True

    return False


def sanitize_segments(
    segments: list[dict[str, Any]],
    min_duration: float = 0.1,
    drop_repetitive: bool = True,
    drop_duplicate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cleaned = []
    dropped_empty = 0
    dropped_invalid = 0
    fixed_duration = 0
    dropped_duplicate = 0
    dropped_repetitive = 0
    seen_texts = set()

    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            dropped_invalid += 1
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            dropped_empty += 1
            continue

        if drop_repetitive and is_repetitive_noise_text(text):
            dropped_repetitive += 1
            continue

        if drop_duplicate:
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
