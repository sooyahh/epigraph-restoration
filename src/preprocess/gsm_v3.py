"""gsm_preprocess_v3 — NRICH reading normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.nrich.text_utils import collapse_whitespace, is_hangul_char, normalize_unicode

PREPROCESS_RULE_VERSION = "gsm_preprocess_v3"
NORMALIZED_DEFECT_SYMBOL = "□"

RAW_DEFECT_MARKERS = (
    "▨",
    "▢",
    "□",
    "○",
    "◯",
    "〇",
    "●",
    "■",
    "?",
    "？",
)

HANGUL_REPLACEMENT_NOTES = "[hangul->□]"


@dataclass
class PreprocessResult:
    text_raw: str
    text_normalized: str
    text_length: int
    hanja_count: int
    has_real_defect: bool
    defect_count: int
    defect_markers_raw: str
    is_training_text: bool
    exclude_reason: str
    quality_flag: str
    notes: str
    hangul_replaced: int = 0
    observed_markers: set[str] = field(default_factory=set)


def _count_hanja(text: str) -> int:
    count = 0
    for ch in text:
        if unicodedata.category(ch).startswith("Lo") or (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3400" <= ch <= "\u4dbf"
            or "\u20000" <= ch <= "\u2a6df"
            or "\uf900" <= ch <= "\ufaff"
        ):
            count += 1
    return count


def preprocess_reading_text(text_raw: str, source_note: str = "") -> PreprocessResult:
    raw = normalize_unicode(text_raw or "")
    observed: set[str] = set()
    for marker in RAW_DEFECT_MARKERS:
        if marker in raw:
            observed.add(marker)

    normalized_chars: list[str] = []
    hangul_replaced = 0
    for ch in raw:
        if ch in RAW_DEFECT_MARKERS:
            normalized_chars.append(NORMALIZED_DEFECT_SYMBOL)
            continue
        if is_hangul_char(ch):
            normalized_chars.append(NORMALIZED_DEFECT_SYMBOL)
            hangul_replaced += 1
            continue
        normalized_chars.append(ch)

    text_normalized = collapse_whitespace("".join(normalized_chars))
    defect_count = text_normalized.count(NORMALIZED_DEFECT_SYMBOL)
    has_real_defect = defect_count > 0

    quality_flag = "ok"
    notes_parts: list[str] = []
    if source_note:
        notes_parts.append(source_note)
    notes_parts.append(f"rule={PREPROCESS_RULE_VERSION}")

    if hangul_replaced:
        quality_flag = "review"
        notes_parts.append(
            f"hangul_replaced={hangul_replaced} (->□, needs_review)"
        )

    is_training_text = quality_flag == "ok" and not has_real_defect

    marker_parts = sorted(observed)
    if hangul_replaced:
        marker_parts.append(HANGUL_REPLACEMENT_NOTES)

    return PreprocessResult(
        text_raw=raw,
        text_normalized=text_normalized,
        text_length=len(text_normalized),
        hanja_count=_count_hanja(text_normalized),
        has_real_defect=has_real_defect,
        defect_count=defect_count,
        defect_markers_raw=";".join(marker_parts),
        is_training_text=is_training_text,
        exclude_reason="",
        quality_flag=quality_flag,
        notes=" | ".join(notes_parts),
        hangul_replaced=hangul_replaced,
        observed_markers=observed,
    )
