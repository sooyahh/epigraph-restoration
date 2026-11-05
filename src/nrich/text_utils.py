"""Shared text normalization helpers for NRICH titles and readings."""

from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text or "")


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def normalize_korean_title(title: str) -> str:
    """Korean-title lookup key: NFC, then remove every whitespace char (incl. U+3000).

    Same behavior as the former ``nrich_common.normalize_korean_title``; Python's
    ``\\s`` already matches U+3000, so ``collapse_whitespace`` covers its regex.
    """
    return collapse_whitespace(normalize_unicode(title))


def normalize_title_key(title_ko: str, title_hanja: str) -> str:
    ko = normalize_unicode(str(title_ko or "").strip())
    han = collapse_whitespace(normalize_unicode(str(title_hanja or "").strip(), "NFKC"))
    return f"{ko}|{han}"


def parse_list_title(combined: str) -> tuple[str, str]:
    """Parse list-page title like '한글(漢字)' into components."""
    text = normalize_unicode(combined.strip())
    if "(" in text and text.endswith(")"):
        ko, han = text.rsplit("(", 1)
        return ko.strip(), han[:-1].strip()
    return text, ""


def is_hangul_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
    )


def is_cjk_ideograph(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )
