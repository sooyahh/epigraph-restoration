"""Shared mask-window helpers for training and evaluation."""

from __future__ import annotations

CONTEXT_WINDOW = 64
MAX_SEQ_LEN = 512


def build_masked_window(
    original_text: str,
    mask_position: int,
    label_char: str,
    mask_token: str,
    *,
    context_window: int = CONTEXT_WINDOW,
) -> tuple[str, int] | None:
    """Return (window_text_with_mask, local_mask_position) or None if invalid."""
    if mask_position < 0 or mask_position >= len(original_text):
        return None
    if original_text[mask_position] != label_char:
        return None

    left = max(0, mask_position - context_window)
    right = min(len(original_text), mask_position + context_window + 1)
    window = original_text[left:right]
    local_mask = mask_position - left
    masked_window = window[:local_mask] + mask_token + window[local_mask + 1 :]
    return masked_window, local_mask
