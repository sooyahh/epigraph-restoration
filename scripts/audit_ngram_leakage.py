"""Audit cross-split n-gram context overlap (memorization / leakage risk)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_NGRAMS = (8, 4, 2, 1)
CONTEXT_WINDOW = 64


def _context_keys(text: str, pos: int, ngram_sizes: tuple[int, ...]) -> set[tuple[str, int, str]]:
    left = text[max(0, pos - CONTEXT_WINDOW) : pos]
    right = text[pos + 1 : pos + 1 + CONTEXT_WINDOW]
    keys: set[tuple[str, int, str]] = set()
    for n in ngram_sizes:
        if len(left) >= n:
            keys.add(("left", n, left[-n:]))
        if len(right) >= n:
            keys.add(("right", n, right[:n]))
    return keys


def _build_train_key_index(
    train_df: pd.DataFrame,
    ngram_sizes: tuple[int, ...],
) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for _, row in train_df.iterrows():
        text = str(row["original_text"])
        pos = int(row["mask_position"])
        keys.update(_context_keys(text, pos, ngram_sizes))
    return keys


def _row_overlap_flags(
    text: str,
    pos: int,
    train_keys: set[tuple[str, int, str]],
    ngram_sizes: tuple[int, ...],
) -> dict[str, bool | int]:
    keys = _context_keys(text, pos, ngram_sizes)
    matched = keys & train_keys
    return {
        "num_matching_keys": len(matched),
        "has_train_ngram_match": bool(matched),
        "has_train_2gram_match": any(n_size >= 2 for _, n_size, _ in matched),
        "has_train_4gram_match": any(n_size >= 4 for _, n_size, _ in matched),
        "has_left_match": any(side == "left" for side, _, _ in matched),
        "has_right_match": any(side == "right" for side, _, _ in matched),
    }


def build_per_sample_overlap_flags(
    mask_csv: Path,
    *,
    split: str = "test",
    ngram_sizes: tuple[int, ...] = DEFAULT_NGRAMS,
) -> pd.DataFrame:
    """Return per-sample overlap flags for eval split rows."""
    df = pd.read_csv(mask_csv)
    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == split].copy()
    train_keys = _build_train_key_index(train_df, ngram_sizes)

    rows: list[dict] = []
    for _, row in eval_df.iterrows():
        text = str(row["original_text"])
        pos = int(row["mask_position"])
        flags = _row_overlap_flags(text, pos, train_keys, ngram_sizes)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": split,
                "label_char": row.get("label_char", ""),
                **flags,
            }
        )
    return pd.DataFrame(rows)


def audit_cross_split_ngram_overlap(
    mask_csv: Path,
    *,
    split: str = "test",
    ngram_sizes: tuple[int, ...] = DEFAULT_NGRAMS,
) -> dict:
    df = pd.read_csv(mask_csv)
    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == split]
    train_keys = _build_train_key_index(train_df, ngram_sizes)

    total = len(eval_df)
    any_match = 0
    any_match_n_ge_2 = 0
    left_match = 0
    right_match = 0
    by_n: dict[int, dict[str, int]] = {
        n: {"matched_samples": 0, "total_samples": total} for n in ngram_sizes
    }
    match_counts: list[int] = []

    for _, row in eval_df.iterrows():
        text = str(row["original_text"])
        pos = int(row["mask_position"])
        flags = _row_overlap_flags(text, pos, train_keys, ngram_sizes)
        n_matched = int(flags["num_matching_keys"])
        match_counts.append(n_matched)
        if flags["has_train_ngram_match"]:
            any_match += 1
        if flags["has_train_2gram_match"]:
            any_match_n_ge_2 += 1
        if flags["has_left_match"]:
            left_match += 1
        if flags["has_right_match"]:
            right_match += 1
        keys = _context_keys(text, pos, ngram_sizes)
        matched = keys & train_keys
        for n in ngram_sizes:
            if any(side_n == n for _, side_n, _ in matched):
                by_n[n]["matched_samples"] += 1

    return {
        "split": split,
        "ngram_sizes": list(ngram_sizes),
        "context_window": CONTEXT_WINDOW,
        "num_train_samples": len(train_df),
        "num_eval_samples": total,
        "samples_with_any_train_ngram_match": any_match,
        "fraction_with_any_match": any_match / total if total else 0.0,
        "samples_with_match_n_ge_2": any_match_n_ge_2,
        "fraction_with_match_n_ge_2": any_match_n_ge_2 / total if total else 0.0,
        "samples_with_left_match": left_match,
        "samples_with_right_match": right_match,
        "avg_matching_keys_per_sample": sum(match_counts) / total if total else 0.0,
        "by_ngram_size": {
            str(n): {
                "matched_samples": by_n[n]["matched_samples"],
                "fraction": by_n[n]["matched_samples"] / total if total else 0.0,
            }
            for n in ngram_sizes
        },
    }


def render_markdown(payload: dict) -> str:
    frac = payload["fraction_with_any_match"] * 100
    lines = [
        "# Cross-split n-gram context overlap audit",
        "",
        f"- **생성일:** {date.today().isoformat()}",
        f"- **평가 split:** {payload['split']}",
        f"- **train samples:** {payload['num_train_samples']:,}",
        f"- **eval samples:** {payload['num_eval_samples']:,}",
        f"- **n-gram sizes:** {payload['ngram_sizes']}",
        f"- **context window:** {payload['context_window']}",
        "",
        "## 요약",
        "",
        f"- test 샘플 중 train과 **동일 좌/우 n-gram 키(길이≥2)** 를 가진 비율: "
        f"**{payload.get('samples_with_match_n_ge_2', 0):,} / "
        f"{payload['num_eval_samples']:,} "
        f"({payload.get('fraction_with_match_n_ge_2', 0) * 100:.1f}%)**",
        f"- test 샘플 중 train과 **어떤 n-gram 키든(길이≥1)** 매칭: "
        f"{payload['samples_with_any_train_ngram_match']:,} "
        f"({frac:.1f}%) — unigram(1-gram) 포함 시 거의 전부",
        f"- left context match: {payload['samples_with_left_match']:,}",
        f"- right context match: {payload['samples_with_right_match']:,}",
        f"- 샘플당 평균 매칭 키 수: {payload['avg_matching_keys_per_sample']:.2f}",
        "",
        "## n-gram 크기별 (test 샘플 기준)",
        "",
        "| n | matched samples | fraction |",
        "|--:|----------------:|---------:|",
    ]
    for n, row in payload["by_ngram_size"].items():
        lines.append(
            f"| {n} | {row['matched_samples']:,} | {row['fraction'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- inscription 단위 split은 **동일 비문** 누수를 막지만, **정형구·반복 문맥**은 "
            "train/test에 걸쳐 n-gram 키가 겹칠 수 있다.",
            "- ngram baseline(EXP-004)은 train에서 본 context key를 lookup하므로, "
            "위 비율이 높으면 test 성능 일부는 **memorization**일 수 있다.",
            "- BERT/SillokBert zero-shot은 lookup을 쓰지 않지만, 동일 문맥 반복은 "
            "여전히 쉬운 케이스를 늘릴 수 있다 → 층화 분석과 함께 해석한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--export-flags-csv",
        type=Path,
        default=None,
        help="Optional per-sample overlap flags for overlap-adjusted eval.",
    )
    args = parser.parse_args()

    if args.export_flags_csv:
        flags_df = build_per_sample_overlap_flags(args.mask_dataset, split=args.split)
        args.export_flags_csv.parent.mkdir(parents=True, exist_ok=True)
        flags_df.to_csv(args.export_flags_csv, index=False)
        print(f"Wrote {len(flags_df)} overlap flag rows to {args.export_flags_csv}")

    payload = audit_cross_split_ngram_overlap(args.mask_dataset, split=args.split)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
