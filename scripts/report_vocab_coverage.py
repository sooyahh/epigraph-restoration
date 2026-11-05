"""Report tokenizer vocabulary coverage on training corpus."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    if unicodedata.category(ch).startswith("Lo"):
        return True
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
    )


def _corpus_chars(readings_csv: Path, *, training_only: bool) -> Counter[str]:
    df = pd.read_csv(readings_csv)
    if training_only:
        df = df[df["is_training_text"] == True]  # noqa: E712
    counter: Counter[str] = Counter()
    for text in df["text_normalized"].fillna("").astype(str):
        for ch in text:
            if ch.isdigit() or ch.isspace():
                continue
            if _is_cjk(ch):
                counter[ch] += 1
    return counter


def _analyze_tokenizer(
    model_name: str,
    char_counts: Counter[str],
) -> dict:
    tok = AutoTokenizer.from_pretrained(model_name)
    unk_id = tok.unk_token_id
    total_occ = sum(char_counts.values())
    unique_chars = len(char_counts)
    unk_chars: list[str] = []
    unk_occ = 0
    multi_token_chars = 0
    for ch, count in char_counts.items():
        ids = tok.encode(ch, add_special_tokens=False)
        is_unk = unk_id is not None and unk_id in ids
        is_multi = len(ids) != 1
        if is_unk:
            unk_chars.append(ch)
            unk_occ += count
        if is_multi and not is_unk:
            multi_token_chars += 1
    unk_chars.sort(key=lambda c: char_counts[c], reverse=True)
    return {
        "model_name": model_name,
        "vocab_size": tok.vocab_size,
        "unique_cjk_chars": unique_chars,
        "total_cjk_occurrences": total_occ,
        "unk_char_types": len(unk_chars),
        "unk_char_type_rate": len(unk_chars) / unique_chars if unique_chars else 0.0,
        "unk_occurrence_rate": unk_occ / total_occ if total_occ else 0.0,
        "multi_token_char_types": multi_token_chars,
        "top_unk_chars": [
            {"char": ch, "count": char_counts[ch]} for ch in unk_chars[:30]
        ],
    }


def render_markdown(results: list[dict], *, training_only: bool) -> str:
    scope = "is_training_text=True" if training_only else "all readings"
    lines = [
        "# Tokenizer vocabulary coverage report",
        "",
        f"- **생성일:** {date.today().isoformat()}",
        f"- **corpus scope:** {scope}",
        "",
        "## 요약",
        "",
        "| 모델 | unique CJK types | UNK types | UNK type % | UNK occurrence % |",
        "|------|----------------:|----------:|-----------:|-----------------:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model_name']} | {row['unique_cjk_chars']:,} | "
            f"{row['unk_char_types']:,} | {row['unk_char_type_rate'] * 100:.2f}% | "
            f"{row['unk_occurrence_rate'] * 100:.2f}% |"
        )
    lines.append("")
    for row in results:
        lines.append(f"## {row['model_name']} — top UNK characters")
        lines.append("")
        if not row["top_unk_chars"]:
            lines.append("(none)")
        else:
            for item in row["top_unk_chars"][:15]:
                lines.append(f"- `{item['char']}` (count={item['count']:,})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings",
        type=Path,
        default=Path("data/processed_layer/readings.csv"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-chinese", "ddokbaro/SillokBert"],
    )
    parser.add_argument("--training-only", action="store_true", default=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    char_counts = _corpus_chars(args.readings, training_only=args.training_only)
    results = [_analyze_tokenizer(name, char_counts) for name in args.models]
    payload = {
        "training_only": args.training_only,
        "models": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_markdown(results, training_only=args.training_only),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
