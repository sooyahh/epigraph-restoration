"""Audit impact of title_key-collapse duplicate readings on mask dataset / EXP-013 bins.

Read-only: does not modify readings, mask, or splits CSVs.

Linking criterion (measured):
  - Dup keys = XLS *meta* title_key with >1 row (same as prior audit: 97 keys).
  - Linked readings = rows in readings.csv whose reading-sheet xls_seq (from
    reading_id suffix _x{N} or notes) belongs to an XLS *reading* row whose
    title_key is in that dup-key set.
  - Do NOT map _x{N} onto meta-sheet row N (meta seq ≠ reading seq).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from src.eval.stratified_analysis import FREQ_BUCKETS, _freq_bucket
from src.nrich.xls_parser import load_meta_rows, load_reading_rows

CASE_KSMS = {
    "유명악영세불망비": "14139",
    "척화비": "11739",
    "김명진영세불망비": "13201",
}


def _seq_of(rid: object) -> int | None:
    m = re.search(r"_x(\d+)$", str(rid))
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=Path("data/processed_layer/readings.csv"))
    parser.add_argument(
        "--mask",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument("--splits", type=Path, default=Path("data/model_inputs/splits.csv"))
    parser.add_argument(
        "--xls",
        type=Path,
        default=Path("data/금석문 사용자 검색_2026-07-03.xls"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/title_key_collapse_mask_impact_2026-08-02.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/title_key_collapse_mask_impact_2026-08-02.md"),
    )
    args = parser.parse_args()

    meta = load_meta_rows(args.xls)
    by_meta_key: dict[str, list] = defaultdict(list)
    for m in meta:
        by_meta_key[m.title_key].append(m)
    dups = {k: v for k, v in by_meta_key.items() if len(v) > 1}
    dup_keys = set(dups)
    extra_rows = sum(len(v) - 1 for v in dups.values())

    xls_readings = load_reading_rows(args.xls)
    # reading-sheet rows whose title_key collides with a meta dup key
    dup_xls_readings = [r for r in xls_readings if r.title_key in dup_keys]
    dup_seqs = {r.xls_seq for r in dup_xls_readings}
    seq_to_reading_key = {r.xls_seq: r.title_key for r in xls_readings}

    readings = pd.read_csv(args.readings, low_memory=False)
    mask = pd.read_csv(
        args.mask,
        usecols=[
            "sample_id",
            "reading_id",
            "inscription_id",
            "source_record_id",
            "title",
            "original_text",
            "label_char",
            "split",
        ],
        low_memory=False,
    )
    splits = pd.read_csv(args.splits)

    readings = readings.copy()
    readings["xls_seq"] = readings["reading_id"].map(_seq_of)
    readings["reading_title_key"] = readings["xls_seq"].map(seq_to_reading_key)
    dup_readings = readings[readings["xls_seq"].isin(dup_seqs)].copy()
    dup_reading_ids = set(dup_readings["reading_id"].astype(str))

    mask = mask.copy()
    mask["reading_id"] = mask["reading_id"].astype(str)
    mask["source_record_id"] = mask["source_record_id"].astype(str)
    mask["xls_seq"] = mask["reading_id"].map(_seq_of)
    mask["reading_title_key"] = mask["xls_seq"].map(seq_to_reading_key)
    dup_mask = mask[mask["reading_id"].isin(dup_reading_ids)]

    # --- Item 2: three cases by KSM ---
    cases = {}
    for name, ksm in CASE_KSMS.items():
        rsub = readings[readings["source_record_id"].astype(str) == ksm]
        msub = mask[mask["source_record_id"] == ksm]
        cases[name] = {
            "ksm": ksm,
            "readings_rows": int(len(rsub)),
            "unique_text_normalized": int(rsub["text_normalized"].nunique()),
            "unique_inscription_id": int(rsub["inscription_id"].nunique()),
            "mask_samples": int(len(msub)),
            "mask_by_split": {str(k): int(v) for k, v in msub["split"].value_counts().items()},
            "mask_inscription_ids": sorted(msub["inscription_id"].astype(str).unique().tolist()),
            "mask_splits_unique": sorted(msub["split"].astype(str).unique().tolist()),
        }

    # --- Item 3: overall scale ---
    n_mask_total = len(mask)
    overall = {
        "xls_meta_rows": len(meta),
        "unique_title_key": len({m.title_key for m in meta}),
        "extra_rows": extra_rows,
        "dup_key_count": len(dups),
        "dup_xls_reading_rows": len(dup_xls_readings),
        "dup_linked_readings_rows": int(len(dup_readings)),
        "dup_linked_training_readings": int(
            (dup_readings["is_training_text"] == True).sum()  # noqa: E712
        ),
        "dup_linked_unique_inscriptions": int(dup_readings["inscription_id"].nunique()),
        "dup_linked_mask_samples": int(len(dup_mask)),
        "mask_total": n_mask_total,
        "dup_mask_ratio": round(len(dup_mask) / n_mask_total, 6) if n_mask_total else None,
    }

    # --- Item 5: cross-split within dup reading-title_key groups ---
    cross_split_groups = []
    single_insc_cross = 0
    multi_insc_cross = 0
    for key, msub in dup_mask.groupby("reading_title_key"):
        splits_present = sorted(msub["split"].astype(str).unique().tolist())
        inscs = sorted(msub["inscription_id"].astype(str).unique().tolist())
        if len(splits_present) > 1:
            entry = {
                "title_key": key,
                "splits": splits_present,
                "inscription_ids": inscs,
                "n_inscriptions": len(inscs),
                "mask_n": int(len(msub)),
            }
            cross_split_groups.append(entry)
            if len(inscs) == 1:
                single_insc_cross += 1
            else:
                multi_insc_cross += 1

    # Also: among dup keys that collapsed to 1 inscription, any cross-split?
    collapsed_cross = [
        g for g in cross_split_groups if g["n_inscriptions"] == 1
    ]

    # --- Item 4: frequency bin shift after deduping identical texts in dup groups ---
    # Criterion: within each reading_title_key in dup set, keep ONE training
    # reading_id per text_normalized (first by reading_id). Recount train label freqs.
    train_mask = mask[mask["split"] == "train"]
    counts_before = Counter(train_mask["label_char"].astype(str))

    drop_reading_ids: set[str] = set()
    keep_from_dup: set[str] = set()
    for key, grp in dup_readings.groupby("reading_title_key"):
        train_grp = grp[grp["is_training_text"] == True].sort_values("reading_id")  # noqa: E712
        seen_text: set[str] = set()
        for _, row in train_grp.iterrows():
            text = str(row["text_normalized"])
            rid = str(row["reading_id"])
            if text in seen_text:
                drop_reading_ids.add(rid)
            else:
                seen_text.add(text)
                keep_from_dup.add(rid)

    # Keep: all non-dup reading_ids + kept dup reading_ids
    non_dup_rids = set(readings.loc[~readings["xls_seq"].isin(dup_seqs), "reading_id"].astype(str))
    keep_reading_ids = non_dup_rids | keep_from_dup

    train_after = train_mask[train_mask["reading_id"].isin(keep_reading_ids)]
    counts_after = Counter(train_after["label_char"].astype(str))

    all_chars = set(counts_before) | set(counts_after)
    shifts = []
    for ch in all_chars:
        b = counts_before.get(ch, 0)
        a = counts_after.get(ch, 0)
        bb, aa = _freq_bucket(b), _freq_bucket(a)
        if bb != aa:
            shifts.append(
                {
                    "char": ch,
                    "count_before": b,
                    "bucket_before": bb,
                    "count_after": a,
                    "bucket_after": aa,
                }
            )
    shifts.sort(key=lambda x: (-abs(x["count_before"] - x["count_after"]), x["char"]))

    def bucket_char_counts(counter: Counter) -> dict[str, int]:
        out = {name: 0 for name, _ in FREQ_BUCKETS}
        for ch, c in counter.items():
            out[_freq_bucket(c)] += 1
        return out

    # Per-key inscription collapse stats
    key_insc_stats = []
    for key, grp in dup_readings.groupby("reading_title_key"):
        key_insc_stats.append(
            {
                "title_key": key,
                "readings": int(len(grp)),
                "inscriptions": int(grp["inscription_id"].nunique()),
                "training_readings": int((grp["is_training_text"] == True).sum()),  # noqa: E712
            }
        )
    n_keys_multi_insc = sum(1 for s in key_insc_stats if s["inscriptions"] > 1)
    n_keys_one_insc = sum(1 for s in key_insc_stats if s["inscriptions"] == 1)

    report = {
        "generated_at": date.today().isoformat(),
        "prior_artifact": None,
        "criteria": {
            "dup_keys": "XLS meta title_key with >1 row",
            "dup_readings_link": (
                "XLS reading-sheet rows with title_key in dup set → xls_seq → "
                "readings.reading_id suffix _x{seq}"
            ),
            "freq_dedupe": (
                "Within dup reading title_key groups, keep one is_training_text "
                "reading per (title_key, text_normalized); recount train label_char"
            ),
            "code_unit": "build_artificial_mask iterates training readings row-wise",
        },
        "item1_generation_unit": "reading_row",
        "item1_code_refs": [
            "src/data/build_artificial_mask.py:105-114",
            "src/data/build_artificial_mask.py:46-91",
        ],
        "item2_cases": cases,
        "item3_overall": overall,
        "item3_key_collapse": {
            "dup_keys_with_linked_readings": len(key_insc_stats),
            "keys_with_1_inscription": n_keys_one_insc,
            "keys_with_gt1_inscription": n_keys_multi_insc,
        },
        "item4_freq_shifts": {
            "n_chars_bucket_changed": len(shifts),
            "shifts": shifts,
            "train_mask_before": int(len(train_mask)),
            "train_mask_after_dedupe": int(len(train_after)),
            "dropped_dup_reading_ids": int(len(drop_reading_ids)),
            "bucket_unique_chars_before": bucket_char_counts(counts_before),
            "bucket_unique_chars_after": bucket_char_counts(counts_after),
        },
        "item5_cross_split": {
            "n_dup_groups_with_cross_split_masks": len(cross_split_groups),
            "n_cross_split_with_single_inscription": single_insc_cross,
            "n_cross_split_with_multi_inscription": multi_insc_cross,
            "n_collapsed_one_insc_cross_split": len(collapsed_cross),
            "groups": cross_split_groups,
        },
        "splits_file_rows": int(len(splits)),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# title_key 붕괴 → mask 독립 샘플 영향 감사",
        "",
        f"- 생성일: {report['generated_at']}",
        "- 선행 산출물: 없음 (신규 집계)",
        "- 범위: 조회·집계만 (readings/mask/splits 미수정)",
        "",
        "## 판정 기준",
        "",
        f"- 중복 키: {report['criteria']['dup_keys']}",
        f"- readings 연결: {report['criteria']['dup_readings_link']}",
        f"- 빈도 중복 제거: {report['criteria']['freq_dedupe']}",
        "",
        "## 1. 마스크 생성 단위 (실측·코드)",
        "",
        "**판정: readings 행 단위** (`inscription_id` 중복 제거 없음).",
        "",
        "- `build_artificial_mask_dataframe` L105–114: `training.iterrows()`로 각 reading 행마다 `_masks_for_reading` 호출",
        "- `_masks_for_reading` L46–91: 해당 행의 `text_normalized`에서 마스크 샘플 생성",
        "",
        "## 2. 대표 3사례 (실측)",
        "",
        "| 사례 | KSM | readings | 고유 text | 고유 insc | mask n | split 분포 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, c in cases.items():
        lines.append(
            f"| {name} | {c['ksm']} | {c['readings_rows']} | {c['unique_text_normalized']} | "
            f"{c['unique_inscription_id']} | {c['mask_samples']} | `{c['mask_by_split']}` |"
        )

    o = overall
    lines += [
        "",
        "## 3. 전체 규모 (실측)",
        "",
        f"- 중복 title_key: **{o['dup_key_count']}** (메타 초과 행 {o['extra_rows']})",
        f"- 해당 키의 XLS 판독문 행: **{o['dup_xls_reading_rows']}**",
        f"- 연결 readings 행: **{o['dup_linked_readings_rows']}** "
        f"(training {o['dup_linked_training_readings']}, "
        f"고유 inscription {o['dup_linked_unique_inscriptions']})",
        f"- 연결 mask 샘플: **{o['dup_linked_mask_samples']}** / {o['mask_total']} "
        f"= **{100 * o['dup_mask_ratio']:.4f}%**",
        f"- 연결 readings를 가진 중복 키 중 inscription 1개: "
        f"{report['item3_key_collapse']['keys_with_1_inscription']}, "
        f">1개: {report['item3_key_collapse']['keys_with_gt1_inscription']}",
        "",
        "## 4. EXP-013 빈도 구간 변동 (실측)",
        "",
        f"- train mask before → after dedupe: "
        f"{report['item4_freq_shifts']['train_mask_before']} → "
        f"{report['item4_freq_shifts']['train_mask_after_dedupe']} "
        f"(dropped reading_ids={report['item4_freq_shifts']['dropped_dup_reading_ids']})",
        f"- 구간이 바뀐 한자 수: **{report['item4_freq_shifts']['n_chars_bucket_changed']}**",
        "",
    ]
    if shifts:
        lines += [
            "| char | before | bucket_before | after | bucket_after |",
            "|---|---:|---|---:|---|",
        ]
        for s in shifts:
            lines.append(
                f"| {s['char']} | {s['count_before']} | {s['bucket_before']} | "
                f"{s['count_after']} | {s['bucket_after']} |"
            )
    else:
        lines.append("구간 변동 한자: **0건**")

    lines += [
        "",
        "## 5. Cross-split (실측)",
        "",
        f"- 마스크가 둘 이상 split에 걸친 중복 title_key 그룹: "
        f"**{report['item5_cross_split']['n_dup_groups_with_cross_split_masks']}**",
        f"  - 그중 inscription 1개인데 cross-split: "
        f"**{report['item5_cross_split']['n_cross_split_with_single_inscription']}** "
        f"(기대값 0이면 정상)",
        f"  - inscription 2개 이상이라 cross-split 가능: "
        f"**{report['item5_cross_split']['n_cross_split_with_multi_inscription']}**",
        "",
    ]
    if cross_split_groups:
        lines += [
            "| title_key | splits | n_insc | mask_n |",
            "|---|---|---:|---:|",
        ]
        for g in cross_split_groups:
            lines.append(
                f"| `{g['title_key']}` | {g['splits']} | {g['n_inscriptions']} | {g['mask_n']} |"
            )

    lines += [
        "",
        "## 확인하지 못한 항목",
        "",
        "- 메타 중복 키 97개 중 판독문 시트에 동일 title_key 행이 없는 키는 "
        "readings 연결 0 (판독문 미존재/미매핑). 키별 미연결 목록은 JSON의 "
        "`item3_key_collapse` 규모만 집계.",
        "- EXP-013 Exact@k 수치 자체는 재계산하지 않음 (빈도 구간 소속 변화만).",
        "",
    ]

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("dup_mask", overall["dup_linked_mask_samples"], overall["dup_mask_ratio"])
    print("freq_shifts", len(shifts))
    print("cross_split", report["item5_cross_split"])
    print("cases", {k: v["mask_samples"] for k, v in cases.items()})
    print("Wrote", args.out_json, args.out_md)


if __name__ == "__main__":
    main()
