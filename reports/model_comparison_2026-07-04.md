# Model Comparison Report

> **번호 체계 안내:** 이 문서의 EXP-XXX는 2026-07-04 노트북 작업 시점 번호이며, main의 `docs/EXPERIMENT_LOG.md` 최신 체계(EXP-002~011, 2026-07-06 이후 정리)와 다르다. 이 문서의 **EXP-006(SillokBERT CPU fine-tune)은 main 기준 EXP-011**에 해당한다(`docs/DECISIONS.md` [009] 참고). 그 외 번호(EXP-002~005/00Xb, GPU EXP-007)는 main과 1:1 대응되지 않을 수 있으니 새 실험 번호 부여 전 교차 확인할 것.


- Generated: 2026-07-04
- Purpose: 발표용 통합 비교표 (subsample + full test)

---

## A. Primary comparison (test subsample n=2000, seed=42)

| EXP | Model | Exact@1 | Exact@5 | Exact@10 |
| --- | --- | ---: | ---: | ---: |
| EXP-002b | frequency | 2.30% | 6.75% | 10.75% |
| EXP-003b | ngram | 15.70% | 28.65% | 34.75% |
| EXP-004 | bert-base-chinese zs | 19.75% | 35.55% | 42.45% |
| EXP-005 | SillokBERT zs | 23.65% | 42.60% | 50.90% |
| **EXP-006** | **SillokBERT CPU FT** | **31.30%** | **50.45%** | **57.75%** |

**Official baseline (DECISIONS [009]):** EXP-006 (= main 기준 EXP-011).

---

## B. Full test reference (n=9719)

| EXP | Model | Exact@1 | Exact@5 | Exact@10 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| EXP-002 | frequency | 2.25% | 6.60% | 10.33% | done |
| EXP-003 | ngram | 14.48% | 28.07% | 34.67% | done |
| EXP-004b | bert-base-chinese zs | 19.69% | 34.98% | 42.38% | done (CPU) |
| EXP-005b | SillokBERT zs | 24.89% | 43.17% | 51.22% | done (CPU) |
| EXP-007 | SillokBERT GPU FT | TBD | TBD | TBD | **PC bang GPU** |

Run EXP-007: `python work/run_exp007_gpu.py`

---

## C. EXP-006 era breakdown (subsample)

See [`baseline_results_2026-07-03.md`](baseline_results_2026-07-03.md) § Era breakdown.

**Note:** 통일신라(n=27), 고구려(n=4) 등 소수 era — interpret with caution.

---

## D. Artifacts

| Path | Description |
| --- | --- |
| `outputs/evaluation/evaluation_summary_canonical.csv` | Latest row per EXP |
| `outputs/predictions/exp006_sillok_ft_predictions.csv` | EXP-006 predictions |
| `reports/exp006_case_study.md` | Hit/miss examples |
| `outputs/predictions/real_defect_demo_30.csv` | Real defect demo (30 cases) |
| `reports/gpu_eve_checklist.md` | Pre-PC-bang checklist |
| `reports/pc_bang_runbook.md` | EXP-007 GPU guide |

---

## E. Limitations (all sections)

Inscription split verified; UNK 2.4%; ±64 window; single-char masks only; EXP-006 train 8k subsample.
