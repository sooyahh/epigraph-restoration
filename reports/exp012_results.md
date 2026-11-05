# EXP-012 Results Snapshot — SillokBERT GPU full-data FT + full test

> **상태:** 완료 · 채택 DECISIONS [010] · 기록 `docs/EXPERIMENT_LOG.md` EXP-012  
> **실행일:** 2026-07-12 · **환경:** Google Colab GPU (`device=cuda`)  
> 모델·대용량 예측 CSV는 Git에 넣지 않는다. 로컬 백업: `notebooks/best/`

## 요약

| 항목 | 값 |
|------|-----|
| Exact@1 | **33.75%** |
| Exact@5 | **53.04%** |
| Exact@10 | **60.47%** |
| test n | **9,719** (1회) |
| best_epoch | **1** |
| best_valid_loss | **3.66192810216807** |
| run_mode | `full` (`full_data_no_row_cap=true`) |

동일 test n=9,719에서 EXP-007 zero-shot(24.89% / 43.17% / 51.22%) 대비 Exact@1/5/10 상승.  
EXP-011(n=2,000, Exact@5 50.45%)과는 **표본 크기가 달라 직접 동일시·병기하지 않는다.**

## 학습 설정

- 모델: `ddokbaro/SillokBert`
- 데이터: `artificial_mask_dataset.csv` — train 46,139 / valid 10,679 / test 9,719
- epochs=2, batch_size=16, lr=5e-5, max_seq_len=512, context ±64
- train_steps=5620 (epoch당 2810)
- 선택: `minimize_valid_loss` (test Exact@k로 checkpoint 재선택 금지)
- epoch 2 valid_loss ≈ 3.770 (미채택)

## 조선 층화 (참고)

| era | n | Exact@1 | Exact@5 | Exact@10 |
|-----|--:|--------:|--------:|---------:|
| 조선 | 7,425 | 35.87% | 55.39% | 62.67% |

출처: `notebooks/best/evaluation_summary.csv`. EXP-011 조선(n=1,546)과 표본이 다름.

## 정답 글자 빈도 층화 (2026-08-01 추가)

zero-shot(EXP-007) → fine-tuned(EXP-012) Exact@5 변화:

| 빈도 구간 | n | EXP-007 zs | EXP-012 FT | 차이 |
|---|--:|--:|--:|--:|
| very_common (100+) | 3,364 | 58.0% | **71.7%** | +13.7pp |
| common (21-100) | 3,690 | 44.9% | **55.5%** | +10.6pp |
| mid (6-20) | 1,588 | 27.6% | **34.1%** | +6.5pp |
| rare (2-5) | 602 | 20.8% | **22.1%** | +1.3pp |
| hapax (count=1) | 232 | 8.2% | 6.0% | **−2.2pp** |
| unseen_in_train | 243 | 3.7% | 3.7% | 0.0pp |

**해석:** fine-tuning 이득은 train 빈도에 비례하며, **hapax 구간은 오히려 하락**했다(Exact@10도 10.8%→9.1%).
unseen은 변화 없음. 즉 fine-tuning은 학습 분포에 모델을 맞춰 빈출 한자 정확도를 크게 올린 대신
극희귀자 일반화는 zero-shot보다 나빠졌다. 전체 평균 +9.9pp를 "모든 구간 개선"으로 서술하면 안 된다.

출처: `reports/phase4_stratified_analysis_2026-08-01.md` / `.json`
(`src/eval/stratified_analysis.py`, EXP-008과 동일 구간 정의)

## 산출물 (로컬)

경로: `notebooks/best/`

- `best_checkpoint.json`, `finetune_metrics.json`, `epoch_metrics.json`
- `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_state.pt`
- `evaluation_summary.csv`, `exp012_sillok_ft_full_predictions.csv` (9,719행)

Colab 원 경로: `/content/exp012/sillokbert_exp012_gpu/` (세션 종료 시 소멸 가능).  
EXP-011 `outputs/finetune/sillokbert_gsm_baseline/` 와 분리·미덮어쓰기.

**무결성·백업:** 위 파일 전체의 SHA-256은 `reports/exp012_artifacts/MANIFEST.md`에 기록했다.
`notebooks/best/`는 현재 **단일 로컬 사본**이며 외부 백업은 미완료다(MANIFEST §3 체크리스트).

## 기각 규칙

- Colab smoke (`_exp012_smoke`, `run_mode=smoke`, steps≈4) — **공식 성능 아님**
- smoke와 full의 동명 파일(`best_checkpoint.json` 등) 혼동 금지
- Cloud CPU / 부분 실행 수치 — 성능 근거로 사용 금지

## Exact@k 재검증 (2026-07-13)

`work/reverify_exp012_metrics.py`로 `exp012_sillok_ft_full_predictions.csv`를 재집계한 결과:

| 항목 | 값 | 대조 |
|------|-----|------|
| Exact@1 / @5 / @10 | 33.75% / 53.04% / 60.47% | `evaluation_summary.csv`·공식 수치와 **일치** |
| n | 9,719 | 일치 |
| `model_version` | `exp012_sillokbert_gpu_best_valid_loss` | 백필 완료 |
| best_epoch / best_valid_loss | 1 / ≈3.6619 | `finetune_metrics.json`·`best_checkpoint.json`과 일치 |

산출 JSON: `reports/exp012_metrics_reverify.json`

## 관련 문서

- `docs/EXPERIMENT_LOG.md` EXP-012
- `docs/DECISIONS.md` [010]
- `reports/colab_exp012_runbook.md`
- `notebooks/exp012_colab.ipynb`
- `reports/real_defect_exp012_runbook.md`
