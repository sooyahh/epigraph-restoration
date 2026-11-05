# EXP-017 — ZS/FT 확률 보간 (재학습 없음)

> **상태:** completed · 2026-08-02 · CONTEXT_WINDOW=±64 · DECISIONS [012]  
> 체크포인트: ZS=`ddokbaro/SillokBert`, FT=`notebooks/best/` (EXP-012 best)

## 1. 선행 확인

| # | 항목 | 결과 |
|---|---|---|
| 1 | EXP-007 ZS **valid** 예측 | **없음** → 본 실험에서 top-50 재추론 |
| 2 | EXP-015 FT valid top-k | **top-10** (+scores) |
| 3 | top-50 필요 | **예** — 합집합 후보에 양쪽 softmax 조회 필요 |
| — | EXP 번호 | EXP-016 다음 → **EXP-017** |
| — | subsample | valid n=2,000 seed=42 (`_select_rows`, EXP-016과 동일) |

스크립트: `scripts/exp017_zs_ft_interpolate.py` (기존 `run_fill_mask_eval` / `finetune_*` 미수정)

## 2. α 스윕 (valid subsample n=2,000)

방법: `p_final(c)=α·p_FT(c)+(1−α)·p_ZS(c)`, 후보=각 모델 top-50 단일문자 합집합.

| α | Exact@1 | Exact@5 | Exact@10 | VC@5 | common@5 | mid@5 | rare@5 | hapax@5 | unseen@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 24.85 | 42.95 | 51.55 | 57.37 | 44.34 | 27.16 | 22.67 | **22.58** | 5.08 |
| 0.1 | 26.45 | 49.45 | 58.65 | 66.08 | 51.37 | 31.31 | 25.33 | **22.58** | 5.08 |
| 0.2 | 27.55 | 51.35 | 59.80 | 67.85 | 53.97 | 32.91 | 25.33 | **22.58** | 6.78 |
| 0.3 | 28.70 | 52.15 | 60.15 | 68.29 | 54.88 | 34.50 | 26.00 | **22.58** | 6.78 |
| 0.4 | 30.50 | 52.20 | 60.55 | 68.73 | 54.88 | 34.19 | 25.33 | **22.58** | 6.78 |
| 0.5 | 32.30 | 52.60 | 61.10 | 69.47 | 55.01 | 35.14 | 24.67 | **22.58** | 6.78 |
| 0.6 | 33.25 | 52.75 | 60.90 | 69.47 | 55.40 | 35.78 | 24.00 | 19.35 | 6.78 |
| 0.7 | 34.00 | 52.95 | 60.60 | 69.62 | 55.79 | 36.74 | 22.00 | 19.35 | 6.78 |
| **0.8** | 34.20 | **53.45** | 60.70 | 70.50 | 56.44 | 36.74 | 22.00 | 16.13 | 6.78 |
| 0.9 | 34.50 | 53.25 | 60.85 | 70.21 | 56.44 | 36.74 | 21.33 | 16.13 | 5.08 |
| 1.0 | 34.30 | 53.40 | 60.70 | 70.94 | 56.83 | 36.74 | 18.67 | 16.13 | 3.39 |

구간 n (subsample): VC 678 / common 769 / mid 313 / rare 150 / **hapax 31** / unseen 59.

### Sanity: α=1.0 ≡ EXP-015 (동일 subsample)

| 지표 | α=1.0 | EXP-015 필터 | |Δ| |
|---|---:|---:|---:|
| Exact@1 | 34.30% | 34.30% | 0 |
| Exact@5 | 53.40% | 53.40% | 0 |
| Exact@10 | 60.70% | 60.70% | 0 |

α=1.0 Exact@5 ≡ 동일 dist의 `ft_top` Exact@5 (내부 일치).

## 3. α* 선정과 트레이드오프

| 후보 | 선정 이유 | subsample Exact@5 | hapax@5 (n=31) | vs FT |
|---|---|---:|---:|---|
| **α=0.8** (1차·마일스톤) | overall Exact@5 최댓값 | 53.45% | 16.13% | overall +0.05pp, hapax 동일 |
| **α=0.5** (트레이드오프) | hapax 고원(0.0–0.5) 끝에서 overall 최대 | 52.60% | 22.58% | overall −0.80pp, hapax +6.45pp |

기계적 overall 최댓값(0.8)은 FT와 거의 같고 hapax 이득이 없다. hapax를 살리려면 overall을 희생하는 0.5 쪽이 해당하나, hapax n=31이라 불안정.

**확정 α\*=0.8** (overall 1차 후보)로 full valid 마일스톤 수행.

## 4. Paired 비교 (vs EXP-015 / α=1.0, McNemar continuity)

### Subsample n=2,000

| 후보 | scope | n | Exact@5 Δpp | only_FT / only_α | p |
|---|---|---:|---:|---|---:|
| α=0.8 | overall | 2000 | **+0.05** | — | **1.00** |
| α=0.8 | hapax | 31 | 0.00 | — | 1.00 |
| α=0.8 | rare | 150 | +3.33 | — | 0.074 |
| α=0.5 | overall | 2000 | −0.80 | 78 / 62 | 0.205 |
| α=0.5 | Exact@1 overall | 2000 | −2.00 | 100 / 60 | **0.002** (악화) |
| α=0.5 | hapax | 31 | +6.45 | — | 0.480 |
| α=0.5 | rare | 150 | +6.00 | — | **0.008** (개선) |

### Full valid n=10,679 (α=0.8)

| scope | n | Exact@5 Δpp | p |
|---|---:|---:|---:|
| overall | 10679 | **−0.028** | 0.917 |
| hapax | 214 | +1.40 | 0.371 |
| rare | 686 | +1.02 | 0.121 |

## 5. Full valid 마일스톤 (n=10,679)

| α | Exact@1 | Exact@5 | Exact@10 | hapax@5 | rare@5 |
|---:|---:|---:|---:|---:|---:|
| **0.8** (확정) | 33.78% | 53.21% | 61.21% | 14.02% | 18.08% |
| 1.0 (=FT/EXP-015) | **33.78%** | **53.24%** | 60.90% | 12.62% | 17.06% |
| 0.5 (참고) | 32.03% | 52.52% | 60.93% | 16.36% | 18.95% |

α=1.0 full Exact@5 **53.235% = EXP-015 공식값과 일치** (재확인).

**결론:** 재학습 없는 ZS/FT 선형 보간은 **overall Exact@5를 FT 대비 유의하게 개선하지 못한다.** subsample의 α=0.8 +0.05pp는 full에서 사라짐(−0.03pp). α=0.5는 rare에서 subsample 유의 개선이 있으나 overall Exact@1 유의 하락·full에서도 overall 열위.

→ **운영/공식 후보는 EXP-015(FT, α=1.0) 유지.**

## 6. 확인하지 못한 항목

| 항목 | 이유 |
|---|---|
| GPU 재현 | 로컬 CUDA 미가용 (cpu) |
| top-k > 50 / 전 vocab 보간 | 계산·저장 비용; top-50 합집합으로 한정 |
| hapax 유의성 | subsample n=31, full n=214 — 검출력 제한 |
| 학습률/epoch 등 HP | 본 실험 범위 밖 (EXP-018) |

## 7. 다음 단계 제안

1. **EXP-018 HP 재탐색**을 주력으로 진행 (단일 마스크·±64 유지, EXP-016 정합).
2. 보간은 **채택하지 않음**. rare/hapax 특화가 제품 우선이면 별도 **구간 라우팅·재순위** 실험을 새 EXP로 (선형 α 스윕과는 분리).
3. test 예산은 보간 후보로는 **쓰지 말 것**.

## 산출물

| 경로 | 내용 |
|---|---|
| `scripts/exp017_zs_ft_interpolate.py` | infer / sweep / paired / milestone |
| `outputs/predictions/exp017_dist_valid_sub2000_topk50.jsonl` | subsample 분포 |
| `outputs/predictions/exp017_dist_valid_full_topk50.jsonl` | full valid 분포 |
| `outputs/predictions/exp017_alpha*_valid_sub_predictions.csv` | α별 subsample 예측 |
| `outputs/predictions/exp017_alpha{0.5,0.8,1.0}_valid_full_predictions.csv` | full 예측 |
| `reports/exp017/exp017_alpha_sweep.json` / `.csv` | 스윕 |
| `reports/exp017/paired_*.json` | paired |
| `reports/exp017/exp017_milestone_alpha*.json` | 마일스톤 |
