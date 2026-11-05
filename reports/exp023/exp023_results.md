# EXP-023 — 층화 커버리지 (빈도 구간별)

- 입력: valid + `exp020_sample_confidence.csv` + train label 빈도 버킷
- 지표: 구간 내부 `top1_score` 스윕; **Exact@5≥90%를 유지하는 최대 커버리지**

## 1. 확신 상위 25% 구성

| 구간 | 전체 비중 | top25% 비중 | lift | n_all | n_top25 |
|---|--:|--:|--:|--:|--:|
| very_common (100+) | 34.1% | 47.3% | 1.39 | 3637 | 1263 |
| common (21-100) | 38.9% | 38.8% | 1.00 | 4158 | 1037 |
| mid (6-20) | 16.2% | 9.8% | 0.61 | 1728 | 263 |
| rare (2-5) | 6.4% | 2.0% | 0.32 | 686 | 54 |
| hapax (count=1) | 2.0% | 1.1% | 0.52 | 214 | 28 |
| unseen_in_train | 2.4% | 0.9% | 0.39 | 256 | 25 |

요약: top25% confidence에서 very_common 47.3% (전체 34.1%, lift 1.389). rare+hapax+unseen 합 4.0%.

## 2. Exact@5≥90% 최대 커버리지 (구간 내부)

| 구간 | n | Exact@5@25% | @50% | @100% | Exact@5≥90% 최대 커버리지 | threshold |
|---|--:|--:|--:|--:|--:|--:|
| very_common (100+) | 3637 | 97.6% | 91.1% | 72.6% | 50% | 0.3802 |
| common (21-100) | 4158 | 90.8% | 78.2% | 55.3% | 25% | 0.6196 |
| mid (6-20) | 1728 | 72.2% | 53.6% | 34.2% | 5% | 0.9592 |
| rare (2-5) | 686 | 36.6% | 26.2% | 17.1% | 달성 불가 (5% 단위 스윕) | — |
| hapax (count=1) | 214 | 33.3% | 20.6% | 12.6% | 달성 불가 (5% 단위 스윕) | — |
| unseen_in_train | 256 | 14.1% | 8.6% | 4.7% | 달성 불가 (5% 단위 스윕) | — |

## 3. 해석 (발표 9 / QA)

- very_common: Exact@5≥90%를 **50%**까지 유지.
- common: 최대 커버리지 25%.
- mid: 최대 커버리지 5%.
- rare/hapax/unseen: Exact@5≥90% **달성 불가** (full Exact@5 각각 17.1% / 12.6% / 4.7%).
- 시사점: 전역 임계값은 고빈도 선택에 유리. 희귀자는 임계값만으로 90%대 Exact@5를 못 만들므로, 구간별 임계값 또는 검색/비예측 보조를 향후 개선·QA 답으로 쓰면 됨.

## 4. 산출물

- `exp023_results.md` / `exp023_summary.json`
- `exp023_top25_composition.csv`
- `exp023_coverage_curves_by_freq.csv`
- `exp023_exact5_90_by_freq.csv`
- `exp023_stratified_coverage_exact5.png` / `.pdf`