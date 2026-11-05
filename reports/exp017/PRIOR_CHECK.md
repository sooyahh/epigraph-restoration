# EXP-017 선행 확인 (2026-08-02)

| # | 항목 | 결과 |
|---|---|---|
| 1 | EXP-007 ZS valid 예측 | **없음** (test만; valid 재추론 필요) |
| 2 | EXP-015 FT valid top-k | **top-10** (+ scores) |
| 3 | top-50 필요 여부 | **예** — top-10 합집합만으로는 교차 누락 가능; top-50 + 합집합에 대한 양쪽 softmax 조회 |

- 실험 번호: EXP-016 다음 → **EXP-017**
- CONTEXT_WINDOW=64 고정 (EXP-016)
- subsample: valid n=2000 seed=42 (_select_rows, EXP-016과 동일)
- device: cpu (CUDA 미가용)
- 스크립트: `scripts/exp017_zs_ft_interpolate.py` (기존 src 미수정)
