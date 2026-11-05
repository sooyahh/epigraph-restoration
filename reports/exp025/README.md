# EXP-025 — same-200 paired Exact + McNemar (발표용 P0)

EXP-024 eval `n=200`의 `sample_id`에 EXP-015(SillokBERT) · EXP-021(ARI-8B ZS beam5) · EXP-024(ARI-8B QLoRA FT beam5) 예측을 inner join한 **동일 표본** 비교.
공식 ARI Top-K / SOTA / Appendix C 주장이 아님. Top-5는 EXP-021/024와 같은 단순화 beam5.

## 입력 (Drive)

| 파일 | 출처 |
|---|---|
| `inputs/exp024_qlora_beam5_n200_predictions.csv` | `MyDrive/exp024/eval/` |
| `inputs/exp021_g3_beam5_n500_predictions.csv` | `MyDrive/exp021/` |
| `inputs/exp015_sillok_ft_valid_predictions.csv` | `MyDrive/exp021/` (valid 전체; 비교는 200개만 join) |

## 동일 표본 Exact (n=200)

| 모델 | Exact@1 | Exact@5 | Exact@10 |
|---|---:|---:|---:|
| EXP-015 SillokBERT | 0.350 | 0.515 | 0.575 |
| EXP-021 ARI-8B ZS | 0.105 | 0.225 | 0.225 |
| EXP-024 ARI-8B QLoRA FT | 0.220 | 0.410 | 0.410 |

ARI beam5라 Exact@10 ≈ Exact@5.

## McNemar (continuity correction) — B = EXP-024 FT

| 비교 | k | Exact A | Exact B | Δ(B−A) pp | χ² | p |
|---|---:|---:|---:|---:|---:|---:|
| EXP-015 vs EXP-024 | 1 | 0.350 | 0.220 | −13.0 | 14.20 | 1.64e-4 |
| EXP-015 vs EXP-024 | 5 | 0.515 | 0.410 | −10.5 | 7.84 | 5.10e-3 |
| EXP-015 vs EXP-024 | 10 | 0.575 | 0.410 | −16.5 | 19.32 | 1.10e-5 |
| EXP-021 vs EXP-024 | 1 | 0.105 | 0.220 | +11.5 | 14.67 | 1.28e-4 |
| EXP-021 vs EXP-024 | 5 | 0.225 | 0.410 | +18.5 | 28.80 | 8.03e-8 |
| EXP-021 vs EXP-024 | 10 | 0.225 | 0.410 | +18.5 | 28.80 | 8.03e-8 |

해석 (발표): FT는 ZS 대비 Exact@1/@5 **유의하게 향상**. 동일 200에서 SillokBERT는 FT보다 Exact@1/@5 **유의하게 높음** → 「BERT 우위」 문구는 이 paired 결과로 가능. S5에 EXP-015 n≈500 Exact@5 0.504를 EXP-024 0.410과 직접 비교하지 말 것.

## 산출물

- `exp025_summary.json` — 전체 요약
- `paired_exp015_vs_exp024_k{1,5,10}.json`
- `paired_exp021_vs_exp024_k{1,5,10}.json`
- `exp015_sillok_same200_as_exp024.csv`, `exp021_g3_beam5_same200_as_exp024.csv`
- `exp024_qlora_beam5_n200_predictions.csv` (복사)

실행: `PYTHONPATH=. python reports/exp025/_run_exp025.py`
