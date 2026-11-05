# EXP-015 Results — EXP-012 best validation Exact@k baseline

> **상태:** 완료 · DECISIONS [012] · 기록 `docs/EXPERIMENT_LOG.md` EXP-015  
> 학습 없음. `notebooks/best/`(EXP-012 loss-best)에 대한 **full valid** fill-mask Exact@k.

## Overall (valid n=10,679)

| 지표 | 값 |
|---|---:|
| Exact@1 | **33.78%** |
| Exact@5 | **53.24%** |
| Exact@10 | **60.90%** |
| device | cpu |
| model_version | `exp012_sillokbert_gpu_best_valid_loss` |

참고: 동일 체크포인트의 EXP-012 **test** n=9,719는 Exact@1 33.75% / @5 53.04% / @10 60.47%.  
공식 발표·모델 간 비교의 test 수치는 계속 EXP-012. **iteration 기준선은 본 EXP-015 valid**.

## 산출물

| 경로 | 비고 |
|---|---|
| `outputs/predictions/exp015_sillok_ft_valid_predictions.csv` | 10,679행 (gitignore) |
| `reports/evaluation/exp015_sillokbert_valid_summary.csv` | 커밋 대상 |
| `reports/exp015_results.json` | 본 스냅샷 수치 |

## 사용법 (이후 실험)

```bash
# iteration: valid subsample vs EXP-015 predictions (paired)
python -m src.eval.run_fill_mask_eval --run-mode smoke --split valid --max-samples 2000 --seed 42 ...
python -m src.eval.paired_exact_compare \
  --a outputs/predictions/exp015_sillok_ft_valid_predictions.csv \
  --b outputs/predictions/<candidate>_valid_predictions.csv \
  --k 5 --label-a EXP015 --label-b CAND
```

후보 CSV가 subsample이면 EXP-015 쪽도 동일 `sample_id`만 조인된다.
