# GPU 실행 상태 — EXP-012 (완료)

> **공식 ID:** EXP-007 = zero-shot SillokBERT(완료). EXP-011 = CPU subsample FT(완료).
> **EXP-012** = GPU full-data fine-tune + full test n=9,719 (**완료 · 2026-07-12 · DECISIONS [010]**).
> 파일명 legacy(`exp007_gpu_status`) 유지, 내용은 EXP-012.

- **Updated:** 2026-07-12
- **실행 환경:** Google Colab GPU (`device=cuda`). Cloud CPU는 준비/smoke만.
- **런북:** `reports/colab_exp012_runbook.md`, `notebooks/exp012_colab.ipynb`
- **결과 스냅샷:** `reports/exp012_results.md`
- **스크립트:** `work/run_exp012_gpu.py` (로컬/GPU 환경용). `work/run_exp007_gpu.py` retired.
- **Resume:** `--resume-from checkpoint-epoch-N` + `training_state.pt`

## Results (Colab 2026-07-12)

| 항목 | 값 |
|------|-----|
| Exact@1 / @5 / @10 | **33.75%** / **53.04%** / **60.47%** |
| test n | 9,719 |
| best_epoch | 1 |
| best_valid_loss | ≈3.6619 |
| 로컬 반출 | `notebooks/best/` (Git 금지) |

## Checkpoint selection

- Metric: validation MLM loss (minimize)
- Artifacts: `epoch_metrics.json`, `best_checkpoint.json`, `best/`
- Test Exact@k must not pick the checkpoint

## Notes

- 출력은 Drive 대신 `/content/exp012/`에 두고 종료 후 로컬 반출 (Drive 용량 이슈)
- smoke (`_exp012_smoke`) 메타는 공식 성능이 아님
