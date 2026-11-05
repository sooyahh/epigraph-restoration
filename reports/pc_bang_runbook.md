# PC방 GPU 실행 Runbook — EXP-012 (legacy 경로)

> **2026-07-12 갱신:** EXP-012는 **Colab GPU에서 실행 완료**됨 (Exact@1 33.75% / @5 53.04% / @10 60.47%, DECISIONS [010]).
> **최신 절차·결과:** [`reports/colab_exp012_runbook.md`](colab_exp012_runbook.md) · [`reports/exp012_results.md`](exp012_results.md) · [`notebooks/exp012_colab.ipynb`](../notebooks/exp012_colab.ipynb)
> 아래는 동일 CLI를 PC/로컬 GPU에서 **재현**할 때의 참고본이다.

> **공식 실험 ID**
> - **EXP-007** = SillokBERT zero-shot (완료, 변경 없음)
> - **EXP-011** = SillokBERT CPU subsample fine-tuning baseline (완료, 변경 없음; test n=2,000)
> - **EXP-012** = SillokBERT GPU full-data fine-tuning + full test n=9,719 (**완료 · Colab**)
>
> 노트북 legacy 문서의 “GPU EXP-007”은 **EXP-012**에 해당한다.
> EXP-011의 Exact@5=50.45%(n=2,000)를 EXP-012 결과와 혼동하지 말 것.

**목적:** EXP-012 재현 — full train → validation best → full test 1회.
**반출:** `.../sillokbert_exp012_gpu/best/` + `best_checkpoint.json` (Git 금지); 공식 백업 `notebooks/best/`
**선택 기준:** `minimize_valid_loss`
**Resume:** `--resume-from .../checkpoint-epoch-N` (완료 epoch만)

단계별 명령은 Colab 런북과 동일하다.

```bash
# smoke
python -m src.data.finetune_sillokbert --run-mode smoke --planned-experiment-id EXP-012 \
  --output-dir outputs/finetune/_exp012_smoke --epochs 1 --batch-size 8

# full-train
python -m src.data.finetune_sillokbert --run-mode full --planned-experiment-id EXP-012 \
  --output-dir outputs/finetune/sillokbert_exp012_gpu --epochs 2 --batch-size 16 --learning-rate 5e-5

# resume example
python -m src.data.finetune_sillokbert --run-mode full --planned-experiment-id EXP-012 \
  --output-dir outputs/finetune/sillokbert_exp012_gpu --epochs 2 --batch-size 16 \
  --resume-from outputs/finetune/sillokbert_exp012_gpu/checkpoint-epoch-1

# full-test
python -m src.eval.run_fill_mask_eval --run-mode full --experiment-id EXP-012 \
  --experiment-name "SillokBERT GPU full-data fine-tuned full test" \
  --model-kind ddokbaro/SillokBert \
  --model-path outputs/finetune/sillokbert_exp012_gpu/best --require-model-path \
  --split test --batch-size 32 --overwrite \
  --predictions-csv outputs/predictions/exp012_sillok_ft_full_predictions.csv \
  --summary-csv outputs/evaluation/evaluation_summary.csv \
  --notes "planned EXP-012; model_path=best by valid_loss; full test n=9719"
```
