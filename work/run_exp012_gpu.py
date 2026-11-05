"""Planned EXP-012 GPU pipeline: full-data fine-tune + validation checkpoint selection.

Does NOT run full test automatically. After training, read best_checkpoint.json and
run eval with an explicit --model-path (see reports/local_gpu_runbook.md).

Official IDs:
  EXP-007 = SillokBERT zero-shot (unchanged)
  EXP-011 = CPU subsample fine-tune baseline (unchanged)
  EXP-012 = planned GPU full-data fine-tune (this script; not completed until logged)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = "outputs/finetune/sillokbert_exp012_gpu"
BEST_JSON = ROOT / FT_DIR / "best_checkpoint.json"
BEST_DIR = ROOT / FT_DIR / "best"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. Run this script at PC bang with GPU PyTorch.\n"
            "Check: python work/check_cuda_env.py"
        )

    py = sys.executable
    print(
        f"Planned EXP-012 GPU train pipeline started "
        f"{datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "This run trains only. Full test must be a separate command with "
        "--model-path pointing at best/ (validation-selected).",
        flush=True,
    )

    run(
        [
            py,
            "-m",
            "src.data.finetune_sillokbert",
            "--run-mode",
            "full",
            "--output-dir",
            FT_DIR,
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--learning-rate",
            "5e-5",
            "--planned-experiment-id",
            "EXP-012",
        ]
    )

    if not BEST_JSON.exists() or not BEST_DIR.exists():
        raise SystemExit(
            f"Missing best checkpoint artifacts under {FT_DIR}. "
            "Expected best_checkpoint.json and best/."
        )

    selection = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    print("\n=== Planned EXP-012 train complete (NOT a finished experiment log entry) ===")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print(
        "\nNext (manual full test — do not use test metrics to re-pick checkpoints):\n"
        f'  python -m src.eval.run_fill_mask_eval --run-mode full \\\n'
        f'    --experiment-id EXP-012 \\\n'
        f'    --experiment-name "SillokBERT GPU full-data fine-tuned full test" \\\n'
        f'    --model-kind ddokbaro/SillokBert \\\n'
        f'    --model-path {FT_DIR}/best \\\n'
        f'    --require-model-path \\\n'
        f'    --split test --batch-size 32 --overwrite \\\n'
        f'    --predictions-csv outputs/predictions/exp012_sillok_ft_full_predictions.csv \\\n'
        f'    --summary-csv outputs/evaluation/evaluation_summary.csv \\\n'
        f'    --notes "planned EXP-012; model_path=best by valid_loss; full test n=9719"\n'
    )
    print(
        "If training was interrupted after an epoch, resume with:\n"
        f"  --resume-from {FT_DIR}/checkpoint-epoch-N\n"
        "See reports/colab_exp012_runbook.md for Colab GPU steps.",
        flush=True,
    )
    print(f"Copy out: {FT_DIR}/best/ and {FT_DIR}/best_checkpoint.json", flush=True)


if __name__ == "__main__":
    main()
