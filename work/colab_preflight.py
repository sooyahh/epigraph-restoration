"""Colab / GPU preflight checks for planned EXP-012 (no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=ROOT / "data/model_inputs/artificial_mask_dataset.csv",
    )
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []

    required = [
        ROOT / "src/data/finetune_sillokbert.py",
        ROOT / "src/eval/run_fill_mask_eval.py",
        ROOT / "reports/colab_exp012_runbook.md",
        ROOT / "notebooks/exp012_colab.ipynb",
        args.mask_dataset,
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing: {path}")

    if args.mask_dataset.exists():
        mask = pd.read_csv(args.mask_dataset)
        counts = mask["split"].value_counts().to_dict()
        if len(mask) != 66537:
            errors.append(f"mask rows={len(mask)} expected 66537")
        if counts.get("test") != 9719:
            errors.append(f"test rows={counts.get('test')} expected 9719")
        g = mask.groupby("split")["inscription_id"].apply(set)
        for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
            overlap = len(g[a] & g[b])
            if overlap:
                errors.append(f"inscription overlap {a}&{b}={overlap}")

    try:
        import torch
        from src.data.finetune_sillokbert import build_arg_parser, resolve_run_config
        from src.eval.run_fill_mask_eval import resolve_eval_run_config

        help_text = build_arg_parser().format_help()
        if "--run-mode" not in help_text or "--resume-from" not in help_text:
            errors.append("finetune CLI missing --run-mode or --resume-from")
        resolve_run_config("full", None, None)
        resolve_eval_run_config("full", None, require_model_path=True, model_path="x")
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import/config failed: {exc}")
        cuda = False

    if args.require_cuda and not cuda:
        errors.append("CUDA required but torch.cuda.is_available() is False")

    payload = {
        "ok": not errors,
        "cuda_available": cuda,
        "errors": errors,
        "hint": "For official EXP-012 use Colab GPU; see reports/colab_exp012_runbook.md",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
