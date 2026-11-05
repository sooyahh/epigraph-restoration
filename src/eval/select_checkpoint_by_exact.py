"""Post-hoc checkpoint selection by validation Exact@5 (DECISIONS [012]).

For each checkpoint-epoch-N under a run directory, evaluate valid Exact@1/5/10,
then select best by:
  1. max Exact@5
  2. min valid_loss (tie-break)
  3. lower epoch (tie-break)

Writes best_exact5/ (separate from in-training loss-based best/) and an epoch
summary JSON. Forces --split valid; never uses test.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.finetune_sillokbert import (
    RunConfigError,
    assert_safe_output_dir,
    get_git_commit,
)
from src.eval.run_fill_mask_eval import EvalConfigError, resolve_eval_run_config, run_evaluation

CHECKPOINT_RE = re.compile(r"^checkpoint-epoch-(\d+)$")
DEFAULT_MASK = Path("data/model_inputs/artificial_mask_dataset.csv")
# EXP-015 full-valid Exact@5 baseline (reference only; not invented here).
EXP015_VALID_EXACT5 = 0.5324


class SelectConfigError(ValueError):
    """Invalid selection arguments or missing checkpoints."""


def list_epoch_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (epoch, path) for checkpoint-epoch-N directories."""
    found: list[tuple[int, Path]] = []
    if not run_dir.is_dir():
        raise SelectConfigError(f"run-dir does not exist: {run_dir}")
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_RE.match(child.name)
        if match:
            found.append((int(match.group(1)), child))
    found.sort(key=lambda item: item[0])
    if not found:
        raise SelectConfigError(f"No checkpoint-epoch-* under {run_dir}")
    return found


def load_valid_loss_map(run_dir: Path) -> dict[int, float | None]:
    """Load epoch → valid_loss from epoch_metrics.json when present."""
    path = run_dir / "epoch_metrics.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}
    out: dict[int, float | None] = {}
    for row in raw:
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        loss = row.get("valid_loss")
        out[epoch] = None if loss is None else float(loss)
    return out


def select_best_by_exact5(epoch_records: list[dict]) -> dict:
    """Pick best by Exact@5 max; ties → min valid_loss → lower epoch."""
    if not epoch_records:
        raise SelectConfigError("No epoch records for Exact@5 selection.")
    for row in epoch_records:
        if row.get("exact_at_5") is None:
            raise SelectConfigError(
                f"Epoch {row.get('epoch')} missing exact_at_5; cannot select."
            )

    def sort_key(row: dict) -> tuple:
        loss = row.get("valid_loss")
        loss_key = float("inf") if loss is None else float(loss)
        return (-float(row["exact_at_5"]), loss_key, int(row["epoch"]))

    return sorted(epoch_records, key=sort_key)[0]


def _alias_checkpoint(src: Path, dest: Path) -> None:
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.copytree(src, dest)


def evaluate_checkpoints(
    run_dir: Path,
    *,
    mask_csv: Path,
    run_mode: str,
    max_samples: int | None,
    seed: int,
    batch_size: int,
    experiment_id: str,
    model_kind: str = "ddokbaro/SillokBert",
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate each epoch checkpoint on valid; select best_exact5."""
    # Force valid only — never test.
    split = "valid"
    eval_cfg = resolve_eval_run_config(run_mode, max_samples, split=split)
    checkpoints = list_epoch_checkpoints(run_dir)
    loss_map = load_valid_loss_map(run_dir)
    pred_root = predictions_dir or (run_dir / "exact5_selection_preds")
    pred_root.mkdir(parents=True, exist_ok=True)

    epoch_records: list[dict] = []
    for epoch, ckpt in checkpoints:
        pred_csv = pred_root / f"epoch{epoch}_valid_predictions.csv"
        summary_csv = pred_root / f"epoch{epoch}_valid_summary.csv"
        print(
            f"[select_exact5] evaluating epoch {epoch} @ {ckpt} "
            f"(split=valid, run_mode={eval_cfg['run_mode']}, "
            f"max_samples={eval_cfg['max_samples']})",
            flush=True,
        )
        payload = run_evaluation(
            mask_csv,
            experiment_id=f"{experiment_id}-E{epoch}",
            experiment_name=f"{experiment_id} epoch {epoch} Exact@5 select",
            model_kind=model_kind,
            split=split,
            model_path=str(ckpt),
            model_version=f"checkpoint-epoch-{epoch}",
            max_samples=eval_cfg["max_samples"],
            seed=seed,
            batch_size=batch_size,
            predictions_csv=pred_csv,
            summary_csv=summary_csv,
            append_outputs=False,
            notes=(
                f"post-hoc Exact@5 selection; split=valid; "
                f"run_mode={eval_cfg['run_mode']}; seed={seed}"
            ),
        )
        overall = payload["overall"]
        record = {
            "epoch": epoch,
            "checkpoint_dir": str(ckpt),
            "valid_loss": loss_map.get(epoch),
            "num_samples": overall["num_samples"],
            "exact_at_1": overall["exact_at_1"],
            "exact_at_5": overall["exact_at_5"],
            "exact_at_10": overall["exact_at_10"],
            "predictions_csv": str(pred_csv),
            "summary_csv": str(summary_csv),
            "used_for_test_selection": False,
            "split": split,
        }
        epoch_records.append(record)
        print(
            f"[select_exact5] epoch {epoch}: "
            f"Exact@1={overall['exact_at_1']:.4f} "
            f"@5={overall['exact_at_5']:.4f} "
            f"@10={overall['exact_at_10']:.4f} "
            f"valid_loss={loss_map.get(epoch)}",
            flush=True,
        )

    best = select_best_by_exact5(epoch_records)
    best_exact5_dir = run_dir / "best_exact5"
    _alias_checkpoint(Path(best["checkpoint_dir"]), best_exact5_dir)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "planned_experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "split": split,
        "run_mode": eval_cfg["run_mode"],
        "max_samples": eval_cfg["max_samples"],
        "seed": seed,
        "selection_metric": "exact_at_5",
        "selection_rule": "maximize_exact_at_5__tie_min_valid_loss__tie_lower_epoch",
        "best_epoch": best["epoch"],
        "best_exact_at_5": best["exact_at_5"],
        "best_exact_at_1": best["exact_at_1"],
        "best_exact_at_10": best["exact_at_10"],
        "best_valid_loss": best.get("valid_loss"),
        "best_checkpoint_dir": best["checkpoint_dir"],
        "best_exact5_alias_dir": str(best_exact5_dir),
        "exp015_valid_exact5_baseline_ref": EXP015_VALID_EXACT5,
        "note": (
            "best_exact5/ is post-hoc Exact@5 selection (DECISIONS [012]). "
            "Distinct from in-training best/ (minimize_valid_loss). "
            "Never use test split for selection. Smoke/subsample metrics are not official."
        ),
        "epoch_records": epoch_records,
        "official_performance": False,
        "smoke_not_official": eval_cfg["run_mode"] == "smoke",
    }
    summary_path = run_dir / "exact5_selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[select_exact5] BEST epoch={best['epoch']} Exact@5={best['exact_at_5']:.4f} "
        f"-> {best_exact5_dir}",
        flush=True,
    )
    print(f"[select_exact5] wrote {summary_path}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Fine-tune run directory containing checkpoint-epoch-*",
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        required=True,
        help="smoke requires --max-samples > 0; full evaluates entire valid split.",
    )
    parser.add_argument("--mask-dataset", type=Path, default=DEFAULT_MASK)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Valid subsample size (smoke). Use 2000 with --seed 42 for iteration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--experiment-id",
        default="EXP-018",
        help="Label written into prediction/summary rows.",
    )
    parser.add_argument(
        "--model-kind",
        default="ddokbaro/SillokBert",
        choices=["ddokbaro/SillokBert", "bert-base-chinese"],
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Where to write per-epoch prediction CSVs (default: <run-dir>/exact5_selection_preds).",
    )
    parser.add_argument(
        "--split",
        default="valid",
        help="Must be 'valid'. Test is refused.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if str(args.split).strip().lower() != "valid":
        parser.error(
            f"--split must be 'valid' for Exact@5 selection (got {args.split!r}). "
            "Never use test for EXP-018 selection/eval."
        )

    try:
        resolve_eval_run_config(args.run_mode, args.max_samples, split="valid")
        # Refuse selecting into protected EXP-012 / notebooks/best trees.
        assert_safe_output_dir(
            args.run_dir / "best_exact5",
            planned_experiment_id=args.experiment_id,
        )
        summary = evaluate_checkpoints(
            args.run_dir,
            mask_csv=args.mask_dataset,
            run_mode=args.run_mode,
            max_samples=args.max_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            experiment_id=args.experiment_id,
            model_kind=args.model_kind,
            predictions_dir=args.predictions_dir,
        )
    except (SelectConfigError, EvalConfigError, RunConfigError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
