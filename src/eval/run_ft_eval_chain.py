"""EXP-018 / FT eval chain: Exact@5 select → subsample → McNemar → stratified → gate.

Chains (DECISIONS [012]):
  1. select_checkpoint_by_exact (per-epoch valid Exact@k → best_exact5/)
  2. valid subsample n=2000 seed=42 predictions (top-10 + is_exact_at_1/5/10)
  3. paired McNemar vs EXP-015 predictions (read-only baseline)
  4. stratified freq buckets (very_common/common/mid/rare/hapax/unseen Exact@5)
  5. Gate documentation / decision JSON:
       - subsample Exact@5 ≥ EXP-015 +0.7pp AND McNemar p<0.05 → allow full valid
       - full valid Exact@5 ≥ EXP-015 +1.0pp → adoption candidate

Never uses test split. Smoke metrics are not official.
Missing EXP-015 prediction CSV is handled gracefully (McNemar skipped with note).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.finetune_sillokbert import RunConfigError, assert_safe_output_dir, get_git_commit
from src.eval.paired_exact_compare import load_predictions, paired_exact_compare
from src.eval.run_fill_mask_eval import run_evaluation
from src.eval.select_checkpoint_by_exact import (
    EXP015_VALID_EXACT5,
    SelectConfigError,
    evaluate_checkpoints,
)
from src.eval.stratified_analysis import build_stratified_report, render_markdown

DEFAULT_MASK = Path("data/model_inputs/artificial_mask_dataset.csv")
DEFAULT_EXP015_PREDS = Path("outputs/predictions/exp015_sillok_ft_valid_predictions.csv")
DEFAULT_EXP015_SUMMARY = Path("reports/evaluation/exp015_sillokbert_valid_summary.csv")
SUBSAMPLE_N = 2000
SUBSAMPLE_SEED = 42
GATE_SUBSAMPLE_PP = 0.7
GATE_FULL_PP = 1.0
GATE_P_VALUE = 0.05


class ChainConfigError(ValueError):
    """Invalid eval-chain arguments."""


def evaluate_gate(
    *,
    candidate_exact5: float,
    baseline_exact5: float = EXP015_VALID_EXACT5,
    mcnemar_p: float | None,
    stage: str,
) -> dict[str, Any]:
    """Document go/no-go gates. Does not invent missing metrics."""
    delta_pp = (candidate_exact5 - baseline_exact5) * 100.0
    if stage == "subsample":
        delta_ok = delta_pp >= GATE_SUBSAMPLE_PP
        p_ok = mcnemar_p is not None and mcnemar_p < GATE_P_VALUE
        allow_full = bool(delta_ok and p_ok)
        return {
            "stage": "subsample",
            "baseline_exact5": baseline_exact5,
            "candidate_exact5": candidate_exact5,
            "delta_pp": delta_pp,
            "threshold_pp": GATE_SUBSAMPLE_PP,
            "delta_ok": delta_ok,
            "mcnemar_p": mcnemar_p,
            "p_threshold": GATE_P_VALUE,
            "p_ok": p_ok,
            "allow_full_valid": allow_full,
            "adoption_candidate": False,
            "rule": (
                "subsample Exact@5 ≥ EXP-015 +0.7pp AND McNemar p<0.05 → allow full valid"
            ),
        }
    if stage == "full_valid":
        delta_ok = delta_pp >= GATE_FULL_PP
        return {
            "stage": "full_valid",
            "baseline_exact5": baseline_exact5,
            "candidate_exact5": candidate_exact5,
            "delta_pp": delta_pp,
            "threshold_pp": GATE_FULL_PP,
            "delta_ok": delta_ok,
            "mcnemar_p": mcnemar_p,
            "p_threshold": GATE_P_VALUE,
            "adoption_candidate": bool(delta_ok),
            "allow_full_valid": True,
            "rule": "full valid Exact@5 ≥ EXP-015 +1.0pp → adoption candidate",
        }
    raise ChainConfigError(f"Unknown gate stage: {stage}")


def run_eval_chain(
    run_dir: Path,
    *,
    mask_csv: Path = DEFAULT_MASK,
    experiment_id: str = "EXP-018",
    exp015_predictions: Path = DEFAULT_EXP015_PREDS,
    selection_run_mode: str = "smoke",
    selection_max_samples: int | None = SUBSAMPLE_N,
    skip_selection: bool = False,
    model_path: Path | None = None,
    batch_size: int = 8,
    run_full_valid: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the post-train eval chain. Always split=valid."""
    run_dir = Path(run_dir)
    out = Path(output_dir) if output_dir is not None else (run_dir / "eval_chain")
    assert_safe_output_dir(out, planned_experiment_id=experiment_id)
    out.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "planned_experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "split": "valid",
        "official_performance": False,
        "gate_docs": {
            "subsample": (
                f"Exact@5 ≥ EXP-015 ({EXP015_VALID_EXACT5:.2%}) +{GATE_SUBSAMPLE_PP}pp "
                f"AND McNemar p<{GATE_P_VALUE} → allow full valid"
            ),
            "full_valid": (
                f"Exact@5 ≥ EXP-015 +{GATE_FULL_PP}pp → adoption candidate"
            ),
            "exp015_baseline_exact5": EXP015_VALID_EXACT5,
            "exp015_summary_committed": str(DEFAULT_EXP015_SUMMARY),
            "exp015_predictions_gitignore": str(DEFAULT_EXP015_PREDS),
            "missing_exp015_preds_workaround": (
                "If outputs/predictions/exp015_* is missing, copy from local backup "
                "or re-run EXP-015 valid predictions once; McNemar is skipped until then. "
                "Committed summary at reports/evaluation/exp015_sillokbert_valid_summary.csv "
                "still documents the 53.24% Exact@5 baseline."
            ),
        },
    }

    # --- 1. Exact@5 selection ---
    if skip_selection:
        ckpt = Path(model_path) if model_path is not None else (run_dir / "best_exact5")
        if not ckpt.exists():
            ckpt = run_dir / "best"
        result["selection"] = {
            "skipped": True,
            "model_path": str(ckpt),
        }
        selected_model = ckpt
    else:
        selection = evaluate_checkpoints(
            run_dir,
            mask_csv=mask_csv,
            run_mode=selection_run_mode,
            max_samples=selection_max_samples,
            seed=SUBSAMPLE_SEED,
            batch_size=batch_size,
            experiment_id=experiment_id,
        )
        result["selection"] = selection
        selected_model = Path(selection["best_exact5_alias_dir"])

    # --- 2. Valid subsample 2000 predictions ---
    subsample_pred = out / f"{experiment_id.lower()}_valid_subsample2000_predictions.csv"
    subsample_summary = out / f"{experiment_id.lower()}_valid_subsample2000_summary.csv"
    print(
        f"[eval_chain] subsample n={SUBSAMPLE_N} seed={SUBSAMPLE_SEED} "
        f"model={selected_model}",
        flush=True,
    )
    subsample_payload = run_evaluation(
        mask_csv,
        experiment_id=experiment_id,
        experiment_name=f"{experiment_id} valid subsample n=2000",
        model_kind="ddokbaro/SillokBert",
        split="valid",
        model_path=str(selected_model),
        model_version="best_exact5",
        max_samples=SUBSAMPLE_N,
        seed=SUBSAMPLE_SEED,
        batch_size=batch_size,
        predictions_csv=subsample_pred,
        summary_csv=subsample_summary,
        append_outputs=False,
        notes="eval_chain subsample; DECISIONS [012]; not official",
    )
    result["subsample"] = {
        "n": SUBSAMPLE_N,
        "seed": SUBSAMPLE_SEED,
        "predictions_csv": str(subsample_pred),
        "summary_csv": str(subsample_summary),
        "overall": subsample_payload["overall"],
    }
    cand_exact5 = float(subsample_payload["overall"]["exact_at_5"])

    # --- 3. Paired McNemar vs EXP-015 ---
    mcnemar_p: float | None = None
    if not exp015_predictions.is_file():
        result["paired"] = {
            "skipped": True,
            "reason": f"EXP-015 predictions missing: {exp015_predictions}",
            "workaround": result["gate_docs"]["missing_exp015_preds_workaround"],
        }
        print(
            f"[eval_chain] WARN: missing EXP-015 preds at {exp015_predictions}; "
            "skipping McNemar",
            flush=True,
        )
    else:
        paired = paired_exact_compare(
            load_predictions(exp015_predictions),
            load_predictions(subsample_pred),
            k=5,
            label_a="EXP015",
            label_b=experiment_id,
            seed=SUBSAMPLE_SEED,
        )
        paired_path = out / "paired_vs_exp015_subsample.json"
        paired_path.write_text(
            json.dumps(paired, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        mcnemar_p = float(paired["mcnemar"]["p_value"])
        result["paired"] = {"skipped": False, "path": str(paired_path), **paired}
        print(
            f"[eval_chain] McNemar p={mcnemar_p:.6g} "
            f"diff_pp={paired['diff_pp']:.3f}",
            flush=True,
        )

    subsample_gate = evaluate_gate(
        candidate_exact5=cand_exact5,
        mcnemar_p=mcnemar_p,
        stage="subsample",
    )
    result["gate_subsample"] = subsample_gate

    # --- 4. Stratified freq buckets ---
    strat_json = out / "stratified_subsample.json"
    strat_md = out / "stratified_subsample.md"
    strat_report = build_stratified_report(
        subsample_pred,
        mask_csv,
        model_key=experiment_id.lower(),
        split="valid",
    )
    strat_json.write_text(
        json.dumps(strat_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    strat_md.write_text(
        render_markdown(
            strat_report,
            title=f"{experiment_id} stratified (valid subsample)",
            models=[(experiment_id.lower(), experiment_id)],
            split="valid",
        ),
        encoding="utf-8",
    )
    result["stratified"] = {
        "json": str(strat_json),
        "md": str(strat_md),
        "by_freq_bucket": strat_report[experiment_id.lower()]["by_freq_bucket"],
    }

    # --- 5. Conditional full valid ---
    if run_full_valid:
        if not subsample_gate.get("allow_full_valid"):
            print(
                "[eval_chain] WARN: --run-full-valid set but subsample gate not passed; "
                "running anyway as explicitly requested.",
                flush=True,
            )
        full_pred = out / f"{experiment_id.lower()}_valid_full_predictions.csv"
        full_summary = out / f"{experiment_id.lower()}_valid_full_summary.csv"
        full_payload = run_evaluation(
            mask_csv,
            experiment_id=experiment_id,
            experiment_name=f"{experiment_id} full valid",
            model_kind="ddokbaro/SillokBert",
            split="valid",
            model_path=str(selected_model),
            model_version="best_exact5",
            max_samples=None,
            seed=SUBSAMPLE_SEED,
            batch_size=batch_size,
            predictions_csv=full_pred,
            summary_csv=full_summary,
            append_outputs=False,
            notes="eval_chain full valid; DECISIONS [012]",
        )
        full_exact5 = float(full_payload["overall"]["exact_at_5"])
        full_paired_p: float | None = None
        if exp015_predictions.is_file():
            full_paired = paired_exact_compare(
                load_predictions(exp015_predictions),
                load_predictions(full_pred),
                k=5,
                label_a="EXP015",
                label_b=experiment_id,
                seed=SUBSAMPLE_SEED,
            )
            full_paired_path = out / "paired_vs_exp015_full.json"
            full_paired_path.write_text(
                json.dumps(full_paired, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            full_paired_p = float(full_paired["mcnemar"]["p_value"])
            result["paired_full"] = {
                "skipped": False,
                "path": str(full_paired_path),
                **full_paired,
            }
        result["full_valid"] = {
            "predictions_csv": str(full_pred),
            "summary_csv": str(full_summary),
            "overall": full_payload["overall"],
        }
        result["gate_full_valid"] = evaluate_gate(
            candidate_exact5=full_exact5,
            mcnemar_p=full_paired_p,
            stage="full_valid",
        )
    else:
        result["full_valid"] = {
            "skipped": True,
            "reason": (
                "Not requested. Run with --run-full-valid only after subsample gate passes "
                f"(need Exact@5 ≥ EXP-015 +{GATE_SUBSAMPLE_PP}pp and McNemar p<{GATE_P_VALUE})."
            ),
            "allow_full_valid": subsample_gate.get("allow_full_valid"),
        }

    chain_path = out / "eval_chain_result.json"
    chain_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_json"] = str(chain_path)
    print(f"[eval_chain] wrote {chain_path}", flush=True)
    print(
        f"[eval_chain] subsample gate allow_full_valid="
        f"{subsample_gate.get('allow_full_valid')} "
        f"(delta_pp={subsample_gate.get('delta_pp'):.3f}, "
        f"p={subsample_gate.get('mcnemar_p')})",
        flush=True,
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mask-dataset", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--experiment-id", default="EXP-018")
    parser.add_argument(
        "--exp015-predictions",
        type=Path,
        default=DEFAULT_EXP015_PREDS,
        help="Read-only EXP-015 valid predictions (gitignore; may be missing).",
    )
    parser.add_argument(
        "--selection-run-mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Mode for per-epoch Exact@5 selection (default smoke / n=2000).",
    )
    parser.add_argument(
        "--selection-max-samples",
        type=int,
        default=SUBSAMPLE_N,
        help="Valid subsample for selection when selection-run-mode=smoke.",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip per-epoch selection; use --model-path or run-dir/best_exact5.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--run-full-valid",
        action="store_true",
        help="Also run full valid after subsample (intended after gate passes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Eval-chain artifacts dir (default: <run-dir>/eval_chain).",
    )
    parser.add_argument(
        "--split",
        default="valid",
        help="Must be valid. Test is refused.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if str(args.split).strip().lower() != "valid":
        parser.error(
            f"--split must be 'valid' (got {args.split!r}). Never use test for EXP-018."
        )
    try:
        result = run_eval_chain(
            args.run_dir,
            mask_csv=args.mask_dataset,
            experiment_id=args.experiment_id,
            exp015_predictions=args.exp015_predictions,
            selection_run_mode=args.selection_run_mode,
            selection_max_samples=args.selection_max_samples,
            skip_selection=args.skip_selection,
            model_path=args.model_path,
            batch_size=args.batch_size,
            run_full_valid=args.run_full_valid,
            output_dir=args.output_dir,
        )
    except (ChainConfigError, SelectConfigError, RunConfigError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
