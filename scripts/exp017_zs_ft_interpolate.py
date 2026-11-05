"""EXP-017: interpolate ZS SillokBert and FT (EXP-012 best) fill-mask distributions.

p_final(c) = α * p_FT(c) + (1-α) * p_ZS(c)
Candidate set = union of each model's top-k single-char predictions.
Does not modify existing eval/train modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.mask_window import build_masked_window  # noqa: E402
from src.eval.metrics import compute_exact_metrics, rank_of_gold  # noqa: E402
from src.eval.paired_exact_compare import paired_exact_compare  # noqa: E402
from src.eval.run_fill_mask_eval import PREDICTION_COLUMNS, _select_rows  # noqa: E402
from src.eval.stratified_analysis import FREQ_BUCKETS, _freq_bucket  # noqa: E402

ZS_MODEL = "ddokbaro/SillokBert"
FT_MODEL = ROOT / "notebooks" / "best"
MASK_CSV = ROOT / "data" / "model_inputs" / "artificial_mask_dataset.csv"
EXP015_PRED = ROOT / "outputs" / "predictions" / "exp015_sillok_ft_valid_predictions.csv"
OUT_PRED = ROOT / "outputs" / "predictions"
OUT_REP = ROOT / "reports" / "exp017"

TOP_K = 50
CONTEXT_WINDOW = 64
BATCH_SIZE = 8
SEED = 42
ALPHAS = [round(i / 10, 1) for i in range(11)]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(path_or_name: str | Path, device: str):
    name = str(path_or_name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)
    model.to(device)
    model.eval()
    return tok, model


def _window_text(row: pd.Series, mask_token: str) -> str | None:
    built = build_masked_window(
        str(row["original_text"]),
        int(row["mask_position"]),
        str(row["label_char"]),
        mask_token,
        context_window=CONTEXT_WINDOW,
    )
    return None if built is None else built[0]


def _single_char_topk(
    probs: torch.Tensor,
    tokenizer,
    k: int,
) -> tuple[list[str], list[float]]:
    pool = min(max(k * 5, k), int(probs.shape[-1]))
    top_ids = torch.topk(probs, k=pool).indices.tolist()
    chars: list[str] = []
    scores: list[float] = []
    for token_id in top_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None or str(token).startswith("##"):
            continue
        ch = str(token).strip()
        if len(ch) == 1 and ch not in chars:
            chars.append(ch)
            scores.append(float(probs[token_id].item()))
        if len(chars) >= k:
            break
    return chars, scores


def _prob_for_char(probs: torch.Tensor, tokenizer, ch: str) -> float:
    tid = tokenizer.convert_tokens_to_ids(ch)
    if tid is None or tid == tokenizer.unk_token_id:
        return 0.0
    return float(probs[int(tid)].item())


@torch.no_grad()
def _batch_mask_probs(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> list[torch.Tensor | None]:
    mask_id = tokenizer.mask_token_id
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = model(**encoded).logits
    out: list[torch.Tensor | None] = []
    for i in range(logits.shape[0]):
        input_ids = encoded["input_ids"][i]
        positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            out.append(None)
            continue
        out.append(torch.softmax(logits[i, int(positions[0])], dim=-1).cpu())
    return out


def infer_distributions(
    eval_df: pd.DataFrame,
    *,
    zs_tok,
    zs_model,
    ft_tok,
    ft_model,
    device: str,
    top_k: int,
    batch_size: int,
    progress_label: str,
) -> list[dict]:
    rows = [row for _, row in eval_df.iterrows()]
    total = len(rows)
    records: list[dict] = []
    mask_token = zs_tok.mask_token or "[MASK]"

    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        if start == 0 or (start // batch_size) % 10 == 0:
            print(
                f"  [{progress_label}] {min(start + batch_size, total)}/{total}",
                flush=True,
            )
        texts: list[str] = []
        keep_idx: list[int] = []
        for i, row in enumerate(batch):
            text = _window_text(row, mask_token)
            if text is None:
                continue
            texts.append(text)
            keep_idx.append(i)

        zs_probs_list: list[torch.Tensor | None] = [None] * len(batch)
        ft_probs_list: list[torch.Tensor | None] = [None] * len(batch)
        if texts:
            zs_batch = _batch_mask_probs(zs_model, zs_tok, texts, device)
            ft_batch = _batch_mask_probs(ft_model, ft_tok, texts, device)
            for j, bi in enumerate(keep_idx):
                zs_probs_list[bi] = zs_batch[j]
                ft_probs_list[bi] = ft_batch[j]

        for i, row in enumerate(batch):
            gold = str(row["label_char"])
            zs_p = zs_probs_list[i]
            ft_p = ft_probs_list[i]
            if zs_p is None or ft_p is None:
                records.append(
                    {
                        "sample_id": row["sample_id"],
                        "label_char": gold,
                        "top_k": top_k,
                        "candidates": [],
                        "zs_top": [],
                        "ft_top": [],
                    }
                )
                continue
            zs_chars, _ = _single_char_topk(zs_p, zs_tok, top_k)
            ft_chars, _ = _single_char_topk(ft_p, ft_tok, top_k)
            union = list(dict.fromkeys(zs_chars + ft_chars))
            cands = []
            for ch in union:
                # Prefer each model's tokenizer for id lookup; vocab should match.
                cands.append(
                    {
                        "char": ch,
                        "p_zs": _prob_for_char(zs_p, zs_tok, ch),
                        "p_ft": _prob_for_char(ft_p, ft_tok, ch),
                    }
                )
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "label_char": gold,
                    "top_k": top_k,
                    "candidates": cands,
                    "zs_top": zs_chars,
                    "ft_top": ft_chars,
                }
            )
    return records


def _rank_with_alpha(cands: list[dict], alpha: float) -> list[str]:
    scored = []
    for item in cands:
        p = alpha * float(item["p_ft"]) + (1.0 - alpha) * float(item["p_zs"])
        scored.append((p, str(item["char"])))
    scored.sort(key=lambda x: (-x[0], x[1]))
    # unique chars preserving score order
    out: list[str] = []
    seen = set()
    for _p, ch in scored:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def records_to_predictions(
    records: list[dict],
    *,
    alpha: float,
    experiment_id: str,
    model_version: str,
    notes: str,
) -> pd.DataFrame:
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for rec in records:
        tops = _rank_with_alpha(rec["candidates"], alpha)[:10]
        gold = str(rec["label_char"])
        rank = rank_of_gold(tops, gold)
        top1 = tops[0] if tops else ""
        top5 = tops[:5]
        top10 = tops[:10]
        rows.append(
            {
                "prediction_id": f"pred_{experiment_id.lower()}_{rec['sample_id']}",
                "experiment_id": experiment_id,
                "model_name": f"interp_zs_ft_alpha_{alpha:.1f}",
                "model_version": model_version,
                "sample_id": rec["sample_id"],
                "case_id": "",
                "input_text": "",
                "label_char": gold,
                "top1_char": top1,
                "top1_score": "",
                "top5_chars": ";".join(top5),
                "top5_scores": "",
                "top10_chars": ";".join(top10),
                "top10_scores": "",
                "rank_of_gold": rank if rank is not None else "",
                "is_exact_at_1": bool(rank is not None and rank <= 1),
                "is_exact_at_5": bool(rank is not None and rank <= 5),
                "is_exact_at_10": bool(rank is not None and rank <= 10),
                "prediction_type": "artificial_mask",
                "created_at": created_at,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def metrics_from_pred(pred_df: pd.DataFrame) -> dict:
    n = len(pred_df)
    if n == 0:
        return {"n": 0, "exact_at_1": 0.0, "exact_at_5": 0.0, "exact_at_10": 0.0}
    return {
        "n": n,
        "exact_at_1": float(pred_df["is_exact_at_1"].astype(bool).mean()),
        "exact_at_5": float(pred_df["is_exact_at_5"].astype(bool).mean()),
        "exact_at_10": float(pred_df["is_exact_at_10"].astype(bool).mean()),
    }


def attach_freq_buckets(pred_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    train_counts = Counter(
        mask_df.loc[mask_df["split"] == "train", "label_char"].astype(str)
    )
    out = pred_df.copy()
    out["train_label_count"] = out["label_char"].astype(str).map(
        lambda ch: int(train_counts.get(ch, 0))
    )
    out["freq_bucket"] = out["train_label_count"].map(_freq_bucket)
    return out


def stratified_exact5(pred_df: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for name, _pred in FREQ_BUCKETS:
        sub = pred_df[pred_df["freq_bucket"] == name]
        m = metrics_from_pred(sub)
        result[name] = m
    return result


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cmd_infer(args: argparse.Namespace) -> None:
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"device={device}", flush=True)
    df = pd.read_csv(MASK_CSV)
    if args.full:
        eval_df = df[df["split"] == "valid"].copy().reset_index(drop=True)
        tag = "valid_full"
    else:
        eval_df = _select_rows(df, "valid", args.max_samples, SEED)
        tag = f"valid_sub{args.max_samples}"
    print(f"infer n={len(eval_df)} tag={tag} top_k={args.top_k}", flush=True)

    print(f"loading ZS: {ZS_MODEL}", flush=True)
    zs_tok, zs_model = _load_model(ZS_MODEL, device)
    print(f"loading FT: {FT_MODEL}", flush=True)
    ft_tok, ft_model = _load_model(FT_MODEL, device)

    records = infer_distributions(
        eval_df,
        zs_tok=zs_tok,
        zs_model=zs_model,
        ft_tok=ft_tok,
        ft_model=ft_model,
        device=device,
        top_k=args.top_k,
        batch_size=args.batch_size,
        progress_label=tag,
    )
    out_path = OUT_PRED / f"exp017_dist_{tag}_topk{args.top_k}.jsonl"
    save_jsonl(out_path, records)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "n": len(records),
        "top_k": args.top_k,
        "context_window": CONTEXT_WINDOW,
        "zs_model": ZS_MODEL,
        "ft_model": str(FT_MODEL),
        "device": device,
        "seed": SEED if not args.full else None,
        "path": str(out_path),
    }
    (OUT_REP / f"infer_meta_{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}", flush=True)


def cmd_sweep(args: argparse.Namespace) -> None:
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.dist_jsonl))
    mask_df = pd.read_csv(MASK_CSV)
    print(f"sweep n={len(records)} alphas={ALPHAS}", flush=True)

    # α=0 / α=1 pure tops for sanity vs stored zs/ft top lists
    sweep_rows = []
    pred_by_alpha: dict[float, pd.DataFrame] = {}
    for alpha in ALPHAS:
        pred = records_to_predictions(
            records,
            alpha=alpha,
            experiment_id=f"EXP017_a{str(alpha).replace('.', '')}",
            model_version=f"exp017_interp_alpha_{alpha:.1f}",
            notes=f"EXP-017 interpolate alpha={alpha:.1f}; subsample from {args.dist_jsonl}",
        )
        pred_path = OUT_PRED / f"exp017_alpha{alpha:.1f}_valid_sub_predictions.csv"
        pred.to_csv(pred_path, index=False)
        pred_f = attach_freq_buckets(pred, mask_df)
        overall = metrics_from_pred(pred_f)
        by_freq = stratified_exact5(pred_f)
        row = {
            "alpha": alpha,
            "n": overall["n"],
            "exact_at_1": overall["exact_at_1"],
            "exact_at_5": overall["exact_at_5"],
            "exact_at_10": overall["exact_at_10"],
            "pred_path": str(pred_path),
        }
        for name, m in by_freq.items():
            key = name.split()[0]  # very_common / common / ...
            row[f"exact_at_5__{key}"] = m["exact_at_5"]
            row[f"n__{key}"] = m["n"]
        sweep_rows.append(row)
        pred_by_alpha[alpha] = pred_f
        print(
            f"  α={alpha:.1f} Exact@5={overall['exact_at_5']:.4f} "
            f"hapax@5={by_freq['hapax (count=1)']['exact_at_5']:.4f}",
            flush=True,
        )

    # Sanity: α=1 vs EXP-015 on overlapping sample_ids
    sanity = {}
    if EXP015_PRED.exists():
        exp015 = pd.read_csv(EXP015_PRED)
        a1 = pred_by_alpha[1.0]
        ids = set(a1["sample_id"])
        base = exp015[exp015["sample_id"].isin(ids)].copy()
        merged = a1.merge(
            base[["sample_id", "is_exact_at_1", "is_exact_at_5", "is_exact_at_10"]],
            on="sample_id",
            suffixes=("_a1", "_exp015"),
        )
        sanity = {
            "n_overlap": len(merged),
            "exact_at_1_alpha1": float(merged["is_exact_at_1_a1"].astype(bool).mean()),
            "exact_at_5_alpha1": float(merged["is_exact_at_5_a1"].astype(bool).mean()),
            "exact_at_10_alpha1": float(merged["is_exact_at_10_a1"].astype(bool).mean()),
            "exact_at_1_exp015": float(merged["is_exact_at_1_exp015"].astype(bool).mean()),
            "exact_at_5_exp015": float(merged["is_exact_at_5_exp015"].astype(bool).mean()),
            "exact_at_10_exp015": float(merged["is_exact_at_10_exp015"].astype(bool).mean()),
            "exact_at_5_abs_diff": abs(
                float(merged["is_exact_at_5_a1"].astype(bool).mean())
                - float(merged["is_exact_at_5_exp015"].astype(bool).mean())
            ),
        }
        # also compare to FT-only ranking from same dist (should match α=1 exactly)
        ft_only_hits = []
        for rec in records:
            tops = rec.get("ft_top", [])[:10]
            gold = str(rec["label_char"])
            r = rank_of_gold(tops, gold)
            ft_only_hits.append(bool(r is not None and r <= 5))
        sanity["exact_at_5_ft_top_from_dist"] = (
            sum(ft_only_hits) / len(ft_only_hits) if ft_only_hits else 0.0
        )
        sanity["alpha1_matches_ft_top_exact5"] = (
            abs(sanity["exact_at_5_alpha1"] - sanity["exact_at_5_ft_top_from_dist"]) < 1e-12
        )

    # Select alpha* candidates
    by_e5 = sorted(sweep_rows, key=lambda r: (-r["exact_at_5"], -r["alpha"]))
    alpha_overall = by_e5[0]["alpha"]
    # hapax recovery: maximize hapax Exact@5 among alphas that don't collapse overall too much
    hapax_sorted = sorted(
        sweep_rows,
        key=lambda r: (-r.get("exact_at_5__hapax", 0.0), -r["exact_at_5"], r["alpha"]),
    )
    alpha_hapax = hapax_sorted[0]["alpha"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dist_jsonl": str(args.dist_jsonl),
        "alphas": sweep_rows,
        "sanity_alpha1_vs_exp015": sanity,
        "candidates": {
            "alpha_overall_max_exact5": alpha_overall,
            "alpha_hapax_max_exact5": alpha_hapax,
        },
        "freq_bucket_names": [name for name, _ in FREQ_BUCKETS],
    }
    out_json = OUT_REP / "exp017_alpha_sweep.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(sweep_rows).to_csv(OUT_REP / "exp017_alpha_sweep.csv", index=False)
    print(f"wrote {out_json}", flush=True)
    print(
        f"candidates: overall_α*={alpha_overall} hapax_α*={alpha_hapax}",
        flush=True,
    )


def cmd_paired(args: argparse.Namespace) -> None:
    OUT_REP.mkdir(parents=True, exist_ok=True)
    mask_df = pd.read_csv(MASK_CSV)
    base = pd.read_csv(args.baseline_csv)
    cand = pd.read_csv(args.candidate_csv)
    base_f = attach_freq_buckets(base, mask_df)
    cand_f = attach_freq_buckets(cand, mask_df)

    # align to candidate sample_ids
    ids = set(cand_f["sample_id"])
    base_f = base_f[base_f["sample_id"].isin(ids)].copy()

    results = {}
    for scope, b_df, c_df in [
        ("overall", base_f, cand_f),
        (
            "hapax",
            base_f[base_f["freq_bucket"] == "hapax (count=1)"],
            cand_f[cand_f["freq_bucket"] == "hapax (count=1)"],
        ),
        (
            "rare",
            base_f[base_f["freq_bucket"] == "rare (2-5)"],
            cand_f[cand_f["freq_bucket"] == "rare (2-5)"],
        ),
    ]:
        # keep overlapping ids within scope
        scope_ids = set(b_df["sample_id"]) & set(c_df["sample_id"])
        b_s = b_df[b_df["sample_id"].isin(scope_ids)]
        c_s = c_df[c_df["sample_id"].isin(scope_ids)]
        if len(scope_ids) == 0:
            results[scope] = {"error": "no overlapping samples", "n": 0}
            continue
        scope_res = {}
        for k in (1, 5, 10):
            scope_res[f"k{k}"] = paired_exact_compare(
                b_s,
                c_s,
                k=k,
                label_a=args.label_a,
                label_b=args.label_b,
                n_boot=2000,
                seed=SEED,
            )
        results[scope] = scope_res
        print(
            f"paired {scope}: n={scope_res['k5']['n_paired']} "
            f"Δ@5pp={scope_res['k5']['diff_pp']:.3f} "
            f"p={scope_res['k5']['mcnemar']['p_value']:.6g}",
            flush=True,
        )

    out = OUT_REP / args.output_name
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


def cmd_milestone(args: argparse.Namespace) -> None:
    """Build predictions + metrics for one alpha on a dist jsonl (usually full valid)."""
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.dist_jsonl))
    mask_df = pd.read_csv(MASK_CSV)
    alpha = float(args.alpha)
    pred = records_to_predictions(
        records,
        alpha=alpha,
        experiment_id="EXP017",
        model_version=f"exp017_interp_alpha_{alpha:.1f}_fullvalid",
        notes=f"EXP-017 milestone full valid alpha={alpha:.1f}",
    )
    pred_path = OUT_PRED / f"exp017_alpha{alpha:.1f}_valid_full_predictions.csv"
    pred.to_csv(pred_path, index=False)
    pred_f = attach_freq_buckets(pred, mask_df)
    overall = metrics_from_pred(pred_f)
    by_freq = {}
    for name, _ in FREQ_BUCKETS:
        sub = pred_f[pred_f["freq_bucket"] == name]
        by_freq[name] = metrics_from_pred(sub)
    # also Exact@1/10 per bucket
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "dist_jsonl": str(args.dist_jsonl),
        "pred_path": str(pred_path),
        "overall": overall,
        "by_freq": by_freq,
    }
    out = OUT_REP / f"exp017_milestone_alpha{alpha:.1f}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    infer = sub.add_parser("infer", help="Run ZS+FT and dump union candidate probs")
    infer.add_argument("--full", action="store_true", help="Full valid (no subsample)")
    infer.add_argument("--max-samples", type=int, default=2000)
    infer.add_argument("--top-k", type=int, default=TOP_K)
    infer.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    infer.set_defaults(func=cmd_infer)

    sweep = sub.add_parser("sweep", help="Alpha sweep on a dist jsonl")
    sweep.add_argument("--dist-jsonl", type=Path, required=True)
    sweep.set_defaults(func=cmd_sweep)

    paired = sub.add_parser("paired", help="Paired compare candidate vs baseline CSV")
    paired.add_argument("--baseline-csv", type=Path, required=True)
    paired.add_argument("--candidate-csv", type=Path, required=True)
    paired.add_argument("--label-a", default="FT_alpha1")
    paired.add_argument("--label-b", default="interp")
    paired.add_argument("--output-name", default="paired.json")
    paired.set_defaults(func=cmd_paired)

    mile = sub.add_parser("milestone", help="Metrics for one alpha on dist jsonl")
    mile.add_argument("--dist-jsonl", type=Path, required=True)
    mile.add_argument("--alpha", type=float, required=True)
    mile.set_defaults(func=cmd_milestone)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
