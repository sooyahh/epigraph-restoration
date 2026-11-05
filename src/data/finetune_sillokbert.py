"""Fine-tune SillokBERT on artificial mask MLM samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "ddokbaro/SillokBert"
MAX_SEQ_LEN = 512
CONTEXT_WINDOW = 64
DEFAULT_SMOKE_MAX_TRAIN = 32
DEFAULT_SMOKE_MAX_EVAL = 16
DEFAULT_FULL_OUTPUT_DIR = Path("outputs/finetune/exp018/run")
DEFAULT_SMOKE_OUTPUT_DIR = Path("outputs/finetune/exp018/_smoke")
DEFAULT_MASK_DATASET = Path("data/model_inputs/artificial_mask_dataset.csv")
# In-training alias best/ uses validation MLM loss (lower is better).
# Official Exact@5 adoption is post-hoc (best_exact5/; DECISIONS [012]).
SELECTION_METRIC = "valid_loss"
SELECTION_RULE = "minimize_valid_loss"

# Paths that must never be overwritten by a new fine-tune run.
_FORBIDDEN_OUTPUT_PATH_PARTS = (
    Path("notebooks/best"),
    Path("outputs/finetune/sillokbert_exp012_gpu"),
    Path("outputs/finetune/sillokbert_gsm_baseline"),
    Path("reports/exp012_artifacts"),
)


class RunConfigError(ValueError):
    """Invalid --run-mode / subsample / output-dir argument combination."""


def get_git_commit(repo_root: Path | None = None) -> str:
    """Return current HEAD SHA, or 'unknown' if git is unavailable."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path, *, max_bytes: int | None = 8_388_608) -> str | None:
    """Hash file contents (full file, or first max_bytes if set for large CSVs)."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if max_bytes is None:
            while True:
                chunk = handle.read(1_048_576)
                if not chunk:
                    break
                digest.update(chunk)
        else:
            digest.update(handle.read(max_bytes))
            digest.update(f"|size={path.stat().st_size}".encode())
    return digest.hexdigest()


def assert_safe_output_dir(output_dir: Path, *, planned_experiment_id: str = "") -> Path:
    """Refuse writes under notebooks/best, EXP-012/011 artifact paths, etc."""
    resolved = output_dir.expanduser().resolve()
    cwd = Path.cwd().resolve()
    candidates = [resolved]
    try:
        candidates.append(resolved.relative_to(cwd))
    except ValueError:
        pass

    for forbidden in _FORBIDDEN_OUTPUT_PATH_PARTS:
        forbidden_resolved = (cwd / forbidden).resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            raise RunConfigError(
                f"Refusing output-dir {output_dir} — overlaps protected path "
                f"{forbidden}. Prefer outputs/finetune/exp018/... for EXP-018. "
                "Never write to notebooks/best or EXP-012/011 artifact dirs."
            )
        for candidate in candidates:
            parts = Path(candidate).parts
            forb_parts = forbidden.parts
            if len(parts) >= len(forb_parts) and parts[: len(forb_parts)] == forb_parts:
                raise RunConfigError(
                    f"Refusing output-dir {output_dir} — overlaps protected path "
                    f"{forbidden}. Prefer outputs/finetune/exp018/... for EXP-018."
                )

    exp_id = (planned_experiment_id or "").strip().upper()
    if exp_id.startswith("EXP-018"):
        # Soft preference: warn via exception only if clearly targeting old EXP-012 name.
        name = resolved.name.lower()
        if "exp012" in name or "gsm_baseline" in name:
            raise RunConfigError(
                f"EXP-018 run must not use EXP-012/011-named output-dir ({output_dir}). "
                "Use outputs/finetune/exp018/{{018a,018b,018c,_smoke_*}}/."
            )
    return resolved


def resolve_run_config(
    run_mode: str,
    max_train: int | None,
    max_eval: int | None,
) -> dict:
    """Resolve subsample caps from --run-mode. Raises RunConfigError on bad combos."""
    mode = (run_mode or "").strip().lower()
    if mode not in {"smoke", "full"}:
        raise RunConfigError("--run-mode must be 'smoke' or 'full'")

    if mode == "full":
        if max_train is not None and max_train > 0:
            raise RunConfigError(
                "full mode forbids --max-train > 0 "
                f"(got {max_train}). Omit it or pass --max-train 0."
            )
        if max_eval is not None and max_eval > 0:
            raise RunConfigError(
                "full mode forbids --max-eval > 0 "
                f"(got {max_eval}). Omit it or pass --max-eval 0."
            )
        return {
            "run_mode": "full",
            "max_train": None,
            "max_eval": None,
            "allow_subsample": False,
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
        }

    # smoke
    train_cap = DEFAULT_SMOKE_MAX_TRAIN if max_train is None else max_train
    eval_cap = DEFAULT_SMOKE_MAX_EVAL if max_eval is None else max_eval
    if train_cap is not None and train_cap <= 0:
        raise RunConfigError(
            "smoke mode requires --max-train > 0 "
            f"(got {train_cap}). Use --run-mode full for uncapped training."
        )
    if eval_cap is not None and eval_cap <= 0:
        raise RunConfigError(
            "smoke mode requires --max-eval > 0 "
            f"(got {eval_cap}). Use --run-mode full for uncapped validation."
        )
    return {
        "run_mode": "smoke",
        "max_train": train_cap,
        "max_eval": eval_cap,
        "allow_subsample": True,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
    }


def select_best_checkpoint(epoch_records: list[dict]) -> dict:
    """Pick best epoch by minimum validation loss. Ties → lower epoch number."""
    if not epoch_records:
        raise RunConfigError("No epoch records available for checkpoint selection.")
    ranked = sorted(
        epoch_records,
        key=lambda row: (
            float("inf") if row.get("valid_loss") is None else float(row["valid_loss"]),
            int(row["epoch"]),
        ),
    )
    best = ranked[0]
    if best.get("valid_loss") is None:
        raise RunConfigError(
            "Cannot select best checkpoint: all epochs missing valid_loss. "
            "Ensure the mask dataset has a non-empty valid split."
        )
    return best


def parse_resume_epoch(resume_from: Path) -> int:
    """Return completed epoch number from a checkpoint-epoch-N directory name."""
    name = resume_from.name
    prefix = "checkpoint-epoch-"
    if not name.startswith(prefix):
        raise RunConfigError(
            f"--resume-from must be a checkpoint-epoch-N directory (got {resume_from})"
        )
    suffix = name[len(prefix) :]
    if not suffix.isdigit() or int(suffix) < 1:
        raise RunConfigError(
            f"--resume-from epoch suffix must be a positive integer (got {name})"
        )
    return int(suffix)


def resolve_resume_start(
    resume_from: Path | None,
    num_epochs: int,
) -> dict:
    """Validate resume path and return 0-based start epoch index + completed epoch."""
    if resume_from is None:
        return {
            "resume_from": None,
            "completed_epoch": 0,
            "start_epoch_idx": 0,
        }
    resume_path = Path(resume_from)
    if not resume_path.is_dir():
        raise RunConfigError(f"--resume-from directory does not exist: {resume_path}")
    completed = parse_resume_epoch(resume_path)
    if completed >= num_epochs:
        raise RunConfigError(
            f"--resume-from {resume_path.name} already completed epoch {completed}, "
            f"but --epochs={num_epochs}. Nothing left to train."
        )
    return {
        "resume_from": resume_path,
        "completed_epoch": completed,
        "start_epoch_idx": completed,  # next epoch index (0-based) == completed count
    }


def _build_mlm_examples(df: pd.DataFrame, tokenizer, *, label: str = "") -> list[dict]:
    examples: list[dict] = []
    total = len(df)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if row_idx % 1000 == 0 or row_idx == total:
            print(
                f"  tokenizing {label or 'rows'}: {row_idx}/{total} "
                f"({len(examples)} examples so far)",
                flush=True,
            )
        original = row["original_text"]
        label_char = row["label_char"]
        mask_pos = int(row["mask_position"])
        label_id = tokenizer.convert_tokens_to_ids(label_char)
        mask_id = tokenizer.mask_token_id
        if label_id is None or label_id == tokenizer.unk_token_id:
            continue
        if mask_pos < 0 or mask_pos >= len(original):
            continue
        if original[mask_pos] != label_char:
            continue

        left = max(0, mask_pos - CONTEXT_WINDOW)
        right = min(len(original), mask_pos + CONTEXT_WINDOW + 1)
        window = original[left:right]
        local_mask = mask_pos - left
        masked_window = window[:local_mask] + tokenizer.mask_token + window[local_mask + 1 :]

        input_ids = tokenizer.encode(masked_window, add_special_tokens=False)
        if mask_id not in input_ids:
            continue
        mask_index = input_ids.index(mask_id)
        labels = [-100] * len(input_ids)
        labels[mask_index] = label_id
        if len(input_ids) > MAX_SEQ_LEN:
            continue
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return examples


class MaskOnlyDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def _collate_batch(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.full((pad,), pad_id, dtype=torch.long),
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(pad, dtype=torch.long),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full((pad,), -100, dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def _stratified_subsample(
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
    era_col: str = "era",
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    rng = seed
    parts: list[pd.DataFrame] = []
    grouped = df.groupby(df[era_col].fillna("unknown"), dropna=False)
    total = len(df)
    for _, group in grouped:
        take = max(1, round(max_rows * len(group) / total))
        parts.append(group.sample(n=min(take, len(group)), random_state=rng))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=rng).reset_index(drop=True)
    return out


def _mean_loss(model, loader, device) -> float | None:
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total += float(outputs.loss.item())
            count += 1
    model.train()
    return total / max(count, 1)


def run_finetune(
    mask_csv: Path,
    output_dir: Path,
    *,
    run_mode: str,
    max_train: int | None,
    max_eval: int | None,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    subsample_seed: int = 42,
    planned_experiment_id: str = "EXP-018",
    resume_from: Path | None = None,
) -> dict:
    config = resolve_run_config(run_mode, max_train, max_eval)
    resolved_max_train = config["max_train"]
    resolved_max_eval = config["max_eval"]
    num_epochs = int(epochs) if epochs >= 1 else 1
    resume_info = resolve_resume_start(resume_from, num_epochs)
    safe_output_dir = assert_safe_output_dir(
        output_dir, planned_experiment_id=planned_experiment_id
    )
    output_dir = Path(output_dir)
    git_commit = get_git_commit()
    mask_csv = Path(mask_csv)
    data_fingerprint = _file_sha256(mask_csv)

    df = pd.read_csv(mask_csv)
    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == "valid"]
    if train_df.empty:
        raise RuntimeError("No train split rows in mask dataset.")
    if eval_df.empty:
        raise RuntimeError(
            "No valid split rows in mask dataset. "
            "Validation-based checkpoint selection requires a non-empty valid split."
        )

    train_full_rows = int(len(train_df))
    eval_full_rows = int(len(eval_df))

    train_subsampled = False
    if (
        config["allow_subsample"]
        and resolved_max_train is not None
        and resolved_max_train > 0
        and len(train_df) > resolved_max_train
    ):
        train_df = _stratified_subsample(train_df, resolved_max_train, subsample_seed)
        train_subsampled = True
    eval_subsampled = False
    if (
        config["allow_subsample"]
        and resolved_max_eval is not None
        and resolved_max_eval > 0
        and len(eval_df) > resolved_max_eval
    ):
        eval_df = _stratified_subsample(eval_df, resolved_max_eval, subsample_seed)
        eval_subsampled = True

    if config["run_mode"] == "full" and (train_subsampled or eval_subsampled):
        raise RuntimeError("Internal error: full mode must not subsample.")

    print(
        f"[run_mode={config['run_mode']}] planned_experiment_id={planned_experiment_id}",
        flush=True,
    )
    print(
        f"[run_mode={config['run_mode']}] "
        f"max_train_cap={resolved_max_train} max_eval_cap={resolved_max_eval} "
        f"allow_subsample={config['allow_subsample']} "
        f"selection={config['selection_rule']}",
        flush=True,
    )
    if config["run_mode"] == "full":
        print(
            "[run_mode=full] Using FULL train/valid splits; "
            "no max-train/max-eval row caps applied.",
            flush=True,
        )
    if resume_info["resume_from"] is not None:
        print(
            f"[resume] loading weights from {resume_info['resume_from']} "
            f"(completed epoch {resume_info['completed_epoch']}; "
            f"next epoch {resume_info['start_epoch_idx'] + 1})",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if resume_info["resume_from"] is not None:
        model = AutoModelForMaskedLM.from_pretrained(str(resume_info["resume_from"]))
    else:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Building {len(train_df)} train / {len(eval_df)} eval rows...", flush=True)
    train_examples = _build_mlm_examples(train_df, tokenizer, label="train")
    eval_examples = _build_mlm_examples(eval_df, tokenizer, label="valid")
    print(
        f"Built {len(train_examples)} train / {len(eval_examples)} eval tokenizable examples",
        flush=True,
    )
    if not train_examples:
        raise RuntimeError("No tokenizable MLM train examples.")
    if not eval_examples:
        raise RuntimeError("No tokenizable MLM valid examples.")

    train_loader = DataLoader(
        MaskOnlyDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer.pad_token_id),
        num_workers=0,
    )
    eval_loader = DataLoader(
        MaskOnlyDataset(eval_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_batch(batch, tokenizer.pad_token_id),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, int(steps_per_epoch * epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(50, total_steps // 10),
        num_training_steps=total_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    early_meta = {
        "planned_experiment_id": planned_experiment_id,
        "git_commit": git_commit,
        "seed": subsample_seed,
        "data_path": str(mask_csv),
        "data_fingerprint_sha256_prefix": data_fingerprint,
        "model_name": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "context_window": CONTEXT_WINDOW,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "run_mode": config["run_mode"],
            "max_train_cap": resolved_max_train,
            "max_eval_cap": resolved_max_eval,
            "subsample_seed": subsample_seed,
        },
        "selection_metric_in_training": SELECTION_METRIC,
        "selection_rule_in_training": SELECTION_RULE,
        "resolved_output_dir": str(safe_output_dir),
        "status": "running",
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(early_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    epoch_records: list[dict] = []
    metrics_path = output_dir / "epoch_metrics.json"
    if resume_info["resume_from"] is not None and metrics_path.exists():
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(prior, list):
            raise RuntimeError(f"Invalid epoch_metrics.json at {metrics_path}")
        epoch_records = [
            row for row in prior if int(row.get("epoch", 0)) <= resume_info["completed_epoch"]
        ]

    global_step = resume_info["completed_epoch"] * steps_per_epoch
    running_loss = 0.0
    state_path = None
    if resume_info["resume_from"] is not None:
        state_path = resume_info["resume_from"] / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state.get("global_step", global_step))
            running_loss = float(state.get("running_loss", 0.0))
            print(f"[resume] restored optimizer/scheduler from {state_path}", flush=True)
        else:
            print(
                f"[resume] no training_state.pt in {resume_info['resume_from']}; "
                "continuing with fresh optimizer/scheduler and loaded model weights only.",
                flush=True,
            )

    model.train()
    start_epoch_idx = resume_info["start_epoch_idx"]

    for epoch_idx in range(start_epoch_idx, num_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.item())
            running_loss += loss_value
            epoch_loss_sum += loss_value
            global_step += 1
            epoch_steps += 1
            if global_step % 25 == 0:
                print(
                    f"step {global_step}/{total_steps} loss={running_loss / global_step:.4f}",
                    flush=True,
                )
            if global_step >= total_steps:
                break

        valid_loss = _mean_loss(model, eval_loader, device)
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch_idx + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir), safe_serialization=False)
        tokenizer.save_pretrained(str(ckpt_dir))
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "running_loss": running_loss,
                "epoch": epoch_idx + 1,
                "learning_rate": learning_rate,
            },
            ckpt_dir / "training_state.pt",
        )

        record = {
            "epoch": epoch_idx + 1,
            "checkpoint_dir": str(ckpt_dir),
            "train_loss_epoch": epoch_loss_sum / max(epoch_steps, 1),
            "valid_loss": valid_loss,
            "steps": epoch_steps,
            "selection_metric": SELECTION_METRIC,
            "used_for_test_selection": False,
        }
        # Replace same-epoch record if re-running that epoch.
        epoch_records = [row for row in epoch_records if int(row["epoch"]) != epoch_idx + 1]
        epoch_records.append(record)
        epoch_records.sort(key=lambda row: int(row["epoch"]))
        (ckpt_dir / "epoch_metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "epoch_metrics.json").write_text(
            json.dumps(epoch_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Epoch {epoch_idx + 1}: valid_loss={valid_loss:.6f} -> {ckpt_dir}",
            flush=True,
        )
        if global_step >= total_steps:
            break

    best = select_best_checkpoint(epoch_records)
    best_dir = Path(best["checkpoint_dir"])
    best_link = output_dir / "best"
    if best_link.exists():
        if best_link.is_dir():
            shutil.rmtree(best_link)
        else:
            best_link.unlink()
    shutil.copytree(best_dir, best_link)

    # Keep last-epoch snapshot as final/ for debugging only — not for official test.
    final_dir = output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    last_dir = Path(epoch_records[-1]["checkpoint_dir"])
    shutil.copytree(last_dir, final_dir)

    run_meta = {
        "planned_experiment_id": planned_experiment_id,
        "git_commit": git_commit,
        "seed": subsample_seed,
        "data_path": str(mask_csv),
        "data_fingerprint_sha256_prefix": data_fingerprint,
        "data_note": (
            "fingerprint = sha256(first 8MiB || size=N); not a full-file hash. "
            "Canonical dataset: artificial_mask_dataset.csv (inscription split, seed=42)."
        ),
        "model_name": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "context_window": CONTEXT_WINDOW,
        "single_mask": True,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "run_mode": config["run_mode"],
            "max_train_cap": resolved_max_train,
            "max_eval_cap": resolved_max_eval,
            "subsample_seed": subsample_seed,
        },
        "selection_metric_in_training": SELECTION_METRIC,
        "selection_rule_in_training": SELECTION_RULE,
        "official_exact5_selection": (
            "post-hoc via src.eval.select_checkpoint_by_exact → best_exact5/ "
            "(DECISIONS [012]); not the in-training best/ alias"
        ),
        "resolved_output_dir": str(safe_output_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selection = {
        "planned_experiment_id": planned_experiment_id,
        "git_commit": git_commit,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "best_epoch": best["epoch"],
        "best_valid_loss": best["valid_loss"],
        "best_checkpoint_dir": str(best_dir),
        "best_alias_dir": str(best_link),
        "resumed_from": (
            str(resume_info["resume_from"]) if resume_info["resume_from"] is not None else None
        ),
        "note": (
            "In-training alias best/ uses validation loss only. "
            "Do not use test Exact@k for model selection. "
            "Preserve every checkpoint-epoch-* until backup; do not delete non-best "
            "epoch dirs before download. "
            "Official epoch adoption for later FT runs: post-hoc valid Exact@5 "
            "(DECISIONS [012]), with valid_loss / lower epoch as tie-break "
            "(see best_exact5/ from select_checkpoint_by_exact). "
            "Official full-test must pass --model-path to the adopted checkpoint."
        ),
        "epoch_records": epoch_records,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "epoch_metrics.json").write_text(
        json.dumps(epoch_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "planned_experiment_id": planned_experiment_id,
        "git_commit": git_commit,
        "run_mode": config["run_mode"],
        "model_name": MODEL_NAME,
        "device": str(device),
        "max_seq_len": MAX_SEQ_LEN,
        "context_window": CONTEXT_WINDOW,
        "data_path": str(mask_csv),
        "data_fingerprint_sha256_prefix": data_fingerprint,
        "train_rows_full_split": train_full_rows,
        "eval_rows_full_split": eval_full_rows,
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_steps": global_step,
        "train_loss": running_loss / max(global_step, 1),
        "eval_loss": best["valid_loss"],
        "best_epoch": best["epoch"],
        "best_valid_loss": best["valid_loss"],
        "best_checkpoint_dir": str(best_dir),
        "best_alias_dir": str(best_link),
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "output_dir": str(output_dir),
        "resolved_output_dir": str(safe_output_dir),
        "max_train_cap": resolved_max_train,
        "max_eval_cap": resolved_max_eval,
        "subsample_seed": subsample_seed,
        "train_subsampled": train_subsampled,
        "eval_subsampled": eval_subsampled,
        "full_data_no_row_cap": config["run_mode"] == "full",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "resume_from": (
            str(resume_info["resume_from"]) if resume_info["resume_from"] is not None else None
        ),
        "resume_completed_epoch": resume_info["completed_epoch"],
        "epoch_records": epoch_records,
        "run_meta": run_meta,
        "official_performance": False,
        "smoke_not_official": config["run_mode"] == "smoke",
    }
    (output_dir / "finetune_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Best checkpoint (by {SELECTION_RULE}): epoch {best['epoch']} "
        f"valid_loss={best['valid_loss']:.6f} -> {best_dir}",
        flush=True,
    )
    print(
        "Preserve all checkpoint-epoch-* directories until backup completes. "
        "Do not delete non-best epoch dirs before download.",
        flush=True,
    )
    print(
        "For full test, pass --model-path to the adopted checkpoint "
        "(post-hoc valid Exact@5 for later FT; see DECISIONS). "
        "Do not select with test metrics.",
        flush=True,
    )
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        required=True,
        help="Required. smoke=small capped run; full=entire train/valid, no row caps.",
    )
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=DEFAULT_MASK_DATASET,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: smoke -> outputs/finetune/exp018/_smoke; "
        "full -> outputs/finetune/exp018/run. "
        "EXP-018 runs should use outputs/finetune/exp018/{018a,018b,018c}/.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Smoke only: cap train rows (default 32). Forbidden in full unless 0/omit.",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Smoke only: cap valid rows (default 16). Forbidden in full unless 0/omit.",
    )
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument(
        "--planned-experiment-id",
        type=str,
        default="EXP-018",
        help="Label written to metrics/run_meta (default EXP-018). Does not invent results.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume after a completed checkpoint-epoch-N directory "
        "(loads weights + training_state.pt if present). Mid-epoch resume is not supported.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_run_config(args.run_mode, args.max_train, args.max_eval)
        num_epochs = int(args.epochs) if args.epochs >= 1 else 1
        resolve_resume_start(args.resume_from, num_epochs)
    except RunConfigError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_SMOKE_OUTPUT_DIR if config["run_mode"] == "smoke" else DEFAULT_FULL_OUTPUT_DIR
        )

    try:
        assert_safe_output_dir(output_dir, planned_experiment_id=args.planned_experiment_id)
    except RunConfigError as exc:
        parser.error(str(exc))

    metrics = run_finetune(
        args.mask_dataset,
        output_dir,
        run_mode=args.run_mode,
        max_train=args.max_train,
        max_eval=args.max_eval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subsample_seed=args.subsample_seed,
        planned_experiment_id=args.planned_experiment_id,
        resume_from=args.resume_from,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
