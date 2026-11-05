# Colab GPU Runbook — EXP-018 Phase B (HP sweep)

> **상태:** Phase A = 코드·테스트·CPU smoke·Colab kit 준비 완료 / **GPU Phase B = 미실행**  
> Model: `ddokbaro/SillokBert` · Data: `data/model_inputs/artificial_mask_dataset.csv`  
> Split: inscription-level, seed=42 (train 46,139 / valid 10,679) · **test 금지 (selection/eval)**  
> Single-mask · context ±64 · max_seq_len 512  
> 기준선 EXP-015 full valid Exact@5 = **53.24%** (DECISIONS [012])  
> Smoke / Phase A CPU 수치는 **공식 성능이 아님**. Exact@k 결과를 이 문서에 기입하지 말 것 (Phase B 후 EXPERIMENT_LOG).

**절대 금지:** `notebooks/best/`, EXP-012 산출물, EXP-011 `sillokbert_gsm_baseline` 덮어쓰기·삭제.

---

## Colab에서 notebook 여는 가장 쉬운 방법

1. [colab.research.google.com](https://colab.research.google.com) → **파일 → 노트북 업로드**
2. PC의 `notebooks/exp018_colab.ipynb` 선택
3. **런타임 → GPU** 후, 셀 **① → ② → ③…** 순서만 실행

(레포 클론은 ① 셀이 자동으로 함. `handoff/exp018` 폴더를 Colab에서 열 필요 없음.)

---

## 아침 5분 체크리스트

- [ ] 브랜치 clone: `cursor/exp018-phase-a-prep-41a8` (또는 위 업로드 방식)
- [ ] `notebooks/exp018_colab.ipynb` + 이 문서 존재
- [ ] `python -m src.data.finetune_sillokbert --help` → `--run-mode`, path guards
- [ ] `python -m src.eval.select_checkpoint_by_exact --help`
- [ ] `python -m src.eval.run_ft_eval_chain --help`
- [ ] 데이터 LFS: `artificial_mask_dataset.csv` (~528MB, 66,537행)
- [ ] Drive 루트: `/content/drive/MyDrive/exp018/` (`018a` / `018b` / `018c` / `_smoke`)
- [ ] Colab GPU: `python work/check_cuda_env.py` → `cuda_available: True`
- [ ] (McNemar용) EXP-015 predictions CSV를 Drive에 배치  
      로컬 준비본: `handoff/exp018/exp015_sillok_ft_valid_predictions.csv`  
      → Drive `/content/drive/MyDrive/exp018/exp015_sillok_ft_valid_predictions.csv`  
      (없으면 아래 workaround)

```bash
git clone -b cursor/exp018-phase-a-prep-41a8 \
  https://github.com/google-ajou-Epigraph/sooyah.git
cd sooyah
git lfs pull --include="data/model_inputs/artificial_mask_dataset.csv"
```

---

## 0) Colab 셋업

```python
from google.colab import drive
drive.mount("/content/drive")
# clone branch (토큰) → /content/sooyah
# pip install -r requirements.txt
# work/check_cuda_env.py · work/colab_preflight.py
```

```bash
export EXP018=/content/drive/MyDrive/exp018
mkdir -p "$EXP018"/{_smoke,018a,018b,018c}
```

---

## 1) GPU smoke (비공식)

```bash
python -m src.data.finetune_sillokbert \
  --run-mode smoke \
  --planned-experiment-id EXP-018 \
  --output-dir /content/drive/MyDrive/exp018/_smoke \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 5e-5
```

성공: `run_meta.json`, `finetune_metrics.json`, `best/`, `checkpoint-epoch-1/`.  
**공식 성능으로 기록하지 말 것.**

---

## 2) Full-train sweeps (4 epoch × 3 LR)

### EXP-018A — lr=5e-5, batch=16, epochs=4

```bash
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-018A \
  --output-dir /content/drive/MyDrive/exp018/018a \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  2>&1 | tee /content/drive/MyDrive/exp018/018a_train.log
```

### EXP-018B — lr=3e-5

```bash
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-018B \
  --output-dir /content/drive/MyDrive/exp018/018b \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  2>&1 | tee /content/drive/MyDrive/exp018/018b_train.log
```

### EXP-018C — lr=2e-5

```bash
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-018C \
  --output-dir /content/drive/MyDrive/exp018/018c \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  2>&1 | tee /content/drive/MyDrive/exp018/018c_train.log
```

OOM 시 `--batch-size 8`. Resume:

```bash
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-018A \
  --output-dir /content/drive/MyDrive/exp018/018a \
  --epochs 4 --batch-size 16 --learning-rate 5e-5 \
  --resume-from /content/drive/MyDrive/exp018/018a/checkpoint-epoch-N
```

학습 중 `best/` = **minimize_valid_loss** alias (유지). 공식 채택은 아래 Exact@5.

---

## 3) Post-hoc Exact@5 selection

```bash
RUN=/content/drive/MyDrive/exp018/018a   # 018b / 018c 반복

# Iteration-speed selection (valid subsample n=2000)
python -m src.eval.select_checkpoint_by_exact \
  --run-dir "$RUN" \
  --run-mode smoke \
  --max-samples 2000 \
  --seed 42 \
  --batch-size 32 \
  --experiment-id EXP-018A \
  --split valid

# 채택 직전: full valid로 재선택 권장
# python -m src.eval.select_checkpoint_by_exact \
#   --run-dir "$RUN" --run-mode full --batch-size 32 \
#   --experiment-id EXP-018A --split valid
```

산출: `best_exact5/`, `exact5_selection_summary.json` (epoch별 Exact@k + valid_loss).  
규칙: max Exact@5 → min valid_loss → lower epoch.

---

## 4) Subsample 2000 + McNemar + stratified

```bash
# EXP-015 predictions: gitignore. Place on Drive before McNemar.
EXP015_PRED=/content/drive/MyDrive/exp018/exp015_sillok_ft_valid_predictions.csv

python -m src.eval.run_ft_eval_chain \
  --run-dir "$RUN" \
  --experiment-id EXP-018A \
  --skip-selection \
  --model-path "$RUN/best_exact5" \
  --batch-size 32 \
  --exp015-predictions "$EXP015_PRED" \
  --output-dir "$RUN/eval_chain"
```

### EXP-015 predictions missing (workaround)

- Committed baseline summary: `reports/evaluation/exp015_sillokbert_valid_summary.csv` (Exact@5=53.24%).
- Full prediction CSV is typically at `outputs/predictions/exp015_sillok_ft_valid_predictions.csv` (local/Drive, gitignored).
- If missing: copy from local backup / Drive, **or** regenerate once from `notebooks/best` (read-only; do not overwrite notebooks/best):

```bash
python -m src.eval.run_fill_mask_eval \
  --run-mode full \
  --experiment-id EXP-015 \
  --experiment-name "EXP-015 valid baseline regen" \
  --model-kind ddokbaro/SillokBert \
  --model-path /path/to/local/exp012/best \
  --require-model-path \
  --split valid \
  --overwrite \
  --predictions-csv "$EXP015_PRED" \
  --summary-csv /tmp/exp015_regen_summary.csv
```

Eval chain skips McNemar gracefully if the file is absent (`gate allow_full_valid` will stay false without p-value).

---

## 5) Conditional full valid

**Gate (subsample):** Exact@5 ≥ 53.24% + **0.7pp** **AND** McNemar p < **0.05** → allow full valid.  
**Adoption:** full valid Exact@5 ≥ 53.24% + **1.0pp** → adoption candidate.

```bash
python -m src.eval.run_ft_eval_chain \
  --run-dir "$RUN" \
  --experiment-id EXP-018A \
  --skip-selection \
  --model-path "$RUN/best_exact5" \
  --batch-size 32 \
  --run-full-valid \
  --exp015-predictions "$EXP015_PRED" \
  --output-dir "$RUN/eval_chain_full"
```

**Never** pass `--split test` for EXP-018 selection/gates.

---

## 백업 (세션 종료 전)

각 `018a`/`018b`/`018c`에 대해 Drive에:

| 산출물 | 비고 |
|---|---|
| `checkpoint-epoch-*` 전부 | best만 받으면 Exact@5 재선택 불가 |
| `best/` | loss alias |
| `best_exact5/` | 공식 후보 |
| `run_meta.json`, `finetune_metrics.json`, `epoch_metrics.json`, `best_checkpoint.json` | git_commit·HP·seed |
| `exact5_selection_summary.json`, `eval_chain/` | paired·stratified·gate |

`notebooks/best`로 복사하지 말 것.

---

## Phase A vs B

| | Phase A (cloud, 본 브랜치) | Phase B (Colab GPU, 내일) |
|---|---|---|
| 학습 | CPU smoke only | 4 epoch × 3 LR |
| Exact@k | 없음 / smoke only | post-hoc + gates |
| EXPERIMENT_LOG | Phase A prep / NOT executed | Phase B 결과 기록 |
| 공식 성능 | 해당 없음 | gate 통과 시만 후보 |
