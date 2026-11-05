# Colab GPU Runbook — EXP-012 (실행 완료)

> **공식 ID:** EXP-007 = SillokBERT zero-shot(완료) · EXP-011 = CPU subsample FT(완료, test n=2,000) · **EXP-012 = GPU full-data FT + full test n=9,719 (완료 · 2026-07-12 · DECISIONS [010])**  
> Cloud CPU smoke·부분 실행 수치는 공식 성능이 아니다.  
> EXP-011 `outputs/finetune/sillokbert_gsm_baseline/` 는 덮어쓰지 않는다.  
> **결과:** Exact@1 33.75% / @5 53.04% / @10 60.47% · 로컬 `notebooks/best/` · 스냅샷 `reports/exp012_results.md`

**학습 방식 요약:** pretrained `ddokbaro/SillokBert`에 대해 인공 single-char mask MLM fine-tuning.  
마스크 ±64자 윈도우, mask 위치만 감독, epoch별 **valid MLM loss 최소**로 `best/` 선택 후 test Exact@k 1회.

**체크포인트 선택:** `minimize_valid_loss` (test 지표로 선택 금지)

아래 본문은 **재현·재실행용** 절차다. 공식 수치는 위 결과·EXPERIMENT_LOG를 따른다.

---

## 아침 인수인계 (Cloud 밤 작업 후)

- **브랜치:** `cursor/cloud-agent-1783706016574-58a09` (PR #11)
- **클론 예:**
  ```bash
  git clone -b cursor/cloud-agent-1783706016574-58a09 https://github.com/google-ajou-Epigraph/sooyah.git
  cd sooyah
  ```
- [ ] 이 문서 + `notebooks/exp012_colab.ipynb` 존재
- [ ] `python -m src.data.finetune_sillokbert --help` 에 `--run-mode`, `--resume-from`
- [ ] `python work/colab_preflight.py` → PASS
- [ ] 데이터: `data/model_inputs/artificial_mask_dataset.csv` (66,537행)
- [ ] Drive 출력 루트: `/content/drive/MyDrive/exp012/`
- [ ] Colab 런타임: **GPU** (`cuda_available: True`)

---

## 0) Colab 셋업

```python
# 예시: Drive 마운트 후 레포 클론
from google.colab import drive
drive.mount("/content/drive")

# REPO를 Drive 또는 /content 아래에 둔다
%cd /content
# !git clone -b <브랜치> https://github.com/google-ajou-Epigraph/sooyah.git
%cd sooyah
!python -m pip install -r requirements.txt
# GPU torch가 필요하면 Colab 기본 CUDA wheel 사용 (CPU wheel로 덮어쓰지 말 것)
```

CUDA 확인:

```bash
python work/check_cuda_env.py
# cuda_available: True 필수
```

출력 디렉터리를 Drive로 두는 것을 권장 (세션 끊김 대비):

```bash
export EXP012_OUT=/content/drive/MyDrive/exp012/sillokbert_exp012_gpu
mkdir -p "$EXP012_OUT"
```

---

## 1) GPU smoke (비공식)

```bash
python -m src.data.finetune_sillokbert \
  --run-mode smoke \
  --planned-experiment-id EXP-012 \
  --output-dir /content/drive/MyDrive/exp012/_exp012_smoke \
  --epochs 1 \
  --batch-size 8
```

성공: `best_checkpoint.json` 생성, 로그에 `device=cuda`, `run_mode=smoke`.  
**공식 성능으로 기록하지 말 것.**

---

## 2) Full-train (EXP-012)

```bash
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-012 \
  --output-dir "$EXP012_OUT" \
  --epochs 2 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  2>&1 | tee /content/drive/MyDrive/exp012/exp012_gpu_train.log
```

확인 로그:
- `Using FULL train/valid splits; no max-train/max-eval row caps applied.`
- epoch마다 `valid_loss=... -> .../checkpoint-epoch-N`
- 종료 시 `Best checkpoint (by minimize_valid_loss): ...`

OOM 시: `--batch-size 8`.

`work/run_exp007_gpu.py` 사용 금지 (retired). Colab에서는 위 모듈 명령을 직접 쓴다.

---

## 3) 세션 끊김 → resume

완료된 epoch 체크포인트에서만 재개 가능 (epoch 중간 저장 없음).

```bash
# 예: epoch 1까지 끝난 뒤 끊긴 경우
python -m src.data.finetune_sillokbert \
  --run-mode full \
  --planned-experiment-id EXP-012 \
  --output-dir "$EXP012_OUT" \
  --epochs 2 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --resume-from "$EXP012_OUT/checkpoint-epoch-1" \
  2>&1 | tee -a /content/drive/MyDrive/exp012/exp012_gpu_train.log
```

- `training_state.pt`가 있으면 optimizer/scheduler도 복구한다.
- 없으면 가중치만 로드하고 optimizer는 새로 시작한다 (로그에 경고).

---

## 4) Best checkpoint 확인

```bash
python - <<'PY'
import json
from pathlib import Path
import os
out = Path(os.environ.get("EXP012_OUT", "outputs/finetune/sillokbert_exp012_gpu"))
print((out / "best_checkpoint.json").read_text(encoding="utf-8"))
PY
```

필드: `best_epoch`, `best_valid_loss`, `best_alias_dir` (= `.../best`).  
`final/`은 마지막 epoch 스냅샷 — 공식 test에 쓰지 않는다.

---

## 5) Full test (best 명시, 1회)

```bash
python -m src.eval.run_fill_mask_eval \
  --run-mode full \
  --experiment-id EXP-012 \
  --experiment-name "SillokBERT GPU full-data fine-tuned full test" \
  --model-kind ddokbaro/SillokBert \
  --model-path "$EXP012_OUT/best" \
  --require-model-path \
  --split test \
  --batch-size 32 \
  --overwrite \
  --predictions-csv /content/drive/MyDrive/exp012/exp012_sillok_ft_full_predictions.csv \
  --summary-csv /content/drive/MyDrive/exp012/evaluation_summary.csv \
  --notes "planned EXP-012; Colab GPU; model_path=best by valid_loss; full test n=9719" \
  2>&1 | tee /content/drive/MyDrive/exp012/exp012_gpu_eval.log
```

기대: 예측 CSV **9719행**. test로 checkpoint를 다시 고르지 않는다.

---

## 성공 확인 / 반출

| 산출물 | 기대 |
|---|---|
| `$EXP012_OUT/best/` | 선택 가중치 (~712MB급) |
| `$EXP012_OUT/checkpoint-epoch-*` | **모든 epoch** (best만 받으면 안 됨) |
| `$EXP012_OUT/best_checkpoint.json` | best_epoch, valid_loss |
| `$EXP012_OUT/epoch_metrics.json` | epoch별 valid_loss |
| `$EXP012_OUT/finetune_metrics.json` | `run_mode=full`, `full_data_no_row_cap=true`, `device=cuda` |
| `exp012_sillok_ft_full_predictions.csv` | 9719행 |
| EXP-011 SHA / 경로 | 불변 |

### 백업 필수 (세션 종료 전)

Colab `/content`는 세션 종료 시 소멸한다. Drive 또는 로컬로 **아래를 전부** 받는다.

1. `checkpoint-epoch-1/`, `checkpoint-epoch-2/`, … (비-best 포함)
2. `best/` (loss alias; 후속 FT는 post-hoc valid Exact@5로 epoch 재채택 가능 — DECISIONS 평가 프로토콜)
3. `best_checkpoint.json`, `epoch_metrics.json`, `finetune_metrics.json`
4. 예측·summary CSV

**best만 백업하면** loss가 아닌 Exact@k 기준으로 epoch를 다시 고를 수 없다 (EXP-012 epoch 2 소실 사례).

모델을 Git에 추가하지 말고 Drive·로컬로만 보관한다.  
공식 실행 결과는 `docs/EXPERIMENT_LOG.md` EXP-012 · DECISIONS [010] · `notebooks/best/` · `reports/exp012_results.md`에 기록됨.  
후속 FT MANIFEST는 `reports/exp012_artifacts/MANIFEST.md` 패턴을 따르되 **epoch별** 파일·SHA를 넣는다 (`reports/ft_checkpoint_manifest_TEMPLATE.md`).
