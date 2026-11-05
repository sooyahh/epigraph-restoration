# EXP-012 GPU Full-Data Fine-Tuning — Pre-Run Readiness Audit

**감사일:** 2026-07-11  
**기준 브랜치:** `review/exp011-integration`  
**기준 commit:** `5d4f78220e7fdd5880f308feb0b7c944256cc59f`  
**원격 동기화:** `origin/review/exp011-integration` 와 동일 (ahead/behind 없음)  
**감사 범위:** 실행 전 준비 상태만 확인. GPU full training·full test 9,719 평가는 수행하지 않음.  
**수정 범위:** 본 보고서만 신규 작성. 다른 파일은 수정하지 않음.

---

## 0. 작업 시작 시 Git 스냅샷

```
On branch review/exp011-integration
Your branch is up to date with 'origin/review/exp011-integration'.

Changes not staged for commit:
  modified:   reports/exp007_gpu_status.md
  modified:   reports/gpu_eve_checklist.md
  modified:   reports/pc_bang_runbook.md
  modified:   reports/presentation_outline.md
  modified:   reports/presentation_qa.md

HEAD = 5d4f78220e7fdd5880f308feb0b7c944256cc59f
```

위 5개 `reports/*` 미커밋 변경은 번호 체계 안내 배너 주변 공백/줄바꿈 수준의 diff로 보이며, 학습 코드·데이터와는 무관하다. GPU 실행 시에는 **커밋된 `5d4f782` 코드**를 기준으로 하고, 이 미커밋 문서 diff는 학습에 영향을 주지 않는다.

**종합 판정:** 데이터·경량 증거물·기본 CLI는 준비되어 있으나, **공식 실험 ID(EXP-012) 미정비·validation 선택 절차 미문서화·중단 재개 미지원** 때문에 “실수 없이 한 번에” 실행 가능한 상태는 **아니다**. 아래 CRITICAL blocker를 고친 뒤 실행할 것.

---

## 1. 감사 결과 요약 (CRITICAL / WARNING / PASS)

### CRITICAL

| ID | 항목 | 근거 |
|---|---|---|
| C1 | **공식 실험 ID 충돌·미등록** | `docs/EXPERIMENT_LOG.md`에서 **EXP-007 = SillokBERT zero-shot (Phase 3)**. 그런데 `work/run_exp007_gpu.py`, `reports/pc_bang_runbook.md`, `reports/gpu_eve_checklist.md`, `reports/exp007_gpu_status.md`는 GPU full FT를 **EXP-007**로 실행·기록하도록 되어 있다. 저장소 어디에도 **EXP-012** 항목/스크립트가 없다. 그대로 실행하면 zero-shot EXP-007과 산출물·로그가 충돌한다. |
| C2 | **validation 기반 모델 선택이 코드·runbook에 없음** | `PROJECT_OVERVIEW.md` §4, `DECISIONS.md` [009], `EXPERIMENT_LOG` 평가 원칙은 “epoch/설정 선택은 validation만”을 요구한다. `finetune_sillokbert.py`는 epoch 끝마다 `checkpoint-epoch-N`을 저장하지만, **epoch별 validation 지표를 계산·비교하지 않고**, 학습 종료 후 최종 상태의 loss만 한 번 기록한다. runbook은 `.../final`만 평가한다. |
| C3 | **기본값 `--max-train 128` / `--max-eval 64`** | CLI 기본값이 subsample이다. 플래그 없이 실행하면 full-data가 아니다. (`0`이면 full — 파싱 검증 완료.) wrapper/runbook은 `0`을 넘기지만, 수동 실행 실수 위험이 크다. |

### WARNING

| ID | 항목 | 근거 |
|---|---|---|
| W1 | **checkpoint에서 학습 재개 불가** | `checkpoint-epoch-*` 저장은 있으나 `--resume`/로드 경로 없음. 중단 시 runbook의 “재실행”은 **처음부터 재학습**이다. |
| W2 | **예측 CSV score 컬럼 비어 있음** | `top1_score` / `top5_scores` / `top10_scores` 스키마는 있으나 `_format_scores`가 항상 `""`를 넣는다. 문자 후보·rank·Exact 플래그는 저장됨. |
| W3 | **`max_length`/context가 metrics에 미기록** | `MAX_SEQ_LEN=512`, `CONTEXT_WINDOW=64`는 코드 상수. `finetune_metrics.json`에는 seed·batch·lr·epochs·max_*_cap은 있으나 max_length/context_window 필드는 없음. |
| W4 | **`--device` 플래그 없음** | `torch.cuda.is_available()` 자동 선택. GPU 환경에 CUDA 빌드 PyTorch가 있으면 cuda 사용 가능(구조상 OK). 강제 cuda 지정은 불가. |
| W5 | **GPU 오케스트레이션 스크립트·문서가 legacy EXP-007 체계** | `run_exp007_gpu.py`가 끝나면 `work.generate_baseline_report`를 호출하며, 해당 리포트는 canonical ID를 `EXP-002b…EXP-006`(노트북 legacy)로 본다. EXP-012 결과를 공식 표로 반영하지 못함. |
| W6 | **`pytest` 미설치** | `python -m pytest` 실패 (`No module named pytest`). `unittest`로는 13/13 통과. |
| W7 | **`work/_finetune_smoke.py`는 `PYTHONPATH` 필요** | 기본 `python work/_finetune_smoke.py` → `No module named 'src'`. `$env:PYTHONPATH=.` 후 성공. |
| W8 | **로컬 감사 환경은 CPU-only torch** | `torch 2.12.1+cpu`, `cuda_available: False`. PC방에서는 CUDA 빌드 재설치/`check_cuda_env.py` 필수. |
| W9 | **working tree dirty (reports 5파일)** | 학습 코드와 무관하나, “깨끗한 트리에서 실행” 관행과 불일치. |
| W10 | **발표/PC방 문서의 EXP-006/007 = GPU FT 서술** | 배너로 main 체계와 다르다고 고지했으나, 본문은 여전히 “EXP-007 = GPU full train”으로 읽힌다. 실행 당일 혼동 위험. |
| W11 | **MANIFEST 일부 stale** | `reports/exp011_baseline/MANIFEST.md`는 “epoch checkpoint 저장은 backup 브랜치에만”이라고 하나, 현재 `finetune_sillokbert.py`에는 checkpoint 저장이 이미 있다. |

### PASS

| ID | 항목 | 근거 |
|---|---|---|
| P1 | 브랜치·commit 확인 | `review/exp011-integration` @ `5d4f782` |
| P2 | EXP-011 경량 증거물 + SHA256 | `reports/exp011_baseline/` 4파일 존재, MANIFEST 해시와 **전부 일치** |
| P3 | 필요 코드·스크립트 존재 | `src/data/finetune_sillokbert.py`, `src/eval/run_fill_mask_eval.py`, `work/run_exp007_gpu.py`, `work/check_cuda_env.py`, `work/run_step1_verify.py`, `reports/pc_bang_runbook.md`, `requirements.txt` |
| P4 | 데이터 행 수 | readings **14,702** / splits **6,084** / artificial_mask **66,537** (train 46,139 / valid 10,679 / test **9,719**) |
| P5 | inscription split 무중복 | mask 기준 train∩test=0, train∩valid=0, valid∩test=0; splits.csv도 동일 |
| P6 | real defect ≠ 정량 gold | `real_defect_inference.csv` 5,000행, `gold_label` 전부 비어 있음, `usage=demo_only`; mask와 reading_id 교집합 0 |
| P7 | label leak | 표본 500·2,000 + `validate_no_label_leak` PASS; `[MASK]` 치환 정상 |
| P8 | full-data 플래그 파싱 | `--max-train 0 --max-eval 0` → cap `None` (full) 확인 |
| P9 | EXP-011 출력 경로 비충돌 | GPU 기본 제안 경로 `sillokbert_gsm_baseline_gpu` ≠ EXP-011 `sillokbert_gsm_baseline` |
| P10 | 예측 CSV 식별자·gold·top-k | `sample_id`, `label_char`, `top1/5/10_chars`, `rank_of_gold`, Exact 플래그 저장 |
| P11 | import / `--help` | finetune·eval 모듈 import OK, argparse help OK |
| P12 | 단위 테스트 | `python -m unittest discover -s tests -v` → **13 tests OK** |
| P13 | 허용 smoke | `PYTHONPATH=. python work/_finetune_smoke.py` → tokenize 64행 + forward loss 출력 (공식 성능 아님) |
| P14 | 핵심 문서 정합 (개요류) | `PROJECT_OVERVIEW` / `ROLE_GUIDE` / `AI_COUNCIL` / `EXPERIMENT_LOG` / `DECISIONS`는 EXP-011을 최종 성능으로 오인하지 않음; unseen에서 SillokBERT 우위 주장 없음 |
| P15 | 금지 문구 | `"현재 확정 학습 데이터와 확정 스키마는 없다"` / `SRC_AUDIT.md` 참조 **없음**. `CLAUDE.md`는 “존재하지 않는다”고 명시한 경고만 존재(파일 자체 없음 = 의도된 경고). |

---

## 2. 내일 GPU 실행 전 반드시 고쳐야 할 Blocker

1. **실험 ID를 EXP-012로 고정**  
   - `work/run_exp007_gpu.py` 및 runbook의 `--experiment-id EXP-007`, 예측 파일명 `exp007_*`를 **사용하지 말 것**.  
   - `EXPERIMENT_LOG`에 EXP-012 planned 항목을 추가하거나, 최소한 실행 명령·산출물 경로·notes에 `EXP-012`만 쓸 것.  
   - legacy 노트북 문서의 “GPU EXP-007”과 main의 zero-shot EXP-007을 혼동하지 말 것.

2. **validation 선택 절차를 실행 순서에 넣을 것** (코드 수정 또는 수동 절차)  
   - 학습 후 `checkpoint-epoch-1`, `checkpoint-epoch-2`(및 필요 시 `final`)를 **valid split**에서 Exact@k(또는 loss)로 비교.  
   - **승자 체크포인트만** test 9,719에 1회 평가.  
   - 현재 runbook의 “바로 `final` → full test”는 프로젝트 원칙 위반.

3. **full-data 플래그를 명령에 명시**  
   - 반드시 `--max-train 0 --max-eval 0`. 기본값(128/64)으로 실행 금지.

4. **(권장·사실상 운영 blocker)** CUDA PyTorch 확인  
   - `python work/check_cuda_env.py` → `cuda_available: True` 전에는 full train 시작 금지.

코드 수정 없이 당일 진행하려면: 아래 §3–5의 **수동 EXP-012 명령**만 쓰고, `python work/run_exp007_gpu.py`는 **실행하지 않는다**.

---

## 3. 그대로 실행 가능한 GPU smoke 명령

> smoke 결과는 공식 성능이 아니다. EXP-011/012 경로를 덮어쓰지 않도록 전용 출력 디렉터리를 쓴다.

```powershell
cd <레포경로>
python -m pip install -r requirements.txt
# GPU 환경: CUDA 빌드 torch 확인
python work/check_cuda_env.py

# (A) 모듈 경로 smoke — 권장
$env:PYTHONPATH = (Get-Location).Path
python work/_finetune_smoke.py

# (B) 소규모 학습 smoke (수십 샘플, 산출물 분리)
python -m src.data.finetune_sillokbert `
  --mask-dataset data/model_inputs/artificial_mask_dataset.csv `
  --output-dir outputs/finetune/_exp012_smoke `
  --max-train 32 `
  --max-eval 16 `
  --epochs 1 `
  --batch-size 8 `
  --learning-rate 5e-5 `
  --subsample-seed 42
```

성공 기준: (A) forward loss 출력 / (B) `outputs/finetune/_exp012_smoke/finetune_metrics.json` 생성, `device`가 `cuda`인지 확인.  
주의: (B)도 모델 전체를 저장하므로 **수백 MB**가 생긴다. Git에 추가하지 말 것.

---

## 4. 그대로 실행 가능한 full-train 명령 (EXP-012)

> `work/run_exp007_gpu.py` 대신 아래를 사용한다.

```powershell
python -m src.data.finetune_sillokbert `
  --mask-dataset data/model_inputs/artificial_mask_dataset.csv `
  --output-dir outputs/finetune/sillokbert_exp012_gpu `
  --max-train 0 `
  --max-eval 0 `
  --epochs 2 `
  --batch-size 16 `
  --learning-rate 5e-5 `
  --subsample-seed 42 `
  2>&1 | Tee-Object -FilePath work/exp012_gpu_train.log
```

CUDA OOM 시: `--batch-size 8`로 재시도.  
출력 경로 `sillokbert_exp012_gpu`는 EXP-011(`sillokbert_gsm_baseline`)과 분리된다.  
(기존 runbook 경로 `sillokbert_gsm_baseline_gpu`도 EXP-011과 충돌하지 않으나, **실험 ID·파일명 일관성**을 위해 `sillokbert_exp012_gpu` 권장.)

---

## 5. 최종 checkpoint 평가 명령

### 5.1 Validation으로 체크포인트 선택 (필수)

```powershell
# epoch 1
python -m src.eval.run_fill_mask_eval `
  --experiment-id EXP-012-valid-e1 `
  --experiment-name "EXP-012 valid checkpoint-epoch-1" `
  --model-kind ddokbaro/SillokBert `
  --model-path outputs/finetune/sillokbert_exp012_gpu/checkpoint-epoch-1 `
  --split valid `
  --batch-size 32 `
  --overwrite `
  --predictions-csv outputs/predictions/exp012_valid_e1_predictions.csv `
  --summary-csv outputs/evaluation/exp012_selection_summary.csv `
  --notes "EXP-012 checkpoint selection; valid only; not final test"

# epoch 2
python -m src.eval.run_fill_mask_eval `
  --experiment-id EXP-012-valid-e2 `
  --experiment-name "EXP-012 valid checkpoint-epoch-2" `
  --model-kind ddokbaro/SillokBert `
  --model-path outputs/finetune/sillokbert_exp012_gpu/checkpoint-epoch-2 `
  --split valid `
  --batch-size 32 `
  --overwrite `
  --predictions-csv outputs/predictions/exp012_valid_e2_predictions.csv `
  --summary-csv outputs/evaluation/exp012_selection_summary.csv `
  --notes "EXP-012 checkpoint selection; valid only; not final test"
```

Exact@5(또는 사전 합의한 지표)가 높은 쪽을 `BEST_CKPT`로 정한다.  
(`final`은 보통 마지막 epoch와 동일 가중치이므로, 선택 대상은 `checkpoint-epoch-*`로 충분하다.)

### 5.2 Test 최종 평가 (1회)

```powershell
python -m src.eval.run_fill_mask_eval `
  --experiment-id EXP-012 `
  --experiment-name "SillokBERT GPU full-data fine-tuned full test" `
  --model-kind ddokbaro/SillokBert `
  --model-path outputs/finetune/sillokbert_exp012_gpu/<BEST_CKPT> `
  --split test `
  --batch-size 32 `
  --overwrite `
  --predictions-csv outputs/predictions/exp012_sillok_ft_full_predictions.csv `
  --summary-csv outputs/evaluation/evaluation_summary.csv `
  --notes "device=cuda; EXP-012; full train max_train=0 epochs=2; selected by valid Exact@k; full test n=9719" `
  2>&1 | Tee-Object -FilePath work/exp012_gpu_eval.log
```

`<BEST_CKPT>` 예: `checkpoint-epoch-2`.  
`--max-samples`를 넣지 않으면 test 전체(9,719)를 평가한다.

---

## 6. 예상 산출물 경로와 파일 목록

| 경로 | 설명 |
|---|---|
| `outputs/finetune/sillokbert_exp012_gpu/finetune_metrics.json` | 학습 설정·loss 로그 |
| `outputs/finetune/sillokbert_exp012_gpu/checkpoint-epoch-1/` | epoch 1 가중치·tokenizer |
| `outputs/finetune/sillokbert_exp012_gpu/checkpoint-epoch-2/` | epoch 2 가중치·tokenizer |
| `outputs/finetune/sillokbert_exp012_gpu/final/` | 학습 루프 종료 시점 저장본 (~712MB급 예상, Git 금지) |
| `outputs/predictions/exp012_valid_e1_predictions.csv` | valid 선택용 |
| `outputs/predictions/exp012_valid_e2_predictions.csv` | valid 선택용 |
| `outputs/evaluation/exp012_selection_summary.csv` | valid 비교 요약 |
| `outputs/predictions/exp012_sillok_ft_full_predictions.csv` | **test 9,719** 예측 (기대 행 수 9719) |
| `outputs/evaluation/evaluation_summary.csv` | EXP-012 overall(+era) 요약 append/overwrite |
| `work/exp012_gpu_train.log` / `work/exp012_gpu_eval.log` | 실행 로그 |

**덮어쓰면 안 되는 EXP-011 원본:**  
`outputs/finetune/sillokbert_gsm_baseline/`, `outputs/predictions/exp006_sillok_ft_predictions.csv`, `reports/exp011_baseline/*` (파일명·legacy ID 유지).

---

## 7. 결과 검증 명령

```powershell
# split·행 수 게이트
python work/run_step1_verify.py

# 학습 metrics
python -c "import json; print(json.load(open('outputs/finetune/sillokbert_exp012_gpu/finetune_metrics.json',encoding='utf-8')))"

# full-data 여부 (subsample 되면 안 됨)
python -c "import json; m=json.load(open('outputs/finetune/sillokbert_exp012_gpu/finetune_metrics.json',encoding='utf-8')); assert m.get('max_train_cap') in (0,None) and m.get('train_subsampled') is False and m.get('device')=='cuda', m; print('full-data+cuda OK', m['train_rows'], m['eval_rows'])"

# test 예측 행 수·지표 재집계
python -c "import pandas as pd; p=pd.read_csv('outputs/predictions/exp012_sillok_ft_full_predictions.csv'); assert len(p)==9719, len(p); print('n',len(p)); print('Exact@1',p['is_exact_at_1'].mean()); print('Exact@5',p['is_exact_at_5'].mean()); print('Exact@10',p['is_exact_at_10'].mean())"

# EXP-011 미손상
python -c "from pathlib import Path; import hashlib; b=Path('reports/exp011_baseline'); exp={'exp006_sillok_ft_predictions.csv':'dc26b3de24db9940e7ee06fd7cafad7285ebb8f776da3e66945ce70c7988fdad','finetune_metrics.json':'feadcd912e5f9c206fa15e40f48f67f8889fe892e018d13f2295ecf682b7f88f'};
print({k: hashlib.sha256((b/k).read_bytes()).hexdigest()==v for k,v in exp.items()})"
```

---

## 8. Stale 문서·참조 목록

| 위치 | 문제 |
|---|---|
| `work/run_exp007_gpu.py` | GPU FT를 `EXP-007`로 기록; 공식 zero-shot EXP-007과 충돌 |
| `reports/pc_bang_runbook.md` | 목적·명령·성공 확인이 전부 EXP-007; validation 선택 없음; “checkpoint 있으면 재실행”은 resume이 아님 |
| `reports/gpu_eve_checklist.md` | “EXP-007 FT + eval”; EXPERIMENT_LOG에 EXP-007 추가하라고 안내 |
| `reports/exp007_gpu_status.md` | pending PC bang / EXP-007 기록 지시 |
| `reports/presentation_outline.md` / `presentation_qa.md` / `model_comparison_2026-07-04.md` / `baseline_results_2026-07-03.md` / `exp006_case_study.md` | 노트북 legacy 번호(EXP-006=FT, EXP-007=GPU). 배너는 있으나 본문은 여전히 GPU=EXP-007 |
| `work/generate_baseline_report.py` | canonical IDs에 legacy EXP-006 FT만 포함; EXP-012 미지원 |
| `reports/exp011_baseline/MANIFEST.md` | checkpoint 기능이 backup에만 있다는 서술은 현재 코드와 불일치 |
| `docs/DATA_STATUS.md` | `bird_lite/runs/smoke_cpu/` 를 archive 금지 목록으로만 언급 (디렉터리 자체는 저장소에 없음 — 역사 참조로 OK, 실행 경로로 쓰면 안 됨) |
| `docs/AI_COUNCIL.md` | `CLAUDE.md` 부재 경고만 존재 (파일 없음 = PASS에 가깝고, 참조해 작업하면 안 됨) |
| `CLAUDE.md` / `SRC_AUDIT.md` | **파일 없음** (검색상 SRC_AUDIT 참조 문장도 없음) |

**교차검증 통과 (핵심 5문서):**  
`PROJECT_OVERVIEW.md`, `ROLE_GUIDE.md`, `AI_COUNCIL.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md` — EXP-011을 최종 성능으로 쓰지 않음, EXP-006을 현재 공식 FT ID로 쓰지 않음(공식은 EXP-011, Phase3 zero-shot EXP-006은 bert-base-chinese), unseen에서 SillokBERT가 BERT보다 높다는 잘못된 주장 없음.

---

## 9. 수정 권장 순서

1. **실험 ID·경로 고정:** EXP-012, `sillokbert_exp012_gpu`, `exp012_*_predictions.csv`로 runbook/스크립트 문구 정리 (또는 당일 수동 명령만 사용하고 wrapper 비활성).  
2. **validation→test 절차를 runbook 최상단에 명시** (코드에 epoch별 valid 로그를 넣으면 더 안전).  
3. **CLI 기본값 변경 권장:** `--max-train`/`--max-eval` default를 `0`(full)로 바꾸거나, default 유지 시 full 실행 전 assert 출력.  
4. **(가능하면) `--resume-from` 추가** — PC방 시간 제약 대비.  
5. **(가능하면) score 저장** — predictors의 logits/prob를 CSV에 기록.  
6. **metrics에 `max_seq_len`·`context_window`·`commit_hash` 기록**.  
7. stale PC방/발표 문서에 “legacy EXP-007(GPU) → 공식 EXP-012” 매핑을 본문에도 반영.  
8. `generate_baseline_report.py`가 EXP-012를 인식하도록 분리 또는 새 리포트 스크립트.  
9. GPU 환경: CUDA torch 설치 → smoke → full-train → valid 선택 → test 1회 → USB 반출(모델은 Git 금지).

---

## 10. 항목별 상세 (A–E)

### A. Git 및 산출물

- 브랜치/해시: 위 §0.  
- EXP-011 번들: `MANIFEST.md` + 4파일, SHA256 전원 OK.  
- 712MB 모델은 gitignore 로컬 `outputs/finetune/sillokbert_gsm_baseline/final/`에 존재(감사 시점 EXISTS).  
- 깨진 필수 경로: 학습 파이프라인 핵심 경로는 존재. `CLAUDE.md`/`SRC_AUDIT.md`/`bird_lite/`는 없음(문서가 실행 전제로 쓰지 않음).

### B. 데이터 불변조건

| 검사 | 결과 |
|---|---|
| readings.csv | 14,702 |
| splits.csv | 6,084 (train 4258 / valid 912 / test 914) |
| artificial_mask_dataset.csv | 66,537 |
| test split | 9,719 |
| mask inscription 교집합 | 전부 0 |
| is_training_text=True | 7,158 |
| real defect gold | 비어 있음 / demo_only |
| label leak | PASS |

### C. EXP-012 실행 코드

| 검사 | 결과 |
|---|---|
| full-data 기본값 | **FAIL 위험** — default 128/64 (C3) |
| device=cuda | 자동 선택 가능; 강제 플래그 없음 (W4) |
| seed/batch/lr/epoch 로그 | PASS (`finetune_metrics.json`) |
| max length 로그 | FAIL (W3) |
| epoch별 valid + best 선택 | FAIL (C2) — 저장만 하고 선택 없음 |
| test 분리 | 학습은 train/valid만 사용(PASS). 평가 스크립트는 `--split`으로 분리 가능. runbook이 바로 test로 가는 것이 문제. |
| 예측 CSV | sample_id/gold/top-k PASS; scores 빈 문자열 (W2) |
| resume | FAIL (W1) |
| EXP-011 덮어쓰기 | 경로 분리 시 PASS |
| EXP-012 독립 ID | 코드/문서 미비 (C1) |

### D. 실행 가능성

| 명령 | 결과 |
|---|---|
| `python -m src.data.finetune_sillokbert --help` | OK |
| `python -m src.eval.run_fill_mask_eval --help` | OK |
| import | OK (`ddokbaro/SillokBert`, max_len 512) |
| `python work/check_cuda_env.py` | `cuda_available: False` (이 PC) |
| `python -m pytest` | **실패** — pytest 미설치 |
| `python -m unittest discover -s tests -v` | **13 OK** |
| `python work/_finetune_smoke.py` | **실패** — `No module named 'src'` |
| `PYTHONPATH=. python work/_finetune_smoke.py` | **OK** (loss≈3.06, 비공식) |
| full-train / full test 9,719 | **의도적으로 미실행** |

### E. 문서 정합성

- 금지 문구 `"현재 확정 학습 데이터와 확정 스키마는 없다"`: 미발견.  
- `bird_lite`: `.gitignore` + DATA_STATUS archive 목록만.  
- `CLAUDE.md`: AI_COUNCIL 경고만.  
- `SRC_AUDIT.md`: 미발견.  
- EXP-011 최종 성능 오인: 핵심 docs는 방지됨; 발표/PC방 legacy 문서는 GPU를 EXP-007로 불러 혼동 가능.  
- EXP-006 공식 FT ID 오인: 핵심 docs는 EXP-011로 정리; legacy 보고서는 배너 고지 하에 EXP-006 유지.  
- unseen SillokBERT > BERT 오주장: DECISIONS/EXPERIMENT_LOG/CURSOR_HANDOFF가 **반대(3.7% < 4.1%)**를 명시.

---

## 11. 감사 환경·제약 기록

- 감사 머신: Windows, Python 3.11.9, torch **CPU** 빌드.  
- 수행하지 않은 것: GPU full training, test 9,719 신규 평가, 데이터 재생성, main 수정/병합, commit/push/PR, 100MB+ git 추가, EXP-011 원본 파일명 변경.  
- 본 작업 산출물: `reports/pre_gpu_readiness_audit_2026-07-11.md` 만.

---

*End of audit.*
