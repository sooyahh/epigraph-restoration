# GPU 전날 체크리스트 — EXP-012 (완료)

> **공식 ID:** EXP-007=zero-shot SillokBERT(완료) · EXP-011=CPU subsample FT(완료, test n=2,000) · **EXP-012=GPU full-data FT+full test(완료 · DECISIONS [010])**.
> EXP-011의 50.45%@5(n=2,000)를 EXP-012(n=9,719)와 비교·혼동하지 말 것.

실행 환경: **Colab GPU** (PC방 대체). 상세는 `reports/colab_exp012_runbook.md`.
당일 순서: smoke → full-train → best(valid_loss) 확인 → full-test(명시적 `--model-path`).
세션 끊김 시: `--resume-from .../checkpoint-epoch-N` (같은 Colab 출력 디렉터리만).
결과 스냅샷: `reports/exp012_results.md`

---

## 필수 (완료 여부)

- [x] Phase 1 문서 — DECISIONS [007], CURSOR_HANDOFF, PROJECT_OVERVIEW
- [x] `work/run_exp012_gpu.py` + `reports/colab_exp012_runbook.md` + `notebooks/exp012_colab.ipynb`
- [x] `work/check_cuda_env.py`
- [x] `python work/run_step1_verify.py` → split PASS
- [x] finetune/eval `--run-mode {smoke,full}` 안전장치
- [x] `--resume-from` (Colab 세션 끊김 대비)
- [x] 출력 루트: `/content/exp012/` (Drive 용량 이슈로 Drive 대신 로컬 `/content` 사용 후 반출)
- [x] Colab GPU 런타임 + 레포 클론/브랜치 확인

## CPU 선행 (완료 — EXP-012와 별개)

- [x] EXP-004b / EXP-005b full test (n=9719)
- [x] EXP-011 CPU subsample FT baseline (test n=2,000; 공식 ID EXP-011)
- [x] 경량 증거물 `reports/exp011_baseline/`

## Colab 당일 (완료 = EXP-012)

- [x] Colab GPU + `python work/check_cuda_env.py` — CUDA True
- [x] smoke: `python -m src.data.finetune_sillokbert --run-mode smoke ...`
- [x] full-train (`/content/exp012/sillokbert_exp012_gpu`)
- [x] `best_checkpoint.json`에서 best_epoch / valid_loss 확인 (best_epoch=1, ≈3.6619)
- [x] full-test: `--run-mode full --model-path .../best --require-model-path`
- [x] `best/` + `exp012_*_predictions.csv` 로컬 반출 → `notebooks/best/` (Git 금지)
- [x] EXPERIMENT_LOG에 EXP-012 **실행 결과** 기록 + DECISIONS [010]

## 하지 말 것

- `work/run_exp007_gpu.py`로 FT 실행 (retired)
- EXP-011 `sillokbert_gsm_baseline` 재학습·덮어쓰기
- mask dataset·splits 재생성
- test 지표로 checkpoint 재선택
- smoke/EXP-011(n=2,000) 수치를 EXP-012 full-test로 발표
- Cloud CPU / Colab smoke를 공식 성능으로 기록
- full train on CPU를 공식 EXP-012로 사용

---

*상태: EXP-012 completed 2026-07-12. See `reports/exp012_results.md`, `reports/colab_exp012_runbook.md`.*
