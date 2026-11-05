# EXP-011 baseline — 경량 증거물 보존 번들

생성: 2026-07-11 · 목적: EXP-011(SillokBERT CPU subsample fine-tuning) 결과를 **git 추적 가능한 경량 형태**로 보존.

> 원본 산출물은 `outputs/`·`work/`(`.gitignore` 대상)에만 있어 이 로컬 PC에만 존재했다.
> 이 폴더는 결과를 **증명·재현 검증**하는 데 필요한 작은 파일만 미러링한 것이다.
> 712MB 모델 가중치는 아래 사유로 **여기에 포함하지 않는다**(재생성 가능).

## 포함 파일 (원본 경로 ↔ 미러)

| 미러 (이 폴더) | 원본 경로 (gitignore) | 크기 | SHA256 |
|---|---|---|---|
| `exp006_sillok_ft_predictions.csv` | `outputs/predictions/exp006_sillok_ft_predictions.csv` | 1.36MB | `dc26b3de24db9940e7ee06fd7cafad7285ebb8f776da3e66945ce70c7988fdad` |
| `finetune_metrics.json` | `outputs/finetune/sillokbert_gsm_baseline/finetune_metrics.json` | 578B | `feadcd912e5f9c206fa15e40f48f67f8889fe892e018d13f2295ecf682b7f88f` |
| `real_defect_demo.csv` | `outputs/predictions/real_defect_demo.csv` | 7.7KB | `f4a577515b357c2354f30aedc6c878813e5953e47b419823973070be3aabe2f7` |
| `step1_verification.json` | `work/step1_verification.json` | 1.7KB | `420570162d146156741a71e42cbda39fe53c17aab26a5392001cda5cc324b1c7` |

파일명 `exp006_...`는 실행 당시 legacy ID를 보존한 것으로, 공식 실험 ID는 **EXP-011**이다 (`docs/EXPERIMENT_LOG.md` EXP-011 참조).

## 포함하지 않은 것 — 학습된 모델 가중치

| 파일 | 크기 | SHA256 | 사유 |
|---|---|---|---|
| `outputs/finetune/sillokbert_gsm_baseline/final/pytorch_model.bin` | 712MB | `4200f77a37f3d6e67c13682ff723160d8aefa364125fc6761b1a992292d82ef4` | 재생성 가능·GPU full-train으로 대체 예정 |

**재생성 방법** (커밋된 코드·데이터·seed로 동일 CPU run):
- 코드: `src/data/finetune_sillokbert.py` (main에 커밋됨. epoch checkpoint 저장 기능은 `backup/cursor-local-ahead`에만 있으나 최종 metrics에는 영향 없음)
- 데이터: `data/model_inputs/artificial_mask_dataset.csv` (커밋됨)
- 설정: `ddokbaro/SillokBert`, lr=5e-5, batch=8, epochs=1, max_train=8000, max_eval=2000, subsample_seed=42, device=cpu
- 학습 metrics: train_loss 4.33, eval_loss 3.74 (재현 시 대조)

원본 712MB 모델을 그대로 보관하려면 git이 아닌 **외장/클라우드**에 사본을 두고 위 SHA256으로 무결성 확인.

## 검증된 결과 (독립 재집계, 2026-07-11)

`exp006_sillok_ft_predictions.csv` 2,000행을 `rank_of_gold`·`top-k` 후보·`is_exact_*` 플래그로 재집계한 결과가 문서 수치와 정확히 일치:

| 지표 | 값 | 적중 수 |
|---|---|---|
| Exact@1 | 31.30% | 626 / 2,000 |
| Exact@5 | 50.45% | 1,009 / 2,000 |
| Exact@10 | 57.75% | 1,155 / 2,000 |

**한계 (성능 과대해석 금지):** CPU · 1 epoch · train 8k/46k subsample · test **2k/9,719 subsample**. 최종·공식 성능이 아니라 GPU full-train 이전의 CPU baseline이다. 비교 zero-shot(동일 n=2,000: 23.65/42.60/50.90)은 `step1_verification.json`의 canonical run.
