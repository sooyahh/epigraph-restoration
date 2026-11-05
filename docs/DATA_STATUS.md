# Data Status

최종 갱신: **2026-09-17** (EXP-015 valid 기준선, EXP-018 post-hoc Exact@k 평가 완료·gate 미통과, EXP-019 leakage 감사 반영)

---

## 1. 현재 확정된 것

* 프로젝트 목표: 금석문 결손 한자에 대해 사람이 검토할 수 있는 **top-k 후보** 제시.
* **1차 데이터 원천**: NRICH 금석문 XLS (`data/금석문 사용자 검색_2026-07-03.xls`).
* **전처리 규칙**: `gsm_preprocess_v4` (`src/preprocess/gsm_v4.py`)
  * 실제 결손 → `□`
  * 한글 치환 → `◇` (학습 제외, `quality_flag=review`)
* **활성 데이터셋** (baseline 실험에 사용):

| 파일 | 규모 | 상태 |
|------|------|------|
| `data/processed_layer/readings.csv` | 14,702행 | v4 정규화 완료 |
| `data/model_inputs/artificial_mask_dataset.csv` | 66,537 샘플 | train/valid/test split 연동 |
| `data/model_inputs/splits.csv` | 6,084 inscription | 70/15/15, seed=42 |
| `data/model_inputs/real_defect_inference.csv` | 5,000건 | 데모용 (`□`만) |

* **스키마 정본**: `docs/DATA_SCHEMA.md`
* **split 기준**: `inscription_id` 단위 (`group_split_by_inscription_v1`)
* **정량 평가 데이터**: 인공 single-mask만 (`artificial_mask_dataset.csv`)
* **학습 대상 무결성**: `is_training_text=True` 7,158행에 한글 0자 (DECISIONS [006])
* **SillokBert fine-tuning GPU full-train (EXP-012)**: 완료·채택(DECISIONS [010]). Exact@1 33.75% / @5 53.04% / @10 60.47% (test n=9,719). 로컬 백업 `notebooks/best/`. EXP-011은 subsample baseline.
* **EXP-012 best valid Exact@k 기준선 (EXP-015)**: 완료(DECISIONS [012]). Exact@1 33.78% / @5 53.24% / @10 60.90% (valid n=10,679). 이후 모든 FT 개선·신규 모델(ARI-8B 등) 비교의 **공식 incumbent baseline**으로 고정됨(DECISIONS [014]).
* **(2026-09-17 추가) EXP-018 018A/018B post-hoc valid Exact@k 평가**: 완료(subsample n=2,000, seed=42, checkpoint-epoch-1~4 전체 평가). 018A(best epoch 1) Exact@5 52.20%, EXP-015(동일 subsample 53.40%) 대비 −1.20pp, McNemar p=0.093455 → gate 미통과. 018B(best epoch 2) Exact@5 54.10%, EXP-015 대비 +0.70pp, McNemar p=0.350642(비유의) → gate 미통과. **둘 다 공식 incumbent로 승격하지 않으며, EXP-015를 계속 공식 valid 기준선으로 유지한다.**
* **(2026-09-17 추가) EXP-018 018A/018B checkpoint 저장 경로 확인**: Google Drive `/content/drive/MyDrive/exp018/018a/`, `018b/` 각각에 `checkpoint-epoch-1`~`4` 존재, `model.safetensors`+tokenizer 확인됨. 기존 "저장 위치 TODO" 해소.
* **title_key 붕괴 감사 (DECISIONS [013])**: 동명 비문이 하나의 `inscription_id`로 병합된 결함 확인. 영향 마스크 샘플 587/66,537(0.88%). corpus 전면 재생성은 하지 않고 한계로 기록·보존.
* 원본 PDF·기획 문서·XLS는 보존 대상.

---

## 2. 아직 확정되지 않은 것

* **최종 "검수 완료" 선언** — 현재 corpus는 NRICH XLS 기반 1차 가공본이며, 전수 수동 검수는 미완.
* **이체자 Group@k** 평가 방식 및 매핑 소스.
* **공식 대회/발표 지표**와의 최종 정렬 (`EXACT@k` 후보 확정).
* **국사편찬위 XML** 등 2차 원천의 본격 통합 범위.
* `전면`/`후면`/`출전` 등 구조 표기 제거로 `review` 행 일부 회수 여부 (1,628행 후보, DECISIONS [006]).
* **cross-split exact duplicate 26건의 원인** — DECISIONS [013]의 title_key 붕괴와 동일한 현상인지, 별개(예: 정형구 반복)인지 미확인. retrieval(BM25/ChromaDB) 실험용 corpus를 구성하기 전에 반드시 해소해야 한다(DECISIONS [015]).
* **(2026-09-17 추가) EXP-018 018A/018B 학습 loss 표의 실제 정정 수치** — `drive_loss_vs_docs.csv` 대조 결과 Drive 실제 `best_checkpoint.json` loss 값이 `EXPERIMENT_LOG.md`의 기존 표와 일부 다른 것으로 확인됐으나, 구체 정정 수치를 아직 확보하지 못해 표는 잠정 유지 중이다.
* **EXP-018C(lr=2e-5)** — 여전히 실행 결과 미확인(미실행 또는 결과 미확보). 후순위.
* **ARI-8B(EXP-019) Top-K decoder 구현·GPU 실행** — 설계·저장소 감사만 완료됐고, 실제 모델 로딩·추론·성능 수치는 아직 없음. **현재 최우선 작업.**

---

## 3. archive로 이동된 자료 (사용 금지)

* `_archive/fake_data_cleanup_2026-06-26/` 내 샘플·smoke·toy 데이터
* `data_conversion/bird_ready/`, `preprocessing/outputs/`, `bird_lite/runs/smoke_cpu/` 등
* archive 경로를 참조하는 옛 문서·명령은 **성능 근거로 사용하지 않는다.**

---

## 4. 현재 사용 금지 자료

* archive 안의 샘플/혼합 데이터, PDF 자동 추출 후보, BIRD-ready toy 출력, smoke 성능표.
* v3 기준 결손 통계(475,423개) — 한글 `□` 혼입분 포함. **v4 기준 152,277개**를 사용.
* **(2026-09-17 갱신) EXP-018 018A/018B의 Exact@k 수치는 이제 존재하지만(subsample n=2,000), 둘 다 DECISIONS [012] gate를 통과하지 못했다. "성능 개선에 성공했다"거나 새 공식 baseline으로 인용하지 않는다. full valid(n=10,679) 수치는 존재하지 않으므로 있는 것처럼 쓰지 않는다.**
* `gate_results.csv`가 자동 계산한 018B의 +0.86pp(full-valid EXP-015 53.24% vs subsample 018B 54.10%, 표본 상이)는 paired 개선량으로 인용하지 않는다. 공식 paired 개선량은 동일 subsample 기준 +0.70pp다.
* leakage audit을 통과하지 않은 BM25/ChromaDB retrieval 결과(EXP-019B/C)는 공식 성능으로 인용하지 않는다.

---

## 5. 참고·역사 문서

| 문서 | 용도 |
|------|------|
| `reports/pre_baseline_data_gate_2026-07-03.md` | 7/3 baseline 착수 전 점검 (**v3 시점 스냅샷**) |
| `docs/DECISIONS.md` [004] | v3 결정 이력 |
| `docs/DECISIONS.md` [007] | v4 결손/한글 분리 |
| `docs/DECISIONS.md` [012] | FT 개선 루프 평가 프로토콜(valid Exact@k 기준선 EXP-015, gate 기준) |
| `docs/DECISIONS.md` [013] | title_key 충돌·동명 비문 붕괴 |
| `docs/DECISIONS.md` [014] | EXP-015를 신규 모델 비교의 incumbent baseline으로 고정 |
| `docs/DECISIONS.md` [015] | ARI-8B 통합: critical path, 019C 보류, leakage audit 게이트 |
| `docs/EXPERIMENT_LOG.md` EXP-001, EXP-002 | 데이터 정합성·v4 재생성 |
| `docs/EXPERIMENT_LOG.md` EXP-013 | FT 층화 분석 (hapax 하락) |
| `docs/EXPERIMENT_LOG.md` EXP-018 | SillokBERT lr 재탐색 — GPU 학습 + post-hoc Exact@k 평가(gate 미통과) |
| `docs/EXPERIMENT_LOG.md` EXP-019 | ARI-8B 통합 설계 |

---

## 6. 데이터 파이프라인 (현재)

```text
NRICH XLS (+ 선택적 detail 보강)
  → build_readings.py / reprocess_readings.py (gsm_preprocess_v4)
  → readings.csv
  → build_artificial_mask.py  → artificial_mask_dataset.csv
  → split_by_inscription.py   → splits.csv
  → build_real_defect_inference.py → real_defect_inference.csv
  → run_fill_mask_eval.py     → baseline/FT Exact@k
  → paired_exact_compare.py   → EXP-015 대비 paired 비교 (FT iteration, DECISIONS [012])
  → [설계 완료, 미실행] ARI runner → constrained Top-K → 기존 Exact@k evaluator (EXP-019)
```

---

## 7. 모델 개발 진행 조건

| 조건 | 상태 |
|------|------|
| 학습/평가 데이터 존재 | ✅ 66,537 artificial mask |
| 라벨·마스킹 방식 문서화 | ✅ DATA_SCHEMA, EXP-001 |
| split 기준·버전 기록 | ✅ splits.csv, EXP-001/002 |
| EXPERIMENT_LOG 기록 준비 | ✅ (baseline 결과만 추가하면 됨) |
| fine-tuning 채택 여부 | ✅ EXP-011 subsample(DECISIONS [009]) + EXP-012 full(DECISIONS [010]) |
| GPU full-train + full test(9,719) | ✅ EXP-012 완료 (Exact@1 33.75% / @5 53.04% / @10 60.47%) |
| valid Exact@k 기준선 확보 | ✅ EXP-015 완료 (Exact@1 33.78% / @5 53.24% / @10 60.90%, DECISIONS [012]/[014]) |
| FT 개선 iteration(형식 정합·보간) | ✅ EXP-016 정합 확인 / EXP-017 보간 기각 |
| lr 재탐색 GPU 학습·평가 | ✅ 완료(gate 미통과) — 018A Δ−1.20pp, 018B Δ+0.70pp(McNemar p=0.35, 비유의). **공식 후보 승격 없음, EXP-015 유지** |
| 신규 모델(ARI-8B) 비교 준비 | 🟡 EXP-019 설계·leakage 감사 완료, **구현·실행 없음 — 최우선 작업** |
| retrieval corpus leakage audit | ❌ 미실행 — cross-split duplicate 26건 원인 확인부터 필요(DECISIONS [015]) |

**baseline(EXP-003~007)·CPU subsample(EXP-011)·GPU full-data(EXP-012)·valid 기준선(EXP-015)·lr 재탐색 평가(EXP-018, gate 미통과) 완료. 다음은 EXP-019 Top-K decoder 구현·leakage audit이 최우선이며, 그 외 발표 정합·실제 결손 정성·산출물 외부 백업·EXP-018 loss 표 정정(TODO)이 남아 있다.**
