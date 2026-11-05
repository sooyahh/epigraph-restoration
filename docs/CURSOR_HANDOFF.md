# CURSOR_HANDOFF.md — Cursor 작업 인계 문서 (2026-07-06 최초 작성, 2026-09-17 EXP-013~019/DECISIONS [011]~[016] 반영)

이 문서 하나만 읽으면 Cursor가 현재 상황을 파악하고 다음 작업을 시작할 수 있도록 정리한 인계서다.

---

## 0. 목차

1. 프로젝트 한 줄 요약
2. 현재 단계
3. 현재 데이터 상태 (스냅샷)
4. 먼저 읽을 문서
5. 절대 규칙 (가드레일)
6. 다음 작업 — ARI-8B 통합(EXP-019) 진행 · EXP-018 Exact@k 평가 · leakage audit
7. 주요 명령어
8. 완료된 작업 (참고)

---

## 1. 프로젝트 한 줄 요약

한국 금석문 판독문의 결손 한자(`□`)를 text-only MLM/LM으로 top-k 복원하는 시스템.
**전문가가 검토할 후보를 제시**하는 것이 목표이며, 정량 평가는 **인공 single-mask 데이터**로만 한다.

---

## 2. 현재 단계

```text
[완료] NRICH XLS 수집 · 전처리 v4 · splits · artificial_mask · real_defect_inference
[완료] Phase 1 baseline — frequency (EXP-003), ngram (EXP-004)
[완료] Phase 2 층화 분석 (EXP-005) — `reports/phase2_stratified_analysis_2026-07-06.md`
[완료] Phase 3 zero-shot — bert-base-chinese (EXP-006), SillokBert (EXP-007)
[완료] 파인튜닝 전 게이트 — 층화 EXP-008, n-gram overlap EXP-009, vocab EXP-010
[완료] Phase 4 SillokBert CPU subsample fine-tuning — EXP-011 (DECISIONS [009])
[완료] Phase 4 GPU full-data fine-tune + full test(9,719) — EXP-012 (DECISIONS [010])
[완료] Phase 4 FT 층화 분석 — EXP-013 (유지, DECISIONS [011]) — hapax 구간 하락 확인
[완료] title_key 충돌 감사 — DECISIONS [013] (마스크 587/66,537 영향, corpus 전면 재생성 안 함)
[완료] EXP-012 best valid Exact@k 기준선 — EXP-015 (DECISIONS [012]) — 이후 모든 FT/신규모델 비교 기준선
[완료] 학습·평가 형식 정합 진단 + context window 스윕 — EXP-016 (±64 유지, ±256 유의 악화)
[완료] ZS/FT 확률 선형 보간 — EXP-017 (FT overall 유의 개선 못함 → 공식 후보는 FT(EXP-015) 유지)
[완료] SillokBERT lr 재탐색 — EXP-018: GPU Phase B(018A/018B) 학습 완료 + **post-hoc valid Exact@k 평가 완료(2026-09-17)**. 018A Δ−1.20pp(gate 미통과), 018B Δ+0.70pp(McNemar p=0.350642, 비유의·gate 미통과) → **공식 후보 승격 없음, EXP-015 유지.** 018C 미실행
[설계 완료] ARI-8B 통합 평가 설계 — EXP-019: Top-K decoding 설계·저장소 leakage 감사까지 완료, **구현·GPU 실행·수치 없음**
[결정] EXP-015를 신규 모델(ARI-8B 등) 비교의 공식 incumbent validation baseline으로 고정 — DECISIONS [014]
[결정] EXP-019 critical path: 019A(no-retrieval)→019B(BM25); 019C(ChromaDB/dense)는 담당자 이탈로 deferred/optional — DECISIONS [015]
[결정] AI Council에 자율 에이전트(GrokBot) 도입 — EXP-018 모니터링·백업 / 논문조사 / EXP-019 하이퍼파라미터 반복 3개 용도 한정 — DECISIONS [016]
[보류] 클라우드 EXP-014(hapax 가설) · real_defect 데모 shortlist — `origin/cursor/demo-cases-hapax-3c60` 미병합 (DECISIONS [011])
[다음] ① **ARI Top-K constrained decoder 구현(EXP-019A 준비) — 최우선** ② cross-split duplicate 26건 leakage audit (DECISIONS [015]) ③ EXP-018 loss 표 정정 수치 확인(TODO) ④ (해당 시) GrokBot에 EXP-018 모니터링·EXP-019 반복 실행 위임(DECISIONS [016])
```

데이터 파이프라인·CPU subsample(EXP-011)·**GPU full-data(EXP-012)**·**층화(EXP-013)**·**valid 기준선(EXP-015)**·**형식 정합(EXP-016)**·**ZS/FT 보간(EXP-017, 기각)**·**lr 재탐색(EXP-018, GPU 학습+평가 완료·gate 미통과)**·**ARI-8B 설계(EXP-019)** 까지 진행됐다. **동일 test n=9,719 공식 비교 수치는 여전히 EXP-012**이며, **valid 기준 일상 비교·신규 모델 비교의 기준선은 여전히 EXP-015**다(DECISIONS [012], [014]). EXP-011 subsample 수치를 full-test 성능으로 대체·혼동하지 않는다. **EXP-018은 2026-09-17부로 post-hoc valid Exact@k 평가까지 완료됐지만, 018A/018B 모두 DECISIONS [012] gate를 통과하지 못해 공식 후보가 아니다** — "Exact@k를 확보했다"와 "gate를 통과해 채택됐다"를 혼동하지 않는다. 원격 클라우드 브랜치의 EXP-014·데모는 **merge하지 않은 채 보류**한다. **Cursor는 GrokBot(자율 에이전트)이 아니며, GrokBot 위임 범위(DECISIONS [016])는 Cursor의 작업 범위와 무관하다 — Cursor는 기존과 동일하게 코드 구현·git 작업을 담당한다.**

### 파인튜닝 전 게이트 (2026-07-06 완료)

| 게이트 | EXP | 산출물 |
|--------|-----|--------|
| BERT 층화 분석 | EXP-008 | `reports/phase3_stratified_analysis_2026-07-06.md` |
| n-gram cross-split overlap | EXP-009 | `reports/ngram_cross_split_overlap_2026-07-06.md` |
| tokenizer vocab coverage | EXP-010 | `reports/tokenizer_vocab_coverage_2026-07-06.md` |

### Phase 1–3 baseline 요약 (test n=9,719)

| EXP | 모델 | Exact@1 | Exact@5 | Exact@10 |
|-----|------|--------:|--------:|---------:|
| EXP-003 | frequency | 2.25% | 6.60% | 10.33% |
| EXP-004 | ngram | 14.48% | 28.07% | 34.67% |
| EXP-006 | bert-base-chinese (cpu) | 19.69% | 34.98% | 42.38% |
| EXP-007 | SillokBert (cpu) | 24.89% | 43.17% | 51.22% |
| **EXP-012** | **SillokBert FT (Colab GPU, best by valid_loss)** | **33.75%** | **53.04%** | **60.47%** |

EXP-006/007은 클라우드 CPU 실행. 로컬 RTX 3060 Ti에서 `device=cuda` 재실행은 선택.

> **EXP-011** (SillokBert CPU subsample fine-tune, DECISIONS [009]): test **subsample n=2,000** 기준 Exact@1 31.30% / @5 50.45% / @10 57.75%. 위 표(test n=9,719)와 **표본 크기가 달라 직접 병기하지 않는다.** full-data 공식 수치는 **EXP-012**. 경량 증거물: `reports/exp011_baseline/` (모델 712MB는 재생성 대상, gitignore). EXP-012 로컬 백업: `notebooks/best/`.

### FT 개선 루프 요약 (valid n=10,679, DECISIONS [012])

| EXP | 내용 | Exact@1 | Exact@5 | Exact@10 | 비고 |
|-----|------|--------:|--------:|---------:|------|
| **EXP-015** | EXP-012 best full-valid 기준선 | **33.78%** | **53.24%** | **60.90%** | 이후 모든 비교의 incumbent baseline (DECISIONS [012], [014]) |
| EXP-016 | 학습·평가 형식 정합 + context window 스윕 | 34.30%(±64) | 53.40%(±64) | 60.70%(±64) | ±64 유지; ±256은 유의 악화(p=1.6e−22) |
| EXP-017 | ZS/FT 확률 선형 보간(α\*=0.8) | 33.78% | 53.21% | 61.21% | vs EXP-015 Δ−0.028pp, p=0.917 → **기각, FT 유지** |
| EXP-018A | lr 재탐색 018A(lr=5e-5), post-hoc 평가 | 34.05% | 52.20% | 59.70% | vs EXP-015(subsample) Δ−1.20pp, p=0.093455 → gate 미통과 |
| EXP-018B | lr 재탐색 018B(lr=3e-5), post-hoc 평가 | 35.65% | 54.10% | 61.55% | vs EXP-015(subsample) Δ+0.70pp, p=0.350642(비유의) → gate 미통과 |

---

## 3. 현재 데이터 상태 (스냅샷, 2026-07-06 원본 · 2026-09-17 duplicate 감사 추가)

### 원천·전처리

| 항목 | 내용 |
|------|------|
| 원천 XLS | `data/금석문 사용자 검색_2026-07-03.xls` (읽기 전용) |
| 전처리 규칙 | **`gsm_preprocess_v4`** (`src/preprocess/gsm_v4.py`) |
| 결손 기호 | 원문 `○`, `▨`, `?` 등 → **`□`** |
| 한글 치환 | **`◇`** (결손과 분리; v3에서 둘 다 `□`이던 문제 해소) |
| 재정규화 | `python -m src.data.reprocess_readings` (기존 `text_raw` 기준) |

### `readings.csv` (14,702행)

| 지표 | 값 |
|------|-----|
| 비문(`inscription_id`) | 6,248 |
| 학습 대상 (`is_training_text=True`) | 7,158행 / 비문 3,400 |
| 실제 결손 (`has_real_defect=True`) | 5,286행 |
| `quality_flag=review` (한글 `◇` 포함) | 3,637행 |
| 실제 결손 `□` 총량 | 152,277 |
| 한글→`◇` 치환 | 323,146자 |

`is_training_text=True`와 `has_real_defect=True`는 **교차 0건**.

> **2026-09-17 추가 (EXP-019 저장소 감사):** 서로 다른 `inscription_id` 사이에서도 cross-split exact duplicate 원문이 **26건** 발견됐다(train/valid 사이 약 405자 장문 중복 포함). DECISIONS [013]의 title_key 붕괴와 동일 원인인지는 **미확인**. retrieval(BM25/ChromaDB) 실험에서는 train-only index만으로 leakage가 없다고 단정할 수 없으므로, EXP-019B/C 전에 반드시 exact/near-duplicate audit을 통과해야 한다(DECISIONS [015], RISK_CHECKLIST §1).

### 모델 입력 데이터

| 파일 | 규모 | 용도 |
|------|------|------|
| `data/model_inputs/artificial_mask_dataset.csv` | 66,537 샘플 | 학습·**Exact@k 정량 평가** |
| `data/model_inputs/splits.csv` | 6,084 inscription | train 70% / valid 15% / test 15% |
| `data/model_inputs/real_defect_inference.csv` | 5,000건 | **데모·정성 분석** (`□`만, 정답 없음) |

> `artificial_mask_dataset.csv`는 Git LFS(~550MB). checkout 후 pointer만 보이면 `git lfs pull` 실행.

### 최근 결정·실험

- `docs/DECISIONS.md` **[007]** — 결손 `□` / 한글 `◇` 분리 (v4)
- `docs/DECISIONS.md` **[013]** — title_key 충돌·동명 비문 붕괴 확인·보존
- `docs/DECISIONS.md` **[014]** — EXP-015를 신규 모델 비교의 incumbent baseline으로 고정
- `docs/DECISIONS.md` **[015]** — ARI-8B 통합: critical path(019A→019B), 019C(ChromaDB) 보류, Top-K 자체 설계, leakage audit 게이트
- `docs/DECISIONS.md` **[016]** — AI Council에 자율 에이전트(GrokBot) 도입, 3개 용도 한정
- `docs/EXPERIMENT_LOG.md` **EXP-002~019** — v4 재생성, Phase 1–4 baseline/FT, valid 기준선, lr 재탐색(gate 미통과), ARI-8B 설계

---

## 4. 먼저 읽을 문서 (순서대로)

1. `docs/PROJECT_OVERVIEW.md` — 프로젝트 방향·평가 원칙·현재 상태
2. `docs/DATA_SCHEMA.md` — 산출 파일 컬럼 정의 (**스키마 정본**)
3. `docs/DATA_PREPROCESSING_RULES.md` §2·§3 — 결손·한글·마스킹 분리
4. `docs/DECISIONS.md` [012], [013], [014], [015], [016] — FT 평가 프로토콜, title_key 붕괴, incumbent baseline, ARI-8B 통합 결정, GrokBot 도입
5. `docs/EXPERIMENT_LOG.md` — 최신 실험(EXP-019) 확인 후 작업 진행

---

## 5. 절대 규칙 (가드레일)

- 원본 XLS·`data/raw_layer/` **이동·삭제·덮어쓰기 금지.** 가공은 새 파일에서만.
- 실제 결손 `□`는 정답이 없으므로 **Exact@k에 쓰지 않는다.** 정량 평가는 `artificial_mask_dataset.csv`만.
- `real_defect_inference.csv`는 **`□` 위치만** 데모용. `◇`(한글 치환)는 포함하지 않는다.
- 모델 `score`/`top*_score`는 **후보 간 상대적 선호 점수**다. 전문가·발표 화면에서 "신뢰 확률"이라 부르지 않는다.
- split은 **inscription 단위** (`splits.csv`). 랜덤 문자 단위 split 금지.
- **신규(2026-09-17): retrieval(BM25/ChromaDB) corpus도 exact/near-duplicate leakage audit을 통과하기 전에는 공식 결과로 보고하지 않는다** (DECISIONS [015]).
- **학습 완료(loss 확보)와 Exact@k 공식 평가 완료를 구분한다.** (2026-09-17: EXP-018 018A/018B는 이제 둘 다 완료됐으나, gate 미통과이므로 여전히 "채택 가능한 결과"로 서술하지 않는다.)
- **표본 크기가 다른 두 수치를 직접 빼서 paired 개선량으로 쓰지 않는다.** (예: full-valid 53.24% vs subsample 54.10% → 직접 뺀 +0.86pp는 오해의 소지가 있다. paired 비교는 반드시 동일 subsample 기준으로만 한다.)
- **신규(2026-09-17): GrokBot 등 자율 에이전트에게는 git add/commit/push, 원본 데이터 쓰기를 위임하지 않는다. 위임 가능 범위는 EXP-018 모니터링·백업, 논문조사, EXP-019 하이퍼파라미터 반복 3가지뿐이다(DECISIONS [016]).**
- 실험 결과는 **`docs/EXPERIMENT_LOG.md`**, 새 결정은 **`docs/DECISIONS.md`**에 기록.
- 전략·평가 설계 변경은 Cursor 단독 결정하지 말고 팀 검토 (`docs/AI_COUNCIL.md`).

---

## 6. 다음 작업 — ARI-8B 통합(EXP-019) 진행 · EXP-018 Exact@k 평가 · leakage audit

```
DECISIONS [008]: SillokBert = primary fine-tuning target.
DECISIONS [009]: EXP-011(CPU subsample fine-tune) = 1차 subsample baseline.
DECISIONS [010]: EXP-012(Colab GPU full-data + full test n=9,719) = 공식 full-test 비교 수치.
DECISIONS [011]: 로컬 EXP-013 유지 · 클라우드 EXP-014·데모 shortlist 보류(미병합).
DECISIONS [012]: FT 개선 루프 = valid Exact@k iteration + paired + test 잔여 2회(시점 미정).
                 checkpoint 채택 = 전 epoch valid Exact@5 최대(동점→valid_loss→낮은 epoch);
                 epoch 산출물 전부 보존·백업.
DECISIONS [013]: title_key 붕괴 확인·기록. corpus 전면 재생성 안 함(영향 0.88%).
                 제목 단독 매칭으로 식별자 생성 금지. cross-split duplicate 26건(EXP-019 감사)과의
                 관계는 미확인 — 확인되면 본 항목 재검토.
DECISIONS [014]: EXP-015를 ARI-8B·retrieval/BM25·reranking 등 신규 모델 비교의 공식
                 incumbent validation baseline으로 고정. 신규 모델은 동일 valid sample_id로 비교.
DECISIONS [015]: EXP-019 critical path = 019A(no-retrieval)→019B(BM25). 019C(ChromaDB/dense)는
                 담당자 이탈로 deferred/optional. Top-K 후보 생성 규칙은 공개 구현 부재로 자체 설계
                 (PAPER/PUBLIC/OURS/UNKNOWN 태깅). BM25/dense retrieval 실험 전 leakage audit 게이트 필수.
DECISIONS [016]: AI Council에 자율 에이전트(GrokBot) 도입. 위임 범위는 EXP-018 GPU 모니터링·백업,
                 논문/2차 원천 조사, EXP-019 하이퍼파라미터 반복 실행 3가지로 한정. git·원본 데이터
                 접근 금지, 완료 선언은 실제 산출물로 재검증(PROMPTS.md §6 템플릿 사용).
EXP-012: Exact@1 33.75% / @5 53.04% / @10 60.47% (test); best_epoch=1, valid_loss≈3.6619.
EXP-015: Exact@1 33.78% / @5 53.24% / @10 60.90% (valid n=10,679) — **모든 비교의 기준선**.
EXP-013: FT 이득이 고빈도 집중, hapax Exact@5 8.2%→6.0%(−2.2pp).
EXP-016: 학습·평가 형식 정합 확인(문제 없음). context window ±64 유지, ±256 유의 악화.
EXP-017: ZS/FT 선형 보간 α*=0.8 → full valid에서 EXP-015 대비 유의 개선 없음(p=0.917) → 기각.
EXP-018: 018A(lr=5e-5)/018B(lr=3e-5) GPU 4-epoch 학습 완료, valid loss는 epoch 1 이후 악화(과적합).
         **2026-09-17 post-hoc valid Exact@k 평가 완료**(subsample n=2,000, seed=42, checkpoint-epoch-1~4 전체,
         repo commit c6df2c2d4c6e215c850df66e1fa0d36c9200895e, GPU NVIDIA L4, test 미사용):
           018A(best epoch=1): Exact@1 34.05% / @5 52.20% / @10 59.70%. vs EXP-015(subsample 53.40%) Δ−1.20pp,
             McNemar p=0.093455 → gate 미통과.
           018B(best epoch=2, Exact@5 기준 — valid_loss 기준 best epoch 1과 다름): Exact@1 35.65% / @5 54.10% / @10 61.55%.
             vs EXP-015(subsample 53.40%) Δ+0.70pp, McNemar p=0.350642(비유의) → gate 미통과.
         **018A/018B 어느 쪽도 공식 incumbent로 승격하지 않는다. full valid(n=10,679) 평가는 하지 않았다.**
         `gate_results.csv`의 자동계산 +0.86pp(표본 상이, full-valid vs subsample)는 paired 개선량으로 쓰지 않는다 —
         공식 paired 개선량은 동일 subsample 기준 +0.70pp뿐이다.
         체크포인트 저장 경로 확인됨: `/content/drive/MyDrive/exp018/018a/`, `018b/` (checkpoint-epoch-1~4,
         model.safetensors+tokenizer 확인). `drive_loss_vs_docs.csv` 대조 결과 기존 loss 표 일부가 Drive 실제 값과
         다를 수 있음이 확인됐으나 구체 정정 수치 미확보(TODO).
         018C(lr=2e-5) 결과 미확보. 후순위로 둠. (GrokBot 모니터링·백업 위임 가능, DECISIONS [016])
EXP-019: ARI-8B(DAMI-Lab/ARI-8B) 통합 가능성 감사·Top-K constrained decoding 설계 완료.
         구현 코드·모델 다운로드·GPU 실행·성능 수치 전부 아직 없음.
         cross-split exact duplicate 26건 확인 — retrieval leakage audit 필요.
         (하이퍼파라미터 sensitivity 반복 실행은 GrokBot 위임 가능, DECISIONS [016])
보류: origin/cursor/demo-cases-hapax-3c60 (EXP-014 hapax 가설 + 데모 7건). merge/삭제 금지.
다음(우선순위):
  (1) ~~EXP-018 018A/018B post-hoc valid Exact@5 공식 평가~~ — **완료(2026-09-17).** 결과는 위 EXP-018 항목 참조.
  (2) **`docs/ARI_TOPK_DESIGN.md` 기준 ARI Top-K constrained decoder 구현 + GPU 없이 가능한 unit test — 최우선.**
  (3) cross-split exact/near-duplicate 26건의 원인(DECISIONS [013]과 동일 여부) 확인 →
      BM25 retrieval corpus leakage audit 설계·실행 (EXP-019B 전제조건).
  (4) B1/B2/Lmax sensitivity 확인 후 Top-K 규칙 freeze → EXP-019A full validation 실행.
  (5) `drive_loss_vs_docs.csv` 기준 EXP-018 018A/018B loss 표 정정 수치 확인 → `EXPERIMENT_LOG.md` 갱신(TODO).
  (6) (해당 시) EXP-018 GPU 재학습 모니터링·백업, 논문/2차 원천 조사, EXP-019 하이퍼파라미터 반복
      실행을 GrokBot에 위임 — PROMPTS.md §6 템플릿 사용, AI 모델 담당이 사전 승인·사후 검증(DECISIONS [016]).
후순위: 발표 수치 정합 · real_defect 데모 · 전문가 패널 · (선택) cuda 재실행 · EXP-018C(lr=2e-5)
model_version 태그: exp012_sillokbert_gpu_best_valid_loss (SillokBERT 계열 공식 비교 기준)
```

**수용 기준:** 새 FT/신규모델 후보는 **일상적으로** EXP-015 valid Exact@k(+ paired, 필요 시 빈도 구간)로 비교한다(DECISIONS [012], [014]).
슬라이드·보고서의 **full-test** 수치는 EXP-012와 일치시키고, EXP-011은 subsample임을 명시한다.
EXP-018(018A/018B)은 post-hoc Exact@5 평가를 완료했지만 gate를 통과하지 못했으므로 "SillokBERT 개선판"으로 발표에 인용하지 않는다 — 시도했으나 통계적으로 유의한 개선을 입증하지 못한 실패 사례로 다룬다.
ARI-8B 관련 어떤 수치도, Top-K 규칙 freeze와 EXP-015 대비 paired 비교 전까지는 "성능이 좋다/나쁘다"로 서술하지 않는다(DECISIONS [015]).
실제 결손 데이터로 정량 Exact@k를 주장하지 않는다. EXP-014는 재개 전까지 인용·merge하지 않는다.
GrokBot이 수행한 작업은 Cursor·팀원이 산출물을 재확인하기 전까지 EXPERIMENT_LOG.md/DECISIONS.md에 반영하지 않는다(DECISIONS [016]).

---

## 7. 주요 명령어 (프로젝트 루트)

```bash
# 단위 테스트
python3 -m unittest discover -s tests

# readings v4 재정규화 (text_raw 유지)
python3 -m src.data.reprocess_readings

# 파생 데이터셋 재생성
python3 -m src.data.build_artificial_mask
python3 -m src.data.build_real_defect_inference

# baseline 평가 (오프라인)
python3 -m src.eval.run_fill_mask_eval \
  --experiment-id EXP003 --experiment-name freq \
  --model-kind frequency --split test --max-samples 500 --overwrite

# EXP-018 post-hoc Exact@5 공식 평가 (018A/018B epoch 1 체크포인트 대상, 미실행 상태)
python3 -m src.eval.select_checkpoint_by_exact --experiment-id EXP018A
python3 -m src.eval.run_ft_eval_chain --experiment-id EXP018A
```

---

## 8. 완료된 작업 (참고)

- NRICH XLS → `readings.csv` 파이프라인 (`src/data/build_readings.py`)
- 전처리 v3 → **v4** (결손/한글 기호 분리)
- `artificial_mask_dataset.csv`, `splits.csv`, `real_defect_inference.csv` 생성
- EXP-001 데이터 정합성, EXP-002 v4 재생성
- Phase 1 baseline (EXP-003 frequency, EXP-004 ngram)
- Phase 2 층화 분석 (EXP-005)
- Phase 3 zero-shot (EXP-006 bert-base-chinese, EXP-007 SillokBert)
- 파인튜닝 전 게이트 (EXP-008 층화, EXP-009 n-gram overlap, EXP-010 vocab coverage)
- Phase 4 EXP-011 SillokBert CPU subsample fine-tune (DECISIONS [009]) — 증거물 `reports/exp011_baseline/`
- Phase 4 EXP-012 GPU full-data FT + full test (DECISIONS [010]) — 로컬 `notebooks/best/`
- EXP-013 Phase 4 FT 층화 분석 (유지, DECISIONS [011]) — `reports/phase4_stratified_analysis_2026-08-01.*`
- title_key 충돌 감사 (DECISIONS [013]) — `scripts/audit_title_key_collapse_mask_impact.py`
- EXP-015 EXP-012 best valid Exact@k 기준선 (DECISIONS [012]) — `reports/exp015_results.md`
- EXP-016 학습·평가 형식 정합 진단 + context window 스윕 — `reports/exp016/`
- EXP-017 ZS/FT 확률 선형 보간(기각) — `reports/exp017/`
- EXP-018 Phase A(코드·Colab kit) + GPU Phase B(018A/018B 학습) — `outputs/finetune/exp018/`
- EXP-018 018A/018B post-hoc valid Exact@k 평가(2026-09-17, gate 미통과) — Drive `/content/drive/MyDrive/exp018/eval_results/20260917T090850Z/`
- EXP-019 ARI-8B 통합 가능성 감사 + Top-K 설계 — `docs/ARI_INTEGRATION_PLAN.md`, `docs/ARI_TOPK_DESIGN.md`
- DECISIONS [014] EXP-015 incumbent baseline 고정, [015] ARI-8B critical path·leakage audit 게이트, [016] GrokBot 도입·가드레일
- `reports/pre_baseline_data_gate_2026-07-03.md` (7/3 시점 스냅샷; v4 수치는 EXP-002 참조)

---

*분업 원칙(`docs/AI_COUNCIL.md`): 전략·평가 설계 판단은 GPT, 코드 구현·수정은 Cursor, 승인된 범위의 자율 실행은 GrokBot(DECISIONS [016]). 작업 결과는 `docs/EXPERIMENT_LOG.md`에, 새 결정은 `docs/DECISIONS.md`에 기록.*
