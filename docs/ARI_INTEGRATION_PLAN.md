# ARI Integration Plan

상태: **inspection / design only** (2026-09-16)  
범위: repository 감사 → 통합 설계. 코드 수정·모델 다운로드·GPU 실행·기존 EXP 결과 변경 없음.

비교 기준 (DECISIONS [014]): **EXP-015**가 incumbent validation baseline이다.

- 데이터: `data/model_inputs/artificial_mask_dataset.csv`
- split: `valid`, n=**10,679**, inscription-level, seed=42
- Exact@1 **33.78%** / Exact@5 **53.24%** / Exact@10 **60.90%**
- 주력 지표: **Exact@5**
- test는 모델 확정 전까지 사용하지 않음

EXP-018은 018A/018B GPU 학습(loss)만 완료되었고 **Exact@k 공식 평가는 미실행**이다. ARI 비교 상대는 EXP-018이 아니라 EXP-015다.

---

## 1. Current Pipeline

현재 정량 파이프라인은 BERT 계열 **single-mask MLM**이다. 실제 코드 경로:

```text
NRICH XLS
  → readings.csv          (전처리 v4)
  → splits.csv            (inscription 단위)
  → artificial_mask_dataset.csv
  → [학습] finetune_sillokbert.py
  → [평가] run_fill_mask_eval.py
       FillMaskPredictor → logits → Top-K 한자
       compute_exact_metrics → Exact@1/5/10
  → predictions CSV + summary CSV
```

### 1.1 원본 금석문 데이터

| 항목 | 내용 |
|---|---|
| 파일 | `data/금석문 사용자 검색_2026-07-03.xls` (원본, 수정 금지) |
| 빌더 | `src/data/build_readings.py` → `build_readings_dataframe()` |
| 보조 | `src/nrich/xls_parser.py` (`iter_readings_with_meta`), `src/nrich/list_index.py` (`TitleIndexBuilder`) |
| 출력 | `data/processed_layer/readings.csv` (14,702행) |
| ID | `source_record_id` = NRICH `KSM_IDX`; `inscription_id` = `insc_nrich_{ksm_idx:06d}` |
| 입력 | XLS 판독문 + 메타(`era`, `year`, `location`, 제목) |
| 출력 행 | `text_raw`, `text_normalized`, `is_training_text`, `has_real_defect`, `era`, `year` 등 (`docs/DATA_SCHEMA.md` §5) |
| 다음 단계 | `is_training_text=True` 행만 인공 마스크 원천 |

식별자 생성 한계: `src/nrich/text_utils.py` `normalize_title_key()`로 한글+한문 제목만 매칭한다. DECISIONS [013] — 동명 이물 비문이 같은 `inscription_id`로 붕괴될 수 있다. 마스크 영향 587/66,537 (0.88%).

재정규화 엔트리: `src/data/reprocess_readings.py`가 기존 `readings.csv`에 `gsm_preprocess_v4`를 다시 적용한다.

### 1.2 Preprocessing

| 항목 | 내용 |
|---|---|
| 파일 | `src/preprocess/gsm_v4.py` |
| 함수 | `preprocess_reading_text(text_raw) -> PreprocessResult` |
| 규칙 버전 | `gsm_preprocess_v4` |
| 입력 | 원문 문자열 |
| 출력 | `text_normalized`, `has_real_defect`, `defect_count`, `is_training_text`, `quality_flag` |

동작:

- 결손 표기(`□`, `▨`, `○`, `?` 등) → `□`
- 한글 → `◇` (위치 보존)
- 공백/줄바꿈 제거: `collapse_whitespace()` (`src/nrich/text_utils.py`)
- `is_training_text = (quality_flag == "ok") and (not has_real_defect)`
- 실제 결손 또는 한글 혼입 판독문은 **인공 마스크 학습/평가에서 제외**

한자 판정: `src/nrich/text_utils.py` `is_cjk_ideograph()`.

### 1.3 Split

| 항목 | 내용 |
|---|---|
| 파일 | `src/data/split_by_inscription.py` |
| 함수 | `assign_inscription_splits(readings, seed=42, train_ratio=0.7, valid_ratio=0.15)` |
| 산출 | `data/model_inputs/splits.csv` |
| 규칙 | `group_split_by_inscription_v1` |
| 단위 | `inscription_id` (문자/문장 단위 랜덤 split 아님) |

`is_training_text=True` inscription을 seed=42로 shuffle한 뒤 70/15/15 절단. 같은 `inscription_id`는 한 split에만 속한다. 실측: train∩valid∩test inscription 교집합 **0건**, reading 교집합 **0건**.

`splits.csv`는 6,084 inscription(학습 가능 전체)을 담고, 마스크 샘플이 있는 inscription은 3,395개다. 마스크가 없는 split 행은 평가에 쓰이지 않는다.

### 1.4 훼손 sample 생성 (인공 single-mask)

| 항목 | 내용 |
|---|---|
| 파일 | `src/data/build_artificial_mask.py` |
| 함수 | `build_artificial_mask_dataframe()`, `_masks_for_reading()`, `_eligible_positions()`, `validate_no_label_leak()` |
| 산출 | `data/model_inputs/artificial_mask_dataset.csv` (66,537행) |
| 규칙 | `single_hanja_mask_v1`, `mask_type=single_char` |
| seed | 42 (`random.Random(42)`) |

샘플 정의:

1. `is_training_text=True` 판독문만 사용
2. CJK 한자 위치만 eligible
3. reading당 마스크 수 = `min(32, max(1, n_hanja // 40))`
4. 그 위치에서 **한 글자만** `[MASK]`로 치환
5. gold = 그 위치의 원 한자 `label_char` (항상 길이 1)
6. `split`은 해당 `inscription_id`의 `splits.csv` 값

한 원문에 한자가 여러 개여도 **한 sample에는 마스크가 하나**다. 같은 원문에서 여러 sample이 나오면 `sample_id`만 다르고 모두 같은 split에 남는다.

원문 전체 `masked_input`은 저장되지만, 학습/평가는 아래 window를 다시 만든다.

라벨 누수 검사: `validate_no_label_leak()` — `original_text[mask_position] == label_char`이고 `masked_input.replace("[MASK]", label, 1) == original_text`.

실제 결손 추론셋은 별도: `src/data/build_real_defect_inference.py` → `real_defect_inference.csv`. `gold_label` 없음. Exact@k에 쓰지 않음 (`src/eval/run_real_defect_demo.py`).

### 1.5 `[MASK]` 입력 생성

| 항목 | 내용 |
|---|---|
| 파일 | `src/eval/mask_window.py` |
| 함수 | `build_masked_window(original_text, mask_position, label_char, mask_token, context_window=64)` |
| 상수 | `CONTEXT_WINDOW=64`, `MAX_SEQ_LEN=512` |

입력: 원문, 0-based 위치, gold 글자, mask token.  
출력: `(window_text_with_mask, local_mask_position)` 또는 위치/라벨 불일치 시 `None`.

좌우 64자만 남기고 해당 글자를 `tokenizer.mask_token` (BERT는 `[MASK]`)으로 치환한다. EXP-015/012/016 공식 평가는 이 window다. EXP-016에서 ±128/±256은 개선이 없고 ±256은 유의 악화.

호출 지점:

- 평가: `src/eval/predictors.py` `FillMaskPredictor._window_for_row()`
- 평가 CSV `input_text`: `src/eval/run_fill_mask_eval.py` `_input_text_for_row()`
- 학습: `src/data/finetune_sillokbert.py` `_build_mlm_examples()`가 동일 ±64 로직을 인라인 복제

서빙 스텁 `handoff/backend/app.py` `_windowed()`도 동일 ±64.

### 1.6 Tokenizer / Model

학습 (`src/data/finetune_sillokbert.py`):

- `AutoTokenizer` / `AutoModelForMaskedLM` from `ddokbaro/SillokBert`
- `_build_mlm_examples()`: window 인코딩, mask 위치 label만 실제 id, 나머지 `-100`
- gold가 UNK이면 **학습 샘플 스킵**
- `len(input_ids) > 512`이면 스킵
- `MaskOnlyDataset` + `_collate_batch()` 패딩만. `DataCollatorForLanguageModeling` 미사용 (EXP-016 확인: 학습·평가 모두 단일 마스크)

평가 (`src/eval/predictors.py` `FillMaskPredictor`):

- `AutoTokenizer.from_pretrained(model_name_or_path)`
- `AutoModelForMaskedLM.from_pretrained(...).eval()`
- `tokenizer(..., padding=True, truncation=True, max_length=512)`

EXP-015 가중치: `notebooks/best/` (`model_version=exp012_sillokbert_gpu_best_valid_loss`). Git 밖.

### 1.7 Prediction logits → Top-K

`FillMaskPredictor._predict_batch_model_scored()`:

1. 첫 번째 mask token 위치의 logits
2. softmax
3. `topk(k*3)` token id
4. `##` prefix 스킵
5. **길이 1 토큰만** 후보. 중복 제거
6. 최대 k개 (평가는 k=10)

점수: vocab softmax 상대 선호. 보정된 신뢰확률이 아님 (`DATA_SCHEMA.md` §9.3).

baseline 경로: `FrequencyBaseline` / `NgramContextBaseline` → `BaselinePredictor.predict_topk()`. BERT 비교의 하한선.

### 1.8 Evaluation → 결과 저장

| 항목 | 내용 |
|---|---|
| 엔트리 | `src/eval/run_fill_mask_eval.py` `run_evaluation()` / `main()` |
| 지표 | `src/eval/metrics.py` `compute_exact_metrics()`, `rank_of_gold()` |
| 예측 CSV | `outputs/predictions/` (gitignore) |
| 커밋 요약 | `reports/evaluation/{exp}_{model}_{split}_summary.csv` |

EXP-015 명령 (실험 로그 원문):

```text
python -m src.eval.run_fill_mask_eval --run-mode full --split valid
  --model-kind ddokbaro/SillokBert --model-path notebooks/best --require-model-path
  --batch-size 16 --overwrite --committed-summary
```

FT 후처리 체인 (EXP-018용, Exact@k는 아직 미실행):

1. `src/eval/select_checkpoint_by_exact.py` — valid Exact@5 max → `best_exact5/`
2. `src/eval/run_ft_eval_chain.py` — subsample 2000 + McNemar vs EXP-015 + 층화 + gate
3. `src/eval/paired_exact_compare.py` — 동일 `sample_id` McNemar
4. `src/eval/stratified_analysis.py` — 빈도/시대 층화

재현 메타 (학습): `run_meta.json` + `finetune_metrics.json` (`git_commit`, seed, data fingerprint, HP). 평가 쪽은 summary CSV `notes`와 `reports/exp015_results.json`.

---

## 2. Current Evaluation

최종 evaluator: **`src/eval/metrics.py` + `src/eval/run_fill_mask_eval.py`**.

EXP-015는 이 스크립트로 full valid를 돌렸다. EXP-018 Exact@k가 실행됐다면 같은 evaluator를 쓰도록 짜여 있으나 **아직 실행되지 않았다**.

### 2.1 Exact@1 / Exact@5

`rank_of_gold(top_chars, gold)`는 후보 리스트를 **앞에서부터** 스캔해 첫 일치 순위를 1-based로 반환한다. 없으면 `None`.

```text
Exact@1  = (# samples where rank == 1) / N
Exact@5  = (# samples where rank is not None and rank <= 5) / N
Exact@10 = (# samples where rank is not None and rank <= 10) / N
```

비교는 **문자 완전 일치** (`ch == gold`). 이체자 Group@k는 매핑 미확정이라 공식 지표가 아니다.

### 2.2 후보 순서가 영향을 주는가

**준다.** Exact@1은 1등만 본다. Exact@5는 상위 5개 **안의 존재**이므로 2~5등 순서는 Exact@5 값 자체에는 영향 없고, Exact@1과 rank 기록에는 영향 있다. 빈 리스트/5개 미만이면 그 길이까지만 본다.

### 2.3 Gold는 어디에 있는가

`artificial_mask_dataset.csv`의 **`label_char`**. 예측 CSV에도 복사된다. 한 sample당 정답 **하나**, 한 글자. 다중 정답/동의어 집합 없음.

### 2.4 Single-mask sample 정의

한 행 = 한 `sample_id` = 원문 한 위치의 한 글자를 `[MASK]`로 가린 것. 평가 단위는 **mask sample**이지 문장도 금석문도 아니다. 비문당 샘플 수는 1~최대 32(코드 상한), 관측 최대는 inscription당 303 샘플(여러 reading 합).

### 2.5 원문에 훼손 위치가 여러 개인 경우

인공 평가에서는 “여러 실제 훼손”이 아니라 **여러 eligible 한자 위치**다. 위치마다 독립 sample을 만들고, 각 sample의 입력에는 마스크가 **하나**다. 동시 다중 마스크 평가는 없다.

실제 결손 `□`가 여러 개인 원문은 `real_defect_inference.csv`에서 위치마다 `case_id`를 만들지만 gold가 없어 Exact@k 밖이다.

### 2.6 평가 단위

**mask (`sample_id`)**. era별 요약은 같은 sample을 `era`로 groupby한 뒤 같은 Exact@k를 다시 계산한다 (`run_evaluation()`). 비문 단위 macro-average는 없다.

### 2.7 작업 예

```text
prediction:
["字1", "字2", "字3", "字4", "字5"]

gold:
"字3"

Exact@1 = 0    # top-1 은 字1
Exact@5 = 1    # 字3 이 3등, 5등 안
```

`rank_of_gold` = 3 → `is_exact_at_1=False`, `is_exact_at_5=True`.

`generate()`가 JSON 한 글자만 내면 후보는 `["某"]`가 되고 Exact@5는 Exact@1과 같아져 **EXP-015와 공정 비교가 아니다.**

### 2.8 후보 정규화 (BERT)

길이 1이 아닌 토큰, `##` 서브워드, 중복 토큰은 후보에서 제외. ARI 출력도 평가 전에 **한 글자 한자만** 남기는 동일 규칙을 써야 한다 (DECISIONS [014] candidate normalization).

---

## 3. Dataset Split

| split | mask samples | inscriptions (in mask) | readings (in mask) | 파일 |
|---|---:|---:|---:|---|
| train | 46,139 | 2,355 | 4,822 | `artificial_mask_dataset.csv` `split=train` |
| valid | 10,679 | 530 | 1,075 | 동일, `split=valid` |
| test | 9,719 | 510 | 1,014 | 동일, `split=test` |
| 합계 | 66,537 | 3,395 | 6,911 | |

`splits.csv`: 6,084 inscription (train 4,258 / valid 912 / test 914). seed=**42**. `split_unit_type=inscription_id`.

생성: `assign_inscription_splits()` — training-text inscription 집합을 `pandas.Series.sample(frac=1.0, random_state=42)`로 셔플 후 70/15/15.

마스크 sample의 split은 `splits.csv`와 **0건 mismatch**.

### 3.1 문장 단위 random split인가

**아니다.** 문장/샘플 단위 랜덤이 아니라 **inscription 단위**다. 같은 금석문의 다른 mask sample이 train/valid로 갈라지지 **않도록** 설계되어 있고, `inscription_id`/`reading_id` 교집합은 0이다.

### 3.2 남은 위험

1. **동일 원문 문자열이 다른 inscription_id로 다른 split에 존재** (실측). 서로 다른 `inscription_id`인데 `original_text`가 완전 동일한 경우 26건. 그중 train∩valid 텍스트 16종, valid 샘플 52건이 train과 동일 원문을 공유. 대부분은 1~수 글자 명문(수복, 척화비 표준문안 등)이나, **장문 중복도 있다**:
   - `예조학 유허비` ↔ `완산이씨 열부비`, 405자, train/valid
   - `연천군심원사지부도군` ↔ `제월당대사비`, 933자, train/test
   - 척화비 계열 24자 표준문안이 train/valid에 걸쳐 19 inscription
2. title_key 붕괴 (DECISIONS [013]): 다른 실물 비문이 같은 `inscription_id`로 합쳐짐. split 교집합 0은 이 붕괴의 구조적 결과일 수 있어 “누수 없음”으로 해석하면 안 된다.
3. n-gram 문맥 overlap (EXP-009, test): 2-gram 이상 train 공유 **59.5%**. inscription split이 정형구 누수를 막지는 못한다. valid에 대한 동일 감사는 코드(`scripts/audit_ngram_leakage.py`)가 있으나 공식 리포트는 test 기준이다.
4. `splits.csv`에는 마스크가 없는 inscription 2,689개가 포함된다. 평가 N에는 안 들어가지만, 향후 retrieval corpus를 `splits.csv` train 전부로 만들면 마스크 없는 원문도 검색 대상이 된다.

---

## 4. Leakage Audit

### 4.0 현재 ChromaDB 상태

이 repository와 `docs/`, `src/`, `requirements.txt`, gitignore 추적 파일 어디에도 **ChromaDB / embedding index / vector store 구현이 없다.**  
chroma 관련 문자열 검색 결과 0건. retrieval corpus 빌더도 없다.

따라서 “현재 ChromaDB가 train만 쓰는지”는 **확인 대상이 존재하지 않는다.** 아래 1–10은 현재 산출물에 대한 판정과, **앞으로 인덱스를 만들 때**의 판정으로 나눈다.

공식 ARI BM25(`EvelynKimm/ARI`)는 `scripts/indexing_train_documents.py`가 **train dataset만** 인덱싱하고, `src/hanja_llm/retrieve_documents.py`가 query `data_id` 및 동일 문장(`are_sentences_equal`)을 제외한다. 우리 코드베이스에는 이 인덱스가 없다.

### 4.1 항목별 판정

| # | 항목 | 현재 상태 | 향후 인덱스를 잘못 만들 때 |
|---|---|---|---|
| 1 | Corpus 원천 | **UNKNOWN** — ChromaDB 없음. 원천 후보는 `readings.csv` / mask `original_text` / 외부 AJD·JRS | 미정 원천을 그대로 넣으면 평가 원문 혼입 가능 |
| 2 | train만 사용? | **UNKNOWN** (인덱스 없음) | valid/test `original_text`를 넣으면 **CRITICAL** |
| 3 | validation 원문 포함? | **UNKNOWN** (인덱스 없음) | 포함 시 **CRITICAL** — gold가 `original_text[mask_position]`에 그대로 있음 |
| 4 | test 원문 포함? | **UNKNOWN** | 포함 시 **CRITICAL** |
| 5 | 같은 비문의 다른 판독본 | **WARNING** — 한 inscription에 training reading이 평균 2.1개, 최대 22개. title_key 붕괴로 이물 비문 텍스트가 한 ID에 붙을 수 있음 | 같은 비문의 다른 판독이 train에만 있어도, valid query가 거의 동일한 문장을 검색할 수 있음 |
| 6 | 동일 문장 다른 ID 중복 | **WARNING** (실측 26개 완전 동일 `original_text`가 split을 넘음) | ID만 다르고 문자열이 같으면 BM25/임베딩 모두 정답 문장을 반환 가능 |
| 7 | gold가 들어 있는 원문 전체 노출 | 현재 BERT 평가 입력은 마스크 위치의 글자를 가림 (**SAFE** for MLM). retrieval이 풀 원문을 반환하면 **CRITICAL** | query가 `臣[D1]奉命`이고 retrieved가 `臣某奉命`이면 gold 직접 노출 |
| 8 | query가 검색 결과에 재등장 | 현재 retriever 없음 **UNKNOWN**. ARI 공식 코드는 exact sentence + `data_id` 제외 | 우리 데이터는 mask sample마다 `sample_id`가 다르고 원문은 공유되므로, **inscription_id / original_text 해시 제외가 없으면 CRITICAL** |
| 9 | duplicate / near-duplicate 제거 | 마스크 생성에 dedup 없음. ARI BM25는 `rapidfuzz.ratio` 0.8 | 미구현이면 척화비·수복 명문 등 표준문안이 반복 검색됨 **WARNING** |
| 10 | metadata로 정답 연결 | 마스크 행에 `label_char`가 있음. 인덱스가 이 컬럼을 저장하면 **CRITICAL**. `title`/`era`/`year`는 정답 글자 자체는 아님 **SAFE**(메타) / 단, 검색 문서에 `label_char`를 붙이면 누수 | |

### 4.2 사용자 예시와의 대응

```text
Validation 문장: 臣□奉命
retrieved document:  臣某奉命
```

BERT 현재 평가에서는 이런 retrieval이 없다.  
앞으로 retrieval을 넣을 때 구분:

- **독립 문헌 증거 (허용 후보):** train split의 **다른 inscription**이며, 문자열이 질의 원문과 동일/근사가 아니고, gold 자리가 쿼리와 다른 문맥에서 등장.
- **복제 누수 (CRITICAL):** valid/test와 동일 `original_text`, 동일 `inscription_id`의 다른 sample, 또는 위 실측 장문 중복(유허비 405자, 부도군/대사비 933자)처럼 다른 ID지만 같은 본문.

실측 장문 교차-split 중복은 “ChromaDB가 이미 누수”가 아니라 **인덱스를 만들면 바로 CRITICAL이 될 수 있는 데이터 특성**이다.

### 4.3 BERT 경로의 기존 누수 (retrieval 이전)

| 이슈 | 판정 |
|---|---|
| 평가 입력에 gold 미노출 (`validate_no_label_leak`) | **SAFE** |
| inscription split 준수 | **SAFE** (ID 기준) |
| 교차-split 완전 동일 원문 26건 | **WARNING** (BERT는 원문 검색을 하지 않으나 암기/정형구 효과) |
| test–train 2-gram overlap 59.5% | **WARNING** (EXP-009) |
| title_key 붕괴 | **WARNING** |

---

## 5. ARI Compatibility

| 차원 | 현재 BERT baseline | ARI-8B |
|---|---|---|
| 모델 | Encoder MLM `ddokbaro/SillokBert` | Decoder-only `DAMI-Lab/ARI-8B` ← `Qwen/Qwen3-8B` |
| 손상 표기 | `[MASK]` 1개 | `[D1]`, `[D2]`, … 다중 가능 |
| 입력 문맥 | ±64자 window, max 512 token | 문서 + 날짜 메타 + related docs, max 4096 |
| 출력 | 길이-1 한자 리스트 + softmax 점수 | JSON `{"[D1]": "某", ...}` |
| Top-K | mask 위치 logits `topk` | 논문: byte-level constrained decoding + 구성 토큰 로그확률 평균 (vLLM). **공개 repo에 추론 코드 없음** |
| Retrieval | 없음 | 논문/공식: BM25 top-20, fuzz 0.8, train corpus |
| 학습 도메인 | 한국 금석문 판독문 | 조선왕조실록·승정원일기 |
| 공개 가중치 | 로컬 `notebooks/best` | Hugging Face `DAMI-Lab/ARI-8B` (LM만. RAG 인덱스 미포함) |
| 의존성 | `transformers`, `torch` (`requirements.txt`) | 공식 예시는 `AutoModelForCausalLM` + chat template. 4-bit/vLLM/bitsandbytes는 **우리 requirements에 없음** |

ARI 공식 repo (`EvelynKimm/ARI`, main tree 전수 확인)에 있는 것:

- `src/hanja_llm/prompt.py` — SYSTEM/USER 프롬프트
- `src/hanja_llm/retrieve_documents.py` — BM25S + fuzz
- `scripts/indexing_train_documents.py` — train 인덱스
- `scripts/sfttrainer.py`, 데이터 전처리, 크롤러

**없는 것:** inference 스크립트, JSON 파서, constrained decoding, vLLM Top-K, 평가 Exact@k.

HF 카드 예시는 `model.generate(..., do_sample=False, max_new_tokens=128)` 한 번 → 문자열 디코드. 이는 Top-1 smoke용이다.

공정 비교를 위해 유지할 것: 동일 valid `sample_id`, 동일 `label_char`, 동일 `rank_of_gold` 정의, 후보를 한 글자 한자 리스트로 정규화.  
의도적으로 달라질 수 있는 것: 손상 마커, 프롬프트, 문맥 길이, 생성 방식. 문맥 길이는 §8에서 기본값을 ±64로 고정해 EXP-015와 맞추고, 장문맥은 ablation으로 분리한다.

---

## 6. Required Adapter

기존 gold/evaluator를 바꾸지 않고 ARI를 끼우는 경로:

```text
artificial_mask_dataset.csv  (valid rows, 기존 sample_id / label_char)
      ↓
ARIInputAdapter
      ↓
ARI prompt (no-RAG or +retriever)
      ↓
ARI-8B
      ↓
ARIOutputParser  →  ordered list[str]  (한 글자 한자)
      ↓
기존 compute_exact_metrics / PREDICTION_COLUMNS CSV
      ↓
기존 Exact@1/@5, paired_exact_compare vs EXP-015
```

### 6.1 `ARIInputAdapter`

책임:

- 한 마스크 행을 ARI 단일 손상 문서로 변환
- `build_masked_window(..., mask_token="[D1]")` 재사용 (기본 ±64)
- 다중 `[D2]`를 **만들지 않음** (기존 sample 정의 유지)
- `sample_id`, `label_char`, `era`는 그대로 통과
- ARI USER_PROMPT의 related documents / 날짜 필드는 실험 플래그로 채움 (019A는 related 빈 목록, 날짜는 비우거나 “unknown”)

우리 `year` 컬럼은 `1820년(순조 20)` 같은 문자열이고, 미상 2,066행이다. ARI의 `date, month, year, king` 슬롯에 그대로 넣으면 형식 불일치. 019A는 chronological grounding을 쓰지 않는 것이 기본.

### 6.2 Prompt 조립

ARI `SYSTEM_PROMPT` / `USER_PROMPT` (`EvelynKimm/ARI` `src/hanja_llm/prompt.py`)를 **복사해 새 파일에 둔다.** 기존 BERT 코드를 수정하지 않는다.  
Qwen3 chat template: `enable_thinking=False` (HF 카드 try/except와 동일).

### 6.3 `ARIOutputParser`

책임:

- 생성 문자열에서 JSON 객체 추출 (코드펜스/전후 잡담 허용)
- `[D1]` 값만 사용 (019A는 마커 1개)
- 값이 한 글자 한자가 아니면 `invalid`
- parse 실패 / 빈 값 / 비한자 → 빈 후보 리스트 (Exact miss, 실패율 별도 집계)
- Top-1 smoke: `[char]` 길이 1 리스트
- Top-5 본평가: constrained decoder가 준 순서 리스트를 그대로 `top5_chars`에 기록

### 6.4 기존 evaluator 연결

새 runner가 `run_fill_mask_eval.PREDICTION_COLUMNS`와 같은 CSV를 쓴다.

- `label_char` = 데이터셋 원본 (모델 출력이 아님)
- `top1_char` / `top5_chars` / `top10_chars`
- `rank_of_gold` / `is_exact_at_*` 는 `src.eval.metrics.rank_of_gold`로 계산
- `input_text` = ARI에 넣은 손상 문서 (추적용). gold와 별개

그 다음 `paired_exact_compare`는 EXP-015 예측 CSV와 `sample_id`로 join. evaluator 함수 본체는 수정하지 않는다.

### 6.5 Retriever 포트 (019B/C만)

최소 인터페이스 — 전면 아키텍처 리팩터 금지:

```python
def retrieve(query: str, k: int, *, exclude_inscription_id: str | None, exclude_text: str | None) -> list[str]:
    ...
```

019A는 이 함수를 호출하지 않거나 no-op을 넘긴다.

---

## 7. Top-K Strategy

정본은 `docs/ARI_TOPK_DESIGN.md`다. 점수·후보 집합·정렬·공개 코드 공백을 그 문서에서 고정한다.

### 7.1 임시 smoke test

```text
model.generate(do_sample=False, max_new_tokens=64)
  → JSON parse
  → Top-1 한 글자
```

목적: 로드, 프롬프트, 파싱, gold 비교, 저장. Exact@5는 이 단계의 공식 비교값이 아니다. 리포트에 `exact_at_5_not_comparable=true`를 명시.

### 7.2 본 평가 (EXP-015와 비교 가능한 Exact@5)

단순 `torch.topk(next_token_logits, 5)`는 **부적합**. 논문 Appendix C:

- Qwen byte-level BPE에서 한자는 자주 2–3 토큰으로 쪼개짐
- 첫 바이트 토큰 이후 유효 문자 시퀀스가 완성될 때까지 생성
- 구성 토큰 로그확률 **평균**으로 후보 점수
- 다음 `[D2]`는 기본 top-1에 조건. 우리는 단일 `[D1]`이므로 이 단계는 없음
- 논문 구현은 vLLM

공개 ARI repo에는 이 알고리즘의 코드가 **없다.** HF `generate()` 예시도 Top-1뿐이다.

따라서 본 평가 Exact@5는 **추가 구현이 필요하다.**

재구현 난이도: **높음**.

필요한 함수 (새 파일, 기존 predictors 미수정):

1. 한자 문자 집합 판정 (`is_cjk_ideograph` 재사용 가능)
2. prefix 토큰에서 완결 한자 문자열 탐지
3. 빔/트리: 첫 스텝 상위 n 확장 → 문자 완결까지 강제
4. 평균 logprob로 정렬 후 길이 1 한자 Top-5
5. JSON 경로로 생성 중이라면 `"{\"[D1]\": \""` 이후만 탐색 (포맷 제약)

테스트 전략 (구현 단계):

- 단일 토큰 한자 / 2–3 토큰 한자 fixture
- gold가 2토큰인 샘플에서 `topk(next_token)`과 constrained 결과가 달라짐을 단언
- 후보가 모두 길이 1 CJK인지
- 파서가 기존 `rank_of_gold`에 리스트를 넘길 수 있는지
- smoke n=20에서 parse 실패율

vLLM은 T4 16GB smoke의 기본값이 되기 어렵다 (오버헤드). 본 평가는 transformers 쪽 constrained 디코딩을 1차로 두고, vLLM은 별도 게이트.

Top-5를 못 만들면 EXP-015 Exact@5와 숫자를 나란히 두지 않는다. Exact@1만 비교한다고 명시한다.

---

## 8. EXP-019A

가칭: **ARI-8B, No Retrieval**, 기존 valid.

```text
기존 Validation (10,679)
  → BERT sample → [D1] 변환 (±64)
  → ARI-8B (관련 문서 없음)
  → 후보 추출
  → 기존 Exact@k
```

목적: retrieval 없이 ARI LM 자체가 EXP-015보다 나은지.

### 8.1 필요한 새 파일

기존 convention (`outputs/finetune/exp018/`, `reports/exp015_*`, `src/eval/run_fill_mask_eval.py`)에 맞춰 **exp019 전용 트리**:

```text
src/eval/ari/__init__.py
src/eval/ari/markers.py          # [MASK] → [D1]
src/eval/ari/prompts.py          # ARI SYSTEM/USER (공식 프롬프트 이식)
src/eval/ari/parse.py            # JSON / 한자 정규화
src/eval/ari/generate.py         # CausalLM generate (smoke)
src/eval/ari/topk.py             # constrained Top-K (본평가; smoke 이후)
src/eval/ari/run_ari_eval.py     # CLI runner
tests/test_ari_parse.py
tests/test_ari_markers.py
notebooks/exp019_colab.ipynb     # 구현 단계에서만
reports/colab_exp019_runbook.md  # 구현 단계에서만
```

결과 (기존과 섞지 않음):

```text
outputs/predictions/exp019a_ari8b_norag_valid_predictions.csv
outputs/eval/exp019a/
reports/evaluation/exp019a_ari8b_valid_summary.csv
reports/exp019/exp019a_results.md
reports/exp019/exp019a_run_meta.json
```

### 8.2 수정이 필요한 기존 파일

019A 최소 경로에서는 **기존 학습/평가 로직을 수정하지 않는다.**

구현 착수 시에만 허용되는 부가 수정:

- `requirements.txt` — `bitsandbytes`, `accelerate` (4-bit 선택 시). 기존 패키지 버전 하한을 함부로 올리지 않음
- `docs/EXPERIMENT_LOG.md` — 실행 **후** EXP-019 기록 (지금은 금지)
- `src/data/finetune_sillokbert.py`의 `assert_safe_output_dir` — 재사용만. EXP-015/018 경로를 쓰는 출력을 거절하도록 **호출**하지, 기본 학습 동작을 바꾸지 않음

명시적으로 건드리지 말 것:

- `outputs/predictions/exp015_*`
- `reports/evaluation/exp015_*`
- `reports/exp015_results.*`
- `notebooks/best/`
- `src/eval/run_fill_mask_eval.py`의 model-kind / 지표 정의
- EXP-018 체크포인트 디렉터리

### 8.3 재사용 가능한 기존 utility

- `src.eval.mask_window.build_masked_window`
- `src.eval.metrics.compute_exact_metrics`, `rank_of_gold`
- `src.eval.run_fill_mask_eval._select_rows`, `PREDICTION_COLUMNS`, `_format_scores`
- `src.eval.paired_exact_compare`
- `src.eval.stratified_analysis` (era/빈도, 본평가 후)
- `src.data.finetune_sillokbert.get_git_commit`, `_file_sha256`, `assert_safe_output_dir`
- `src.nrich.text_utils.is_cjk_ideograph`

### 8.4 Hugging Face 로딩 위치

새 `src/eval/ari/generate.py`에서만:

```text
AutoTokenizer.from_pretrained("DAMI-Lab/ARI-8B")
AutoModelForCausalLM.from_pretrained("DAMI-Lab/ARI-8B", ...)
```

기존 `FillMaskPredictor` / `AutoModelForMaskedLM`에 넣지 않는다.

### 8.5 4-bit inference 적용 위치

같은 로더에서 `BitsAndBytesConfig(load_in_4bit=True)`를 옵션 플래그로. 기본 경로를 4-bit로 강제하지 말 것. GPU 미실행 상태이므로 T4 16GB 가능 여부를 **확정하지 않음**.

### 8.6 Batch inference

- Smoke `generate()`: padding + `attention_mask`로 배치는 가능하나 JSON 길이 가변, 이득이 작음. smoke는 batch=1이 안전.
- Constrained Top-K: 샘플당 트리 탐색이라 1차 구현은 batch=1.
- BERT `predict_batch_scored`를 ARI에 억지로 맞추지 않음.

### 8.7 결과 저장 형식

기존 prediction CSV 스키마 (DATA_SCHEMA §9) + `run_meta.json`을 `reports/exp019/`에 둔다. `experiment_id=EXP-019A`. `notes`에 `retriever=none`, `topk_method=generate|constrained`, `context_window=64`.

era별 Exact@k는 `run_evaluation`과 같이 `era` groupby로 저장 가능. EXP-015 summary에 이미 조선/고려 행이 있다.

---

## 9. EXP-019B

ARI-8B + **BM25** (공식 ARI에 가깝게). 019A와 동일 valid, 동일 모델, 동일 생성 파라미터.

Retriever만 교체:

- 인덱스 원천: **train split `original_text` unique**, `inscription_id` 단위로 문서화
- 쿼리: damage marker 제거 텍스트 (`retrieve_documents.remove_damage_patterns`와 동일 취지)
- 제외: 동일 `inscription_id`, 동일 정규화 문자열, 가능하면 fuzz≥0.8 near-dup
- k: 논문 기본 20. 금석문 단문은 k를 낮출 수 있으나 비교 시 019C와 동일 k
- 구현: `bm25s` + Qwen tokenizer 토큰화는 공식 스크립트를 새 파일로 이식. 기존 BERT 코드 미수정

공개 ARI는 AJD/JRS train을 검색한다. 우리 019B는 **우리 금석문 train**을 검색한다. 논문 수치와의 재현이 아니라 EXP-015 대비 공정성용이다.

---

## 10. EXP-019C

ARI-8B + 현재 논의된 embedding/Chroma 계열.

**현재 ChromaDB가 없으므로 019C는 새 인덱스를 만드는 실험이다.** 기존 인덱스를 “이미 있다”고 가정하지 않는다.

동일 조건: 019B와 **같은 문서 집합, 같은 k, 같은 exclusion, 같은 query 문자열**. 검색 함수 본문만 dense retrieval.

공정성에서 어려운 점:

- BM25는 문자 표면, Chroma는 임베딩 모델/청크 크기/유사도 척도가 추가 자유도
- 공식 ARI도 임베딩(Gemini Embedding) vs BM25를 같은 corpus에서 비교했고 BM25가 더 나았다
- 우리 쪽에 임베딩 모델 선정이 아직 없다
- 짧은 명문(1~4자)은 dense retrieval이 붕괴하기 쉬움

Go 조건: 019A가 파싱·평가 연결을 통과하고, train-only corpus + exclusion 규칙이 코드/테스트로 고정된 다음.

---

## 11. Files to Add

구현 단계 예상 (지금은 생성하지 않음):

| 경로 | 역할 |
|---|---|
| `src/eval/ari/markers.py` | window + `[D1]` |
| `src/eval/ari/prompts.py` | ARI 프롬프트 |
| `src/eval/ari/parse.py` | JSON/한자 파서 |
| `src/eval/ari/generate.py` | 4-bit 옵션 CausalLM 로드·generate |
| `src/eval/ari/topk.py` | constrained Top-5 |
| `src/eval/ari/retrieve.py` | `retrieve()` 프로토콜 + BM25/Chroma 구현 |
| `src/eval/ari/run_ari_eval.py` | EXP-019 runner |
| `tests/test_ari_parse.py` | 파서 |
| `tests/test_ari_markers.py` | 마커/윈도우 |
| `tests/test_ari_retrieve_exclusion.py` | valid 원문 미인덱싱·self-hit 제외 |
| `outputs/eval/exp019a/` | 런 산출 (gitignore `outputs/`) |
| `reports/exp019/` | 리포트·run_meta |
| `reports/evaluation/exp019a_*_summary.csv` | 커밋용 요약 |

---

## 12. Files to Modify

019A 착수 시에도 **기본은 수정 0**.

| 파일 | 왜 필요한가 | 지금 |
|---|---|---|
| `requirements.txt` | 4-bit/BM25 패키지 | 구현 단계, 선택 |
| `docs/EXPERIMENT_LOG.md` | 실행 후 기록 규칙 | 실행 전 수정 금지 |
| `docs/DECISIONS.md` | Top-K 방법·문맥 길이 확정 시 | 팀 결정 후 |
| `.gitignore` | 모델 캐시/인덱스가 생기면 | 필요 시 |

수정하면 안 되는 파일:

- `src/eval/metrics.py` (지표 정의)
- `src/eval/run_fill_mask_eval.py` (EXP-015 재현 경로)
- `src/data/build_artificial_mask.py`, `split_by_inscription.py`
- `reports/exp015_*`, `reports/evaluation/exp015_*`
- EXP-018 학습 산출물

`run_fill_mask_eval.py`에 `model-kind=ari`를 끼워 넣는 것은 **비권장**. MLM 가정과 섞인다. 별도 CLI가 최소 수정이다.

---

## 13. Smoke Test Plan

전체 valid를 바로 돌리지 않는다. n=**32** (가능하면 20–100, seed=42, `_select_rows`와 동일 `df.sample`).

```text
model load
  → prompt
  → generate (Top-1)
  → JSON parse
  → 한자 추출
  → gold 비교
  → CSV + run_meta 저장
```

기록 필드:

```text
experiment_id: EXP-019A-SMOKE
git commit hash
model id: DAMI-Lab/ARI-8B
model revision: (HF sha, 로드 시 기록)
transformers version
torch version
quantization: 4bit | fp16 | unset
GPU: (있으면 이름; 없으면 cpu)
VRAM peak: (nvidia-smi 또는 미측정)
model load success: bool
sample count
total inference time
sec/sample
JSON parse failure: count / rate
empty prediction
invalid prediction (비한자, 다중 글자)
Exact@1
Exact@5: 기입하되 comparable=false (Top-1 only)
dataset split: valid subsample seed=42
evaluation script: src.eval.ari.run_ari_eval
timestamp
```

실패해도 EXP-015 파일을 덮어쓰지 않는다. smoke Exact@k는 공식 비교에 쓰지 않음 (EXP-018 smoke와 동일 원칙).

---

## 14. Full Evaluation Plan

1. Smoke 통과 (로드, parse 실패율 임계, 출력 스키마)
2. Constrained Top-5 단위 테스트 통과 전에는 **full valid Exact@5를 EXP-015 옆에 두지 않음**
3. 본평가 데이터: `artificial_mask_dataset.csv` **valid 전체 10,679**, 동일 `sample_id`
4. 문맥: 1차 **±64** (EXP-015와 동일). 장문맥은 `EXP-019A-CW` ablation
5. 지표: `compute_exact_metrics` Exact@1/5/10
6. 비교: `paired_exact_compare` vs `outputs/predictions/exp015_sillok_ft_valid_predictions.csv` (gitignore — 로컬 사본 필요). 커밋된 기준선은 `reports/evaluation/exp015_sillokbert_valid_summary.csv`
7. 층화: `stratified_analysis` + era (최소 조선/고려). EXP-015 valid 조선 n=8,793 Exact@5 54.70%, 고려 n=642 Exact@5 49.53%
8. test: 사용하지 않음 (DECISIONS [014]/[012])
9. 019B/C는 019A 동일 생성 설정 freeze 후 retriever만 변경
10. 채택 게이트는 EXP-018 FT 게이트(0.7pp subsample)를 자동 적용하지 않음. 모델 계열이 다르므로 별도 결정. 최소 보고: overall Exact@5 Δ, McNemar p, 조선/고려, parse 실패율, runtime

---

## 15. Risks

| 위험 | 내용 | 심각도 |
|---|---|---|
| data leakage | 현재 Chroma 없음. 인덱스에 valid/test 원문 또는 교차-split 중복 장문을 넣으면 gold 노출 | 019A: 낮음 / 019B·C: 높음 |
| GPU OOM | 8B + 4096 + RAG 20문서는 T4 16GB에서 쉽게 초과. 4-bit·±64·RAG 없음으로 완화. **미실측** | 높음 (미확인) |
| tokenizer | 한자 다토큰. next-token top5 ≠ 한자 top5 | 본평가 Exact@5 blocker |
| invalid Hanja output | JSON 값에 한글/ASCII/다중 글자 | smoke에서 계측 |
| JSON parse failure | decoder가 포맷 이탈. 실패는 miss로 처리하고 율을 보고 | 중간 |
| Top-K incompatibility | generate-only Exact@5는 EXP-015와 비비교 | 공식 비교 blocker |
| retrieval contamination | 동일 원문 다른 ID, 척화비 표준문안, title_key 붕괴 | 019B/C |
| 시대 domain shift | ARI는 조선 실록/일기. valid는 조선 8,793 / 고려 642 / 일제강점기 875 등. 고려·전근대 비조선에서 하락 가능 | 중간–높음 |
| 문맥 길이 mismatch | ARI 학습은 장문+다중손상, 우리는 ±64 단일 [D1] | 중간 (공정성 vs 적성) |
| 연대 메타 공란 | ARI 프롬프트가 king/date를 기대. 금석문 `year`는 비표준·미상 다수 | 019A는 비사용으로 회피 |
| EXP-015 예측 CSV 부재 | gitignore. McNemar 불가 시 summary만으로는 paired 불가 | 중간 |
| 가중치 다운로드 | 이번 단계 금지. 구현 시 용량·라이선스(Apache 2.0) 확인 | 절차 |

---

## 16. Go / No-Go Criteria

구현 단계로 넘어가기 **전에** 충족해야 할 것:

1. 비교 대상이 EXP-015 valid n=10,679, Exact@5 주력임이 팀에서 재확인됨 (본 문서·DECISIONS [014]).
2. ARI는 별도 디렉터리에만 쓰며 EXP-015/018 산출물을 읽기만 한다.
3. 019A는 retrieval 없음 (인덱스를 먼저 만들지 않음).
4. 평가 출력은 기존 prediction CSV + `compute_exact_metrics`를 쓴다. metrics.py를 바꾸지 않는다.
5. Smoke는 Top-1, 본평가 Exact@5는 constrained decoding 계획이 있다. 공개 코드만으로 Top-5가 안 됨을 인정한다.
6. Retrieval corpus 정책이 문서화됨: train-only, 동일 inscription/원문 제외, 교차-split 장문 중복 리스트를 제외 후보로 둔다.
7. 4-bit/T4는 “시도 가능한 코드 구조”이지 성공 보장이 아니다.
8. 이번 inspection에서 코드를 수정하지 않았고, 모델 다운로드를 하지 않았다.

구현 착수 후 full valid 비교 전:

1. n≤100 smoke에서 load success, parse 실패율 허용선 (권장: <10%, 팀 확정).
2. Top-5 경로가 한 글자 한자 5개를 안정적으로 내거나, Exact@1-only로 비교 범위를 축소하는 명시적 결정.
3. GPU 실측 VRAM/OOM 여부 기록.

---

## Q1–Q7

### Q1

ARI-8B를 **기존 Validation과 evaluator를 변경하지 않고** 평가할 수 있는가?

**가능하다.** valid 행의 `sample_id`/`label_char`를 그대로 두고, ARI 예측을 기존 `top_chars` 리스트로 변환한 뒤 `rank_of_gold` / `compute_exact_metrics`에 넣으면 된다. 데이터 CSV와 `src/eval/metrics.py`를 수정할 필요가 없다.

### Q2

가능하다면 최소 수정 경로는 무엇인가?

**새 `src/eval/ari/` runner만 추가**한다. `build_masked_window(..., "[D1]")` → ARI 프롬프트 → 생성/파싱 → 기존 prediction 스키마 CSV → 기존 metrics / `paired_exact_compare`. `run_fill_mask_eval.py`에 모델 종류를 추가하지 않는다. EXP-015 산출물은 읽기 전용.

### Q3

현재 ChromaDB에 data leakage 위험이 있는가?

**현재 ChromaDB는 없다 (UNKNOWN / 해당 없음).**  
지금부터 인덱스를 만들 때 valid/test 원문, 동일 `original_text`의 다른 ID, 마스크 행의 `label_char` 메타를 넣으면 **CRITICAL**이 된다. 이미 확인된 train/valid 장문 중복(405자 유허비 쌍 등)은 제외 규칙이 필요하다. BERT 평가 자체는 입력에서 gold를 가리므로 SAFE에 가깝다.

### Q4

ARI Top-5 후보 생성은 현재 공개 코드만으로 가능한가, 아니면 추가 구현이 필요한가?

**추가 구현이 필요하다.** `EvelynKimm/ARI`에 constrained decoding / vLLM Top-K / 추론 스크립트가 없다. HF 예시는 `generate()` Top-1. 논문 Appendix C 알고리즘을 우리가 구현하거나, Exact@5 비교를 보류해야 한다.

### Q5

EXP-019A(no-RAG)를 실행하기 전에 해결해야 할 blocker는 무엇인가?

1. 구현 착수 허가 (본 문서는 설계만).
2. `DAMI-Lab/ARI-8B` 다운로드·캐시 정책 (지금은 금지됨).
3. CausalLM 로더 + JSON 파서 + `[D1]` 어댑터 코드 (아직 없음).
4. Exact@5를 쓸 경우 constrained Top-K 구현 또는 “이번 라운드는 Exact@1만” 결정.
5. EXP-015 예측 CSV 로컬 존재 (paired 비교용). 없어도 summary 대비 overall 숫자는 가능.
6. GPU/4-bit 의존성. `requirements.txt`에 bitsandbytes 없음.
7. 출력 경로를 EXP-015/018과 분리.

### Q6

T4 16GB에서 4-bit ARI-8B smoke test를 시도할 수 있도록 현재 코드 구조를 만들 수 있는가?

**구조는 만들 수 있다** (별도 로더, `BitsAndBytesConfig` 옵션, batch=1, ±64, RAG 없음, `max_new_tokens` 제한).  
**실제로 T4 16GB에서 돌아가는지는 이번 단계에서 실행하지 않았으므로 확정하지 않는다.** 현 코드는 MLM 전용이라 그 구조는 아직 없다. vLLM 본평가 경로는 T4 smoke 기본값으로 두지 않는다.

### Q7

EXP-019 구현 난이도: **높음**

이유: (1) MLM logits Top-K와 decoder JSON 생성은 파이프라인이 다름. (2) 공정한 Exact@5에 필요한 constrained decoding이 공개 코드에 없고 재구현이 어렵다. (3) Qwen 한자 다토큰. (4) 8B VRAM/4-bit가 미검증. (5) 019B/C는 인덱스 누수 설계가 BERT보다 치명적. (6) 조선 실록 도메인 vs 금석문·고려 분포.  
019A smoke(Top-1, no-RAG)만 보면 중간 쪽으로 내려가나, EXP-015 Exact@5와 나란히 두는 목표 기준으로는 높음이 맞다.

---

*근거 코드: `src/eval/run_fill_mask_eval.py`, `predictors.py`, `metrics.py`, `mask_window.py`, `src/data/build_artificial_mask.py`, `split_by_inscription.py`, `finetune_sillokbert.py`, `src/preprocess/gsm_v4.py`, `docs/DECISIONS.md` [014][013][012], `docs/EXPERIMENT_LOG.md` EXP-015/018, `EvelynKimm/ARI` (prompt/retrieve/index only), `DAMI-Lab/ARI-8B` model card, ACL 2026 Findings Appendix C.*
