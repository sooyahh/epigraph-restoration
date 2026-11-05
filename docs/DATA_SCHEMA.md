# DATA_SCHEMA.md — 데이터 스키마 정의

이 문서는 프로젝트에서 사용할 데이터 파일의 구조와 컬럼 정의를 고정한다.
모든 수집, 전처리, 학습, 평가 코드는 이 문서의 스키마를 기준으로 작성한다.

---

## 1. 목적

본 프로젝트는 **한국 금석문 판독문에서 결손 또는 마스킹된 한자 위치에 대해 후보 한자를 예측하는 text-only MLM 기반 복원 시스템**을 만든다.

이를 위해 데이터는 다음 목적별 파일로 분리한다.

| 파일                            | 목적                           |
| ----------------------------- | ---------------------------- |
| `source_inventory.csv`        | 원천 데이터 목록과 권리/형식/상태 관리       |
| `readings.csv`                | 판독문 원문과 정규화 텍스트 관리           |
| `artificial_mask_dataset.csv` | 정량 평가와 학습용 인공 single-mask 샘플 |
| `real_defect_inference.csv`   | 실제 결손 `□` 위치에 대한 데모/추론용 데이터  |
| `splits.csv`                  | train/valid/test 분리 정보       |
| `model_predictions.csv`       | 모델 예측 결과 저장                  |
| `evaluation_summary.csv`      | 실험별 평가 지표 요약                 |

---

## 2. 전체 데이터 흐름

```text
원천 데이터
  ↓
source_inventory.csv
  ↓
readings.csv
  ↓
artificial_mask_dataset.csv ──→ 정량 평가 / 학습 / fine-tuning
  ↓
real_defect_inference.csv ────→ 실제 결손 데모 / 정성 분석
  ↓
model_predictions.csv
  ↓
evaluation_summary.csv
```

중요한 원칙은 다음과 같다.

```text
실제 결손 □는 정답이 없으므로 Exact@k 정량 평가에 사용하지 않는다.
Exact@k는 정답이 있는 artificial_mask_dataset.csv에서만 계산한다.
```

---

## 3. 공통 컬럼 명명 규칙

| 규칙                            | 설명                                       |
| ----------------------------- | ---------------------------------------- |
| 컬럼명은 영어 소문자와 `_` 사용           | 예: `source_record_id`, `text_normalized` |
| ID 컬럼은 `_id`로 끝냄              | 예: `inscription_id`, `sample_id`         |
| Boolean 컬럼은 `is_`, `has_`로 시작 | 예: `is_training_text`, `has_real_defect` |
| 원본 텍스트는 `raw` 포함              | 예: `text_raw`                            |
| 정규화 텍스트는 `normalized` 포함      | 예: `text_normalized`                     |
| 정답 라벨은 `label` 사용             | 예: `label_char`                          |
| 위치 정보는 0-based index 사용       | 첫 글자 위치는 `0`                             |

---

## 4. `source_inventory.csv`

### 4.1 목적

수집한 원천 데이터의 출처, 형식, 권리, 사용 상태를 관리한다.
데이터를 모델에 넣기 전에 어떤 원천을 사용할 수 있는지 판단하는 기준 파일이다.

### 4.2 권장 경로

```text
data/interim/source_inventory.csv
```

### 4.3 스키마

| 컬럼                       | 타입     | 필수 | 설명             | 예시                                  |
| ------------------------ | ------ | -: | -------------- | ----------------------------------- |
| `source_id`              | string |  Y | 원천별 내부 ID      | `nrich_api`                         |
| `source_name`            | string |  Y | 원천명            | `NRICH 금석문 API`                     |
| `source_type`            | string |  Y | 원천 형식          | `api`, `xls`, `xml`, `csv`          |
| `source_url`             | string |  N | 원천 URL 또는 설명   | `https://...`                       |
| `raw_path`               | string |  Y | raw 데이터 저장 위치  | `data/raw/nrich/api/page_0001.json` |
| `license_type`           | string |  N | 라이선스/이용 조건     | `공공누리`, `미확인`                       |
| `ai_use_allowed`         | string |  N | AI 학습 사용 가능 여부 | `yes`, `no`, `unknown`              |
| `redistribution_allowed` | string |  N | 재배포 가능 여부      | `yes`, `no`, `unknown`              |
| `collected_at`           | string |  Y | 수집일            | `2026-06-29`                        |
| `status`                 | string |  Y | 사용 상태          | `candidate`, `accepted`, `excluded` |
| `notes`                  | string |  N | 특이사항           | `XLS는 목록용으로만 사용`                    |

### 4.4 `status` 값

| 값           | 의미             |
| ----------- | -------------- |
| `candidate` | 사용 후보. 아직 검수 전 |
| `accepted`  | 사용하기로 결정       |
| `excluded`  | 사용 제외          |
| `unknown`   | 판단 보류          |

---

## 5. `readings.csv`

### 5.1 목적

원천 데이터에서 추출한 금석문 판독문을 관리한다.
학습/평가 데이터의 기본 단위가 되는 파일이다.

이 파일에는 **원본 텍스트와 정규화 텍스트를 모두 보존**한다.

### 5.2 권장 경로

```text
data/processed/readings.csv
```

단, 검수 전에는 다음처럼 후보임을 명시한다.

```text
data/interim/candidate_readings.csv
```

### 5.3 스키마

| 컬럼                   | 타입      | 필수 | 설명                         | 예시                         |
| -------------------- | ------- | -: | -------------------------- | -------------------------- |
| `reading_id`         | string  |  Y | 판독문 단위 내부 ID               | `nrich_000001`             |
| `source`             | string  |  Y | 데이터 출처                     | `nrich`, `nihc`            |
| `source_record_id`   | string  |  Y | 원천 데이터의 원래 ID              | `12345`                    |
| `inscription_id`     | string  |  Y | 비문/금석문 단위 ID. split 기준     | `insc_000001`              |
| `artifact_id`        | string  |  N | 유물 단위 ID. 존재 시 split 보조 기준 | `artifact_0001`            |
| `title`              | string  |  N | 금석문/유물명                    | `○○비`                      |
| `era`                | string  |  N | 시대                         | `고려`, `조선`, `삼국`           |
| `year`               | string  |  N | 연도 또는 추정 연대                | `1178`, `미상`               |
| `material`           | string  |  N | 재질                         | `석`, `금동`                  |
| `location`           | string  |  N | 소재지                        | `경기도`                      |
| `text_raw`           | string  |  Y | 원천에서 가져온 판독문 원문            | `大華嚴宗師...`                 |
| `text_normalized`    | string  |  Y | 정규화된 판독문                   | `大華嚴宗師...`                 |
| `text_length`        | integer |  Y | `text_normalized`의 글자 수    | `128`                      |
| `hanja_count`        | integer |  N | 한자 글자 수                    | `120`                      |
| `has_real_defect`    | boolean |  Y | 실제 결손 표기 포함 여부             | `true`                     |
| `defect_count`       | integer |  Y | 실제 결손 기호 수                 | `3`                        |
| `defect_markers_raw` | string  |  N | 원문에 등장한 결손 표기 목록           | `□;○;缺`                    |
| `is_training_text`   | boolean |  Y | 학습/평가용으로 사용 가능한지           | `true`                     |
| `exclude_reason`     | string  |  N | 제외 이유                      | `translation_mixed`        |
| `quality_flag`       | string  |  Y | 품질 상태                      | `ok`, `review`, `excluded` |
| `notes`              | string  |  N | 메모                         | `한글 해설문 일부 혼입 의심`          |

### 5.4 `quality_flag` 값

| 값          | 의미         |
| ---------- | ---------- |
| `ok`       | 바로 사용 가능   |
| `review`   | 수동 검토 필요   |
| `excluded` | 학습/평가에서 제외 |

### 5.5 `exclude_reason` 후보

| 값                   | 의미                |
| ------------------- | ----------------- |
| `too_short`         | 텍스트가 너무 짧음        |
| `translation_mixed` | 현대어 해석문 또는 설명문 혼입 |
| `too_many_defects`  | 결손 비율이 과도함        |
| `missing_id`        | split 기준 ID가 없음   |
| `not_hanja_text`    | 한자 판독문이 아님        |
| `duplicate`         | 중복 데이터            |
| `license_issue`     | 권리 문제             |
| `unknown`           | 기타 또는 미확인         |

---

## 6. `artificial_mask_dataset.csv`

### 6.1 목적

정답이 있는 판독문에서 한 글자를 인공적으로 `[MASK]` 처리하여 만든 학습/평가용 데이터다.

이 파일은 `Exact@1`, `Exact@5`, `Exact@10` 정량 평가의 기준이다.

### 6.2 권장 경로

```text
data/model_inputs/artificial_mask_dataset.csv
```

검수 전에는 다음 이름을 사용한다.

```text
data/model_inputs/candidate_artificial_mask_dataset.csv
```

### 6.3 스키마

| 컬럼                 | 타입      | 필수 | 설명                     | 예시                     |
| ------------------ | ------- | -: | ---------------------- | ---------------------- |
| `sample_id`        | string  |  Y | 인공 마스킹 샘플 ID           | `mask_00000001`        |
| `reading_id`       | string  |  Y | 원본 판독문 ID              | `nrich_000001`         |
| `source`           | string  |  Y | 데이터 출처                 | `nrich`                |
| `source_record_id` | string  |  Y | 원천 데이터 ID              | `12345`                |
| `inscription_id`   | string  |  Y | split 기준 비문 ID         | `insc_000001`          |
| `artifact_id`      | string  |  N | 유물 ID                  | `artifact_0001`        |
| `title`            | string  |  N | 금석문/유물명                | `○○비`                  |
| `era`              | string  |  N | 시대                     | `고려`                   |
| `original_text`    | string  |  Y | 정답이 포함된 원문             | `大華嚴宗師`                |
| `masked_input`     | string  |  Y | 한 글자가 `[MASK]`로 대체된 입력 | `大華嚴[MASK]師`           |
| `label_char`       | string  |  Y | 정답 한자                  | `宗`                    |
| `mask_position`    | integer |  Y | 마스크 위치. 0-based index  | `3`                    |
| `left_context`     | string  |  N | 마스크 왼쪽 문맥              | `大華嚴`                  |
| `right_context`    | string  |  N | 마스크 오른쪽 문맥             | `師`                    |
| `context_window`   | integer |  N | 좌우 문맥 길이 제한            | `64`                   |
| `mask_type`        | string  |  Y | 마스킹 방식                 | `single_char`          |
| `split`            | string  |  Y | train/valid/test       | `train`                |
| `created_at`       | string  |  Y | 생성일                    | `2026-06-29`           |
| `generation_rule`  | string  |  Y | 생성 규칙 설명               | `single_hanja_mask_v1` |
| `notes`            | string  |  N | 메모                     | `희귀 한자 포함`             |

### 6.4 필수 검증 규칙

다음 조건을 모두 만족해야 한다.

```text
1. masked_input에는 label_char가 해당 mask_position에 그대로 남아 있으면 안 된다.
2. label_char는 한 글자여야 한다.
3. mask_type은 1차 버전에서 single_char만 사용한다.
4. 같은 inscription_id는 train/valid/test 중 하나에만 속해야 한다.
5. split은 splits.csv와 일치해야 한다.
```

### 6.5 `split` 값

| 값          | 의미     |
| ---------- | ------ |
| `train`    | 학습용    |
| `valid`    | 검증용    |
| `test`     | 최종 평가용 |
| `excluded` | 제외     |

---

## 7. `real_defect_inference.csv`

### 7.1 목적

실제 금석문 판독문에 존재하는 결손 `□` 위치를 추론하기 위한 데이터다.

이 파일은 **정량 평가용이 아니다.**
정답이 없는 실제 결손 사례에 대해 모델 후보를 출력하고, 발표에서 정성 예시로 사용한다.

### 7.2 권장 경로

```text
data/model_inputs/real_defect_inference.csv
```

검수 전에는 다음 이름을 사용한다.

```text
data/model_inputs/candidate_real_defect_inference.csv
```

### 7.3 스키마

| 컬럼                  | 타입      | 필수 | 설명                     | 예시              |
| ------------------- | ------- | -: | ---------------------- | --------------- |
| `case_id`           | string  |  Y | 실제 결손 추론 사례 ID         | `real_000001`   |
| `reading_id`        | string  |  Y | 원본 판독문 ID              | `nrich_000001`  |
| `source`            | string  |  Y | 데이터 출처                 | `nrich`         |
| `source_record_id`  | string  |  Y | 원천 데이터 ID              | `12345`         |
| `inscription_id`    | string  |  Y | 비문/금석문 ID              | `insc_000001`   |
| `artifact_id`       | string  |  N | 유물 ID                  | `artifact_0001` |
| `title`             | string  |  N | 금석문/유물명                | `○○비`           |
| `era`               | string  |  N | 시대                     | `고려`            |
| `text_with_defect`  | string  |  Y | 실제 결손이 포함된 텍스트         | `大華嚴□師`         |
| `model_input`       | string  |  Y | 결손 위치를 `[MASK]`로 바꾼 입력 (`◇` 제거됨) | `大華嚴[MASK]師`    |
| `defect_marker_raw` | string  |  Y | 원문 결손 표기               | `□`             |
| `defect_position`   | integer |  Y | 결손 위치. 0-based index   | `3`             |
| `left_context`      | string  |  N | 왼쪽 문맥                  | `大華嚴`           |
| `right_context`     | string  |  N | 오른쪽 문맥                 | `師`             |
| `has_hangul_context`| boolean |  Y | 원문에 한글 치환 `◇` 포함 여부 | `false`          |
| `gold_label`        | string  |  N | 정답. 보통 비워둠             |                 |
| `usage`             | string  |  Y | 사용 목적                  | `demo_only`     |
| `notes`             | string  |  N | 메모                     | `정답 미확인`        |

### 7.4 사용 제한

다음 원칙을 반드시 지킨다.

```text
real_defect_inference.csv는 gold_label이 없으므로 Exact@k 계산에 사용하지 않는다.
이 파일은 실제 결손에 대한 후보 제시, 정성 분석, 발표 데모에만 사용한다.
`model_input`·`left_context`·`right_context`는 BERT 호환을 위해 `◇`를 제거한다.
`text_with_defect`는 원문(◇ 포함)을 보존하고, `has_hangul_context=true`인 케이스는 데모에서 기본 제외한다.
```

### 7.5 `usage` 값

| 값                      | 의미         |
| ---------------------- | ---------- |
| `demo_only`            | 발표/데모용     |
| `qualitative_analysis` | 정성 분석용     |
| `expert_review`        | 향후 전문가 검토용 |
| `excluded`             | 사용 제외      |

---

## 8. `splits.csv`

### 8.1 목적

train, valid, test 분리 결과를 기록한다.
데이터 누수를 막기 위해 split 기준을 명확히 저장한다.

### 8.2 권장 경로

```text
data/model_inputs/splits.csv
```

### 8.3 스키마

| 컬럼                 | 타입      | 필수 | 설명                   | 예시                              |
| ------------------ | ------- | -: | -------------------- | ------------------------------- |
| `split_unit_id`    | string  |  Y | split 기준 ID          | `insc_000001`                   |
| `split_unit_type`  | string  |  Y | split 기준 단위          | `inscription_id`                |
| `split`            | string  |  Y | train/valid/test     | `train`                         |
| `source`           | string  |  N | 출처                   | `nrich`                         |
| `era`              | string  |  N | 시대                   | `고려`                            |
| `num_readings`     | integer |  N | 해당 ID에 속한 판독문 수      | `1`                             |
| `num_mask_samples` | integer |  N | 해당 ID에서 생성된 마스크 샘플 수 | `32`                            |
| `seed`             | integer |  Y | split 생성 seed        | `42`                            |
| `split_rule`       | string  |  Y | split 생성 규칙          | `group_split_by_inscription_v1` |
| `created_at`       | string  |  Y | 생성일                  | `2026-06-29`                    |
| `notes`            | string  |  N | 메모                   | `시대 분포 수동 확인 필요`                |

### 8.4 split 원칙

```text
1. 같은 inscription_id는 train/valid/test 중 하나에만 들어간다.
2. 랜덤 글자 단위 split은 사용하지 않는다.
3. split 이후 시대, 길이, 결손 비율 분포를 점검한다.
4. test set은 최종 평가 전까지 반복적으로 보지 않는다.
```

---

## 9. `model_predictions.csv`

### 9.1 목적

모델이 출력한 후보 한자와 점수를 저장한다.
서비스 MVP는 top-1만 사용해도 되지만, 모델 평가와 발표를 위해 top-k 후보를 저장한다.

### 9.2 권장 경로

```text
outputs/predictions/model_predictions.csv
```

### 9.3 스키마

| 컬럼                | 타입      | 필수 | 설명                    | 예시                               |
| ----------------- | ------- | -: | --------------------- | -------------------------------- |
| `prediction_id`   | string  |  Y | 예측 결과 ID              | `pred_000001`                    |
| `experiment_id`   | string  |  Y | 실험 ID                 | `EXP-005`                        |
| `model_name`      | string  |  Y | 모델명                   | `ddokbaro/SillokBert`            |
| `model_version`   | string  |  N | 체크포인트/버전 추적 태그       | `exp012_sillokbert_gpu_best_valid_loss` |
| `sample_id`       | string  |  N | 인공 마스킹 샘플 ID          | `mask_00000001`                  |
| `case_id`         | string  |  N | 실제 결손 사례 ID           | `real_000001`                    |
| `input_text`      | string  |  Y | 모델 입력                 | `大華嚴[MASK]師`                     |
| `label_char`      | string  |  N | 정답. 실제 결손은 비어 있을 수 있음 | `宗`                              |
| `top1_char`       | string  |  Y | top-1 후보              | `宗`                              |
| `top1_score`      | float   |  Y | top-1 **상대 선호 점수**(보정된 신뢰 확률 아님) | `0.421`                          |
| `top5_chars`      | string  |  N | top-5 후보 목록           | `宗;禪;法;僧;師`                      |
| `top5_scores`     | string  |  N | top-5 상대 선호 점수 목록     | `0.421;0.183;0.110;0.070;0.040`  |
| `top10_chars`     | string  |  N | top-10 후보 목록          | `宗;禪;法;僧;師;...`                  |
| `top10_scores`    | string  |  N | top-10 상대 선호 점수 목록    | `0.421;0.183;...`                |
| `rank_of_gold`    | integer |  N | 정답의 순위. 없으면 null      | `1`                              |
| `is_exact_at_1`   | boolean |  N | Exact@1 여부            | `true`                           |
| `is_exact_at_5`   | boolean |  N | Exact@5 여부            | `true`                           |
| `is_exact_at_10`  | boolean |  N | Exact@10 여부           | `true`                           |
| `prediction_type` | string  |  Y | 예측 대상 종류              | `artificial_mask`, `real_defect` |
| `created_at`      | string  |  Y | 생성일                   | `2026-06-29`                     |
| `notes`           | string  |  N | 메모                    | `실제 결손은 정답 없음`                   |

> `top*_score`는 vocab softmax에서 온 **후보 간 상대적 선호 점수**다. 전문가·발표 UI에서 “이 후보가 정답일 확률/신뢰 확률”로 표현하지 않는다.

### 9.4 `prediction_type` 값

| 값                 | 의미                |
| ----------------- | ----------------- |
| `artificial_mask` | 정답이 있는 인공 마스킹 데이터 |
| `real_defect`     | 정답이 없는 실제 결손 데이터  |
| `demo`            | 발표용 수동 예시         |

---

## 10. `evaluation_summary.csv`

### 10.1 목적

실험별 정량 평가 결과를 요약한다.
발표자료와 `EXPERIMENT_LOG.md`에 들어갈 수치를 이 파일에서 가져온다.

### 10.2 권장 경로

```text
reports/evaluation/{experiment_id}_{model}_{split}_summary.csv   # 버전 관리 (커밋)
outputs/evaluation/evaluation_summary.csv                        # 로컬 전체 로그 (gitignore)
```

예: `reports/evaluation/exp003_freq_test_summary.csv`, `reports/evaluation/baseline_test_summary.csv`

### 10.3 스키마

| 컬럼                | 타입      | 필수 | 설명       | 예시                           |
| ----------------- | ------- | -: | -------- | ---------------------------- |
| `experiment_id`   | string  |  Y | 실험 ID    | `EXP-005`                    |
| `experiment_name` | string  |  Y | 실험명      | `SillokBert zero-shot`       |
| `model_name`      | string  |  Y | 모델명      | `ddokbaro/SillokBert`        |
| `dataset_name`    | string  |  Y | 사용 데이터셋  | `artificial_mask_dataset_v1` |
| `split`           | string  |  Y | 평가 split | `test`                       |
| `num_samples`     | integer |  Y | 평가 샘플 수  | `1000`                       |
| `exact_at_1`      | float   |  Y | Exact@1  | `0.214`                      |
| `exact_at_5`      | float   |  Y | Exact@5  | `0.438`                      |
| `exact_at_10`     | float   |  Y | Exact@10 | `0.512`                      |
| `group_at_1`      | float   |  N | Group@1  |                              |
| `group_at_5`      | float   |  N | Group@5  |                              |
| `group_at_10`     | float   |  N | Group@10 |                              |
| `created_at`      | string  |  Y | 평가일      | `2026-06-29`                 |
| `notes`           | string  |  N | 메모       | `zero-shot, no fine-tuning`  |

### 10.4 주의

```text
evaluation_summary.csv에는 artificial_mask_dataset.csv 기반 정량 평가 결과만 기록한다.
real_defect_inference.csv 기반 결과는 정성 분석으로만 기록한다.
```

---

## 11. 필수 ID 정의

### 11.1 `source_record_id`

원천 데이터에서 제공하는 원래 ID다.
NRICH, 국사편찬위 등 출처마다 형식이 다를 수 있다.

### 11.2 `inscription_id`

비문 또는 금석문 단위 ID다.
train/valid/test split의 1차 기준이다.

### 11.3 `artifact_id`

유물 단위 ID다.
하나의 유물에 여러 판독문이 연결될 경우 split 보조 기준으로 사용한다.

### 11.4 `reading_id`

전처리된 판독문 단위 ID다.
하나의 `inscription_id`에서 여러 `reading_id`가 나올 수 있다.

### 11.5 `sample_id`

인공 마스킹 데이터의 샘플 ID다.

### 11.6 `case_id`

실제 결손 추론 사례 ID다.

---

## 12. 파일 간 관계

```text
source_inventory.csv
    source_id
        ↓
readings.csv
    reading_id
    inscription_id
    artifact_id
        ↓
artificial_mask_dataset.csv
    sample_id
    reading_id
    inscription_id
        ↓
splits.csv
    split_unit_id = inscription_id 또는 artifact_id
        ↓
model_predictions.csv
    sample_id 또는 case_id
        ↓
evaluation_summary.csv
    experiment_id
```

---

## 13. 데이터 누수 방지 규칙

다음 조건을 반드시 만족해야 한다.

```text
1. 같은 inscription_id가 train과 test에 동시에 등장하면 안 된다.
2. 같은 artifact_id가 train과 test에 동시에 등장하면 안 된다.
3. label_char가 masked_input의 마스크 위치에 그대로 남아 있으면 안 된다.
4. 실제 결손 □를 정답이 있는 평가셋처럼 사용하면 안 된다.
5. test set을 보면서 하이퍼파라미터를 반복 조정하면 안 된다.
```

---

## 14. 정량 평가 가능 여부

| 데이터 파일                        |     정답 라벨 |   정량 평가 가능 여부 | 사용 지표                     |
| ----------------------------- | --------: | ------------: | ------------------------- |
| `artificial_mask_dataset.csv` |        있음 |            가능 | `Exact@1/5/10`, `Group@k` |
| `real_defect_inference.csv`   |        없음 |            불가 | 정성 사례                     |
| `readings.csv`                |    원문은 있음 |     직접 평가는 아님 | 인공 마스킹 생성 원천              |
| `model_predictions.csv`       | 경우에 따라 다름 | 인공 마스킹일 때만 가능 | `Exact@k`                 |

---

## 15. 데이터 버전 관리

검수 전 데이터와 검수 완료 데이터를 구분한다.

| 상태    | 파일명 예시                       | 의미          |
| ----- | ---------------------------- | ----------- |
| 후보    | `candidate_readings.csv`     | 검수 전        |
| 검수 완료 | `readings_v1.csv`            | 1차 사용 가능    |
| 수정 버전 | `readings_v2.csv`            | 규칙 수정 후 재생성 |
| 폐기    | `deprecated_readings_v0.csv` | 사용 중단       |

실험에는 반드시 데이터 버전을 기록한다.

예:

```text
EXP-005
dataset_name: artificial_mask_dataset_v1
split_rule: group_split_by_inscription_v1
```

---

## 16. 최소 사용 가능 데이터셋 기준

모델 실험을 시작하려면 최소한 다음 조건을 만족해야 한다.

```text
1. readings.csv에 text_normalized가 존재한다.
2. inscription_id 또는 artifact_id가 존재한다.
3. artificial_mask_dataset.csv에 masked_input, label_char가 존재한다.
4. splits.csv에 train/valid/test 정보가 존재한다.
5. 같은 inscription_id가 여러 split에 섞이지 않는다.
6. Exact@k 계산에 필요한 label_char가 존재한다.
```

이 조건이 충족되기 전에는 모델 fine-tuning을 진행하지 않는다.

---

## 17. 현재 우선순위

현재 단계에서 먼저 확정해야 할 것은 다음이다.

```text
1. NRICH API/XLS에서 source_record_id, title, era, reading text 필드 확인
   → md_data1~22 XML↔명세 불일치 매핑: §18 (2건 XML 교차 검증 완료)
2. 국사편찬위 XML에서 ID, 제목, 시대, 판독문 필드 확인
3. inscription_id로 쓸 수 있는 실제 식별자 확인
4. 판독문과 해석문/주석이 분리되는지 확인
5. 실제 결손 표기 목록 확인
6. readings.csv 후보 생성
7. artificial_mask_dataset.csv 생성
8. splits.csv 생성
```

---

## 18. NRICH OpenAPI `md_data` 필드 매핑

### 18.1 배경

국립문화유산연구원 금석문 검색 OpenAPI(공공데이터포털 [15015633](https://www.data.go.kr/data/15015633/openapi.do))의 **공식 명세서 컬럼 순서**와 **실제 XML 응답의 `md_data1`~`md_data22` 태그 번호**가 일치하지 않는다.

단순히 `+1` 칸 밀림(off-by-one)이 아니라, **태그 번호와 의미가 전면 재배치**되어 있다. 따라서 파서는 명세서의 `md_dataN` 라벨을 그대로 믿으면 안 되고, 아래 **확인된 매핑 테이블**로 의미를 해석해야 한다.

검증 샘플:

| # | 금석문 | `KSM_IDX` (`md_data2`) | `ROW_NUM` (`md_data1`) | 확인일 |
| --- | --- | --- | --- | --- |
| 1 | 이계신도비 | `3` | `1` | 2026-06-29 |
| 2 | 이소한신도비 | `5` | `2` | 2026-06-29 |

> `md_data1`(목록 순번)과 `md_data2`(`KSM_IDX`)는 **서로 다른 값**이다. 샘플 2에서 `ROW_NUM=2`, `KSM_IDX=5`로 재확인됨.

> `md_data1`~`md_data22`는 **메타데이터 목록 응답** 필드다. 판독문·해석문 전문은 이 범위에 없으며, 상세 조회 API·상세 페이지·XLS(일부 생략) 등 별도 경로에서 수집한다.

### 18.2 불일치 유형 요약

| 유형 | 설명 | 대응 |
| --- | --- | --- |
| 태그 번호 ≠ 명세 순서 | `md_data4`가 명세상 `KSM_AGE`가 아니라 `KSM_CHI_NAME` | §18.3 매핑 테이블 사용 |
| 명세 라벨 오독 | `md_data1`을 `DATA_LINK`로 읽으면 `1`(순번)이 URL로 오인됨 | `ROW_NUM`으로 해석 |
| 코드·명칭 분리 | 시대·유형·지정종목·공공누리는 코드/명칭이 다른 `md_data`에 분산 | 코드·명칭 쌍을 함께 저장 |
| 학습 텍스트 부재 | 개관(`KSM_SUVEY`)만 포함, 판독문 없음 | `readings.csv`의 `text_raw`는 별도 수집 |

### 18.3 확인된 매핑 테이블 (역추적 결과)

아래 **「실제 의미」** 열이 XML 파싱 시 사용할 정본이다. **「명세서 기준」** 열은 공공데이터포털 API 명세의 원래 정의이며, 현재 XML과는 태그 번호가 맞지 않는다.

| XML 태그 | 명세서 기준 컬럼 | 명세서 한글명 | **실제 의미 (확인됨)** | **실제 필드 코드** | 샘플 값 (이계신도비) | `readings.csv` / 내부 스키마 매핑 | 검증 근거 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `md_data1` | `DATA_LINK` | 데이터 링크 | **목록 순번** | `ROW_NUM` | `1` | (목록용, split ID 아님) | 정수 순번 형태; URL 아님 |
| `md_data2` | `FORMAL_CD` | 지정종목코드 | **자료식별번호** | `KSM_IDX` | `3` | `source_record_id`, `inscription_id` 후보 | 상세 URL `ksm_idx=3`과 일치 |
| `md_data3` | `FORMAL_CDNM` | 지정종목코드 내용 | **금석문명(한글)** | `KSM_KOR_NAME` | `이계신도비` | `title` | 한글 명칭 형태 |
| `md_data4` | `KSM_AGE` | 연대 | **금석문명(한문)** | `KSM_CHI_NAME` | `李𡹘神道碑` | `title` 보조 | 한문 명칭·PUA 포함 |
| `md_data5` | `KSM_CHI_NAME` | 금석문명_한문 | **시대 코드** | `KSM_EAR` | `339` | `era` 코드 보조 | `md_data6=조선`과 코드·명칭 쌍 |
| `md_data6` | `KSM_COPYRIGHT` | 저작권자 | **시대 코드명** | `KSM_EAR_NM` | `조선` | `era` | 시대명 텍스트 |
| `md_data7` | `KSM_EAR` | 시대 코드 | **연대** | `KSM_AGE` | `1629년(인조 7)` | `year` | 연도·연호 표기 |
| `md_data8` | `KSM_EAR_NM` | 시대 코드명칭 | **유형 코드** | `KSM_TYPE` | `2844` | (메타) | `md_data9=비`와 코드·명칭 쌍 |
| `md_data9` | `KSM_FONT` | 서체 | **유형 코드명** | `KSM_TYPENM` | `비` | (메타) | 유형명(비·탁본 등) |
| `md_data10` | `KSM_IDX` | 자료식별번호 | **크기** | `KSM_LIFESIZE` | `높이 203.0 너비 95.0 두께 39.7` | (메타) | 치수 텍스트 |
| `md_data11` | `KSM_KOR_NAME` | 금석문명_한글 | **소재지 주소** | `KSM_LOCATION_ADDR` | `경기도 가평군 상면 태봉리 295` | `location` | 행정 주소 |
| `md_data12` | `KSM_LIFESIZE` | 크기 | **서체** | `KSM_FONT` | `해서(楷書)` | (메타) | 서체·한자 병기 |
| `md_data13` | `KSM_LOCATION_ADDR` | 소재지 주소 | **찬자** | `KSM_PROD_CJ` | `신흠(申欽)` | (메타) | 인명·한자 병기 |
| `md_data14` | `KSM_PROD_CJ` | 찬자 | **각자** | `KSM_PROD_GJ` | `미상` | (메타) | 각자 필드 |
| `md_data15` | `KSM_PROD_GJ` | 각자 | **서자** | `KSM_PROD_SJ` | `제액 : 김상용(金尙容) 본문 : 이홍주(李弘胄)` | (메타) | 제액·본문 집필 정보 |
| `md_data16` | `KSM_PROD_SJ` | 서자 | **공공누리유형 코드** | `KSM_USE_SCOPE` | `2533` | `source_inventory.license_type` 보조 | 이용범위 코드(숫자) |
| `md_data17` | `KSM_SUVEY` | 개관 | **지정종목코드명** | `FORMAL_CDNM` | `비지정유산` | (메타) | `md_data20` 코드와 명칭 쌍 |
| `md_data18` | `KSM_TYPE` | 유형코드 | **개관** | `KSM_SUVEY` | `이계(1528~1593)의 신도비명이다. …` | `exclude_reason`/`notes` 참고용 (학습 텍스트 아님) | 장문 설명문; 판독문 아님 |
| `md_data19` | `KSM_TYPENM` | 유형코드 내용 | **저작권자** | `KSM_COPYRIGHT` | `국립문화유산연구원` | `source_inventory` | 기관명 |
| `md_data20` | `KSM_USE_SCOPE` | 공공누리유형 코드 | **지정종목코드** | `FORMAL_CD` | `371` | (메타) | `md_data17`과 코드·명칭 쌍 |
| `md_data21` | `ROW_NUM` | 순번 | **공공누리 이용조건명** | `USE_SCOPE_NM` | `출처표시` | `source_inventory.license_type` | 공공누리 제1유형 문구 |
| `md_data22` | `USE_SCOPE_NM` | 공공누리유형 이용조건 | **데이터 링크(상세 URL)** | `DATA_LINK` | `http://portal.nrich.go.kr/kor/ksmUsrView.do?menuIdx=584&ksm_idx=3` | `source_inventory.source_url` | `ksm_idx`와 `md_data2` 일치 |

### 18.3.1 2건 교차 검증 (이소한신도비)

§18.3 매핑을 **이소한신도비** (`KSM_IDX=5`) XML로 재확인했다. 22개 필드 모두 동일한 의미·태그 대응이 유지되었다.

| XML 태그 | 실제 필드 | 샘플 2 값 | 교차 검증 포인트 |
| --- | --- | --- | --- |
| `md_data1` | `ROW_NUM` | `2` | 샘플 1(`1`)과 달리 목록 2번째 행; `KSM_IDX`와 무관 |
| `md_data2` | `KSM_IDX` | `5` | `md_data22` URL `ksm_idx=5` 일치 |
| `md_data3` | `KSM_KOR_NAME` | `이소한신도비` | 한글 명칭 |
| `md_data4` | `KSM_CHI_NAME` | `李昭漢神道碑` | 한문 명칭 |
| `md_data5` | `KSM_EAR` | `339` | 샘플 1과 동일 코드; `md_data6=조선` |
| `md_data6` | `KSM_EAR_NM` | `조선` | 시대명 |
| `md_data7` | `KSM_AGE` | `1675년(숙종 1)` | 연대·연호 |
| `md_data8` | `KSM_TYPE` | `2844` | 샘플 1과 동일 코드; `md_data9=비` |
| `md_data9` | `KSM_TYPENM` | `비` | 유형명 |
| `md_data10` | `KSM_LIFESIZE` | `높이 197.0 너비 89.0 두께 25.0` | 치수 텍스트 |
| `md_data11` | `KSM_LOCATION_ADDR` | `경기도 가평군 조종면 현리 150-7` | 행정 주소 |
| `md_data12` | `KSM_FONT` | `해서(楷書)` | 서체 |
| `md_data13` | `KSM_PROD_CJ` | `송시열(宋時烈)` | 개관(`md_data18`)의 “찬자 … 송시열”과 일치 |
| `md_data14` | `KSM_PROD_GJ` | `미상` | 각자 |
| `md_data15` | `KSM_PROD_SJ` | `이정영(李正英)` | 개관의 “서자 … 이정영”과 일치 |
| `md_data16` | `KSM_USE_SCOPE` | `2533` | 샘플 1과 동일 코드 |
| `md_data17` | `FORMAL_CDNM` | `비지정유산` | `md_data20=371`과 명칭·코드 쌍 |
| `md_data18` | `KSM_SUVEY` | `이소한(1598~1645)의 신도비…` | 현대어 개관; 판독문 아님 |
| `md_data19` | `KSM_COPYRIGHT` | `국립문화유산연구원` | 저작권/제공 기관 |
| `md_data20` | `FORMAL_CD` | `371` | 샘플 1과 동일 코드 |
| `md_data21` | `USE_SCOPE_NM` | `출처표시` | 공공누리 제1유형 |
| `md_data22` | `DATA_LINK` | `…ksm_idx=5` | 상세 URL |

**샘플 간 공통 코드값** (유형·지정·이용범위 코드表가 레코드 간 공유됨을 시사):

| 필드 | 샘플 1 (이계) | 샘플 2 (이소한) | 해석 |
| --- | --- | --- | --- |
| `KSM_EAR` | `339` | `339` | 조선 시대 코드 |
| `KSM_TYPE` | `2844` | `2844` | `비` 유형 코드 |
| `KSM_USE_SCOPE` | `2533` | `2533` | 공공누리 코드 (명칭: 출처표시) |
| `FORMAL_CD` | `371` | `371` | 비지정유산 코드 |

**필드별 값 형식 차이 (매핑은 동일):** `KSM_PROD_SJ`(`md_data15`)는 레코드마다 서식이 다를 수 있다. 샘플 1은 `제액 : … 본문 : …` 복합 문자열, 샘플 2는 `이정영(李正英)` 단일 인명.

### 18.4 역방향 조회 (실제 필드 → XML 태그)

파서·변환 코드에서 **의미 기준**으로 XML 태그를 찾을 때 사용한다.

| 실제 필드 코드 | XML 태그 | 내부 변수명 권장 |
| --- | --- | --- |
| `ROW_NUM` | `md_data1` | `row_num` |
| `KSM_IDX` | `md_data2` | `ksm_idx` |
| `KSM_KOR_NAME` | `md_data3` | `title_ko` |
| `KSM_CHI_NAME` | `md_data4` | `title_hanja` |
| `KSM_EAR` | `md_data5` | `era_code` |
| `KSM_EAR_NM` | `md_data6` | `era` |
| `KSM_AGE` | `md_data7` | `year` |
| `KSM_TYPE` | `md_data8` | `type_code` |
| `KSM_TYPENM` | `md_data9` | `object_type` |
| `KSM_LIFESIZE` | `md_data10` | `dimensions` |
| `KSM_LOCATION_ADDR` | `md_data11` | `location` |
| `KSM_FONT` | `md_data12` | `script_style` |
| `KSM_PROD_CJ` | `md_data13` | `author_praise` |
| `KSM_PROD_GJ` | `md_data14` | `author_engrave` |
| `KSM_PROD_SJ` | `md_data15` | `author_compose` |
| `KSM_USE_SCOPE` | `md_data16` | `use_scope_code` |
| `FORMAL_CDNM` | `md_data17` | `formal_name` |
| `KSM_SUVEY` | `md_data18` | `overview` |
| `KSM_COPYRIGHT` | `md_data19` | `copyright_holder` |
| `FORMAL_CD` | `md_data20` | `formal_code` |
| `USE_SCOPE_NM` | `md_data21` | `use_scope_name` |
| `DATA_LINK` | `md_data22` | `detail_url` |

### 18.5 변환 코드용 매핑 딕셔너리

```python
# NRICH OpenAPI XML: md_data tag number -> canonical field code
# Verified against 이계신도비 (KSM_IDX=3) and 이소한신도비 (KSM_IDX=5), 2026-06-29.
NRICH_MD_DATA_TO_FIELD = {
    "md_data1": "ROW_NUM",
    "md_data2": "KSM_IDX",
    "md_data3": "KSM_KOR_NAME",
    "md_data4": "KSM_CHI_NAME",
    "md_data5": "KSM_EAR",
    "md_data6": "KSM_EAR_NM",
    "md_data7": "KSM_AGE",
    "md_data8": "KSM_TYPE",
    "md_data9": "KSM_TYPENM",
    "md_data10": "KSM_LIFESIZE",
    "md_data11": "KSM_LOCATION_ADDR",
    "md_data12": "KSM_FONT",
    "md_data13": "KSM_PROD_CJ",
    "md_data14": "KSM_PROD_GJ",
    "md_data15": "KSM_PROD_SJ",
    "md_data16": "KSM_USE_SCOPE",
    "md_data17": "FORMAL_CDNM",
    "md_data18": "KSM_SUVEY",
    "md_data19": "KSM_COPYRIGHT",
    "md_data20": "FORMAL_CD",
    "md_data21": "USE_SCOPE_NM",
    "md_data22": "DATA_LINK",
}
```

### 18.6 `readings.csv` 생성 시 권장 매핑

| `readings.csv` 컬럼 | NRICH 실제 필드 | XML 태그 | 비고 |
| --- | --- | --- | --- |
| `source` | — | — | 고정값 `nrich` |
| `source_record_id` | `KSM_IDX` | `md_data2` | split·조회 키 |
| `inscription_id` | `KSM_IDX` | `md_data2` | 1차 split 기준 후보; 다중 판독문 확인 후 확정 |
| `title` | `KSM_KOR_NAME` + `KSM_CHI_NAME` | `md_data3`, `md_data4` | 한글 우선, 한문은 보조 |
| `era` | `KSM_EAR_NM` | `md_data6` | |
| `year` | `KSM_AGE` | `md_data7` | |
| `location` | `KSM_LOCATION_ADDR` | `md_data11` | |
| `text_raw` | *(별도 수집)* | — | **이 XML 블록에 없음** |
| `text_normalized` | *(별도 수집)* | — | 판독문 수집 후 전처리 |
| `is_training_text` | — | — | 판독문 확보·권리 확인 후 결정 |
| `notes` | `KSM_SUVEY` | `md_data18` | 개관은 메모용; 학습 텍스트로 쓰지 않음 |

### 18.7 파싱 시 주의사항

1. **명세서 컬럼명을 XML 태그에 직접 대입하지 않는다.** 반드시 §18.3 또는 §18.5를 사용한다.
2. **`KSM_IDX`는 `md_data2`다.** 명세서상 `md_data10`이 아니다.
3. **`DATA_LINK`는 `md_data22`다.** `md_data1`의 `1`은 URL이 아니다.
4. **`md_data18`(개관)은 현대어 설명문**이다. `readings.csv`의 `text_raw`로 넣지 않는다(`PROJECT_OVERVIEW.md` §6.2).
5. **추가 레코드로 교차 검증**한다. 현재 2건 XML에서 동일 매핑 확인. JSON 응답·상세 API operation은 별도 확인 필요.
6. **`KSM_USE_SCOPE` 코드 `2533`**과 조사 문서의 AI 학습 가능 코드 **`3397`**은 서로 다른 코드값이다. 레코드별 이용범위는 코드表·상세 페이지로 확인한다.

### 18.8 검증 상태

| 항목 | 상태 |
| --- | --- |
| 샘플 건수 | **2건** (이계신도비 `KSM_IDX=3`, 이소한신도비 `KSM_IDX=5`) |
| 매핑 확신도 | 2건 모두 22/22 필드 동일 매핑; URL·ID·코드·명칭 쌍·개관↔찬자/서자 교차 일치 |
| 다건 교차 검증 | **2건 완료** — 수집 파이프라인 구현 시 3~8건 추가 확인 권장 |
| JSON 응답 동일 여부 | **미확인** — XML 목록 응답 기준 |

---

## 19. 관련 문서

* `docs/PROJECT_OVERVIEW.md`
* `docs/DATA_PREPROCESSING_RULES.md`
* `docs/DATA_INTAKE_RUNBOOK.md`
* `docs/DECISIONS.md`
* `docs/EXPERIMENT_LOG.md`
* `docs/RISK_CHECKLIST.md`
* `docs/WEEKLY_PLAN.md`
