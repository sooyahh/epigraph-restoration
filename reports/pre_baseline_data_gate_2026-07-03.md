# Baseline 착수 전 데이터 분포 점검 리포트

- **생성일:** 2026-07-03
- **범위:** `readings.csv`, `splits.csv`, truncation 비교 리포트 (읽기 전용 점검)
- **전제 (사용자 확인):** 한글 치환(`hangul → □`)은 `is_training_text=False` / `quality_flag=review`로 학습에서 분리되어 **학습 오염 없음**으로 간주
- **본 리포트는 데이터·코드·git을 수정하지 않고 산출됨**

---

## 1. readings.csv 전체 통계

**입력:** `data/processed_layer/readings.csv`

| 항목 | 값 |
| --- | ---: |
| 전체 row 수 | 14,702 |
| 고유 `inscription_id` 수 | 6,248 |
| 고유 `reading_id` 수 | 14,702 (row와 1:1) |

### is_training_text

| 값 | row 수 |
| --- | ---: |
| `True` | 7,158 |
| `False` | 7,544 |

**확인된 사실:** `is_training_text=True`와 `has_real_defect=True`는 교차 0건 (완전 배타).

### quality_flag

| 값 | row 수 |
| --- | ---: |
| `ok` | 11,065 |
| `review` | 3,637 |

**확인된 사실:** `quality_flag=review`이면서 `is_training_text=True`인 row는 **0건**.

### has_real_defect

| 값 | row 수 |
| --- | ---: |
| `True` | 7,544 |
| `False` | 7,158 |

### 보조 집계

| 항목 | 값 |
| --- | ---: |
| `is_training_text=True` 고유 inscription | 3,400 |
| `is_training_text=False`만 있는 inscription | 2,848 |
| `notes` 비어 있지 않은 row | 14,702 |
| `text_length` 최대 | 100,259 |

---

## 2. 시대/시기 컬럼 탐색

**readings.csv 컬럼 전체:**  
`reading_id`, `source`, `source_record_id`, `inscription_id`, `artifact_id`, `title`, **`era`**, **`year`**, **`material`**, `location`, `text_raw`, `text_normalized`, `text_length`, `hanja_count`, `has_real_defect`, `defect_count`, `defect_markers_raw`, `is_training_text`, `exclude_reason`, `quality_flag`, `notes`

### 시대를 나타낼 가능성이 있는 컬럼

| 컬럼 | 근거 | 고유값 수 | 비고 |
| --- | --- | ---: | --- |
| **`era`** | 컬럼명·값이 왕조/시대 체계 | 15 | 전 row 채움 (14,702) |
| **`year`** | 연대·년호·「中國 ○○」 표기 | 1,336 | 14566/14702 row 채움, 136건 `(NA)` |
| `material` | 재질(돌· bronze 등) 가능성 | 0 (전부 공란) | 시대 지표로 부적합 |
| `location` | 지역/출토지 | (미집계) | **시대가 아닌 공간 정보**로 추정 |
| `title` | 제목에 연호 포함 가능 | — | 파생 필요, 원본 컬럼 아님 |

### `era` 값 상위 (전체 15값)

| era | row 수 |
| --- | ---: |
| 조선 | 8,753 |
| 고려 | 2,074 |
| 낙랑 | 968 |
| 일제강점기 | 906 |
| 통일신라 | 554 |
| 대한제국 | 480 |
| 고구려 | 340 |
| 신라 | 336 |
| 백제 | 198 |
| 발해 | 37 |
| 가야 | 30 |
| 시대미상 | 19 |
| 고대 | 4 |
| 근대이후 | 2 |
| 현대 | 1 |

### `year` 값 상위 30

| year | row 수 |
| --- | ---: |
| 미상 | 2,066 |
| 중국 한대(漢代) | 205 |
| (NA) | 136 |
| 1871년(고종 8) | 106 |
| 중국 후한(後漢) | 84 |
| 591년(진평왕 13) | 70 |
| 1889년(고종 26) | 65 |
| 1900년(광무 4) | 56 |
| 1713년(숙종 39) | 49 |
| 1932년 | 46 |
| 1873년(고종 10) | 46 |
| 1890년(고종 27) | 44 |
| 1884년(고종 21) | 43 |
| 1707년(숙종 33) | 43 |
| 1881년(고종 18) | 42 |
| 1936년 | 41 |
| 1696년(숙종 22) | 41 |
| 1710년(숙종 36) | 41 |
| 1894년(고종 31) | 40 |
| 1933년 | 39 |
| 1716년(숙종 42) | 39 |
| 1737년(영조 13) | 39 |
| 1727년(영조 3) | 39 |
| 1901년(광무 5) | 39 |
| 1941년 | 38 |
| 1740년(영조 16) | 38 |
| 1725년(영조 1) | 38 |
| 1879년(고종 16) | 38 |
| 1688년(숙종 14) | 37 |
| 1705년(숙종 31) | 37 |

### 시대 컬럼 후보 (확정 아님, 제안만)

1. **1순위 후보: `era`** — 왕조/시대 단위 분포·split 교차표가 이미 `splits.csv`에도 존재. 해상도는 거칠지만 해석 용이.
2. **2순위 후보: `year`** — 연대별 세분 평가 가능. `미상`·`(NA)`·「中國 ○○」 혼재로 **정규화 없이는 시대별 지표 해석이 어려울 수 있음** (추측).
3. **파생 후보:** `year`에서 서기/년호 추출, 또는 `era`×`year` 교차 — baseline gate 범위 밖.

---

## 3. 시대별 분포 (`era` 기준)

**확인된 사실:** `era` 컬럼이 존재하므로 시대별 집계 **가능**.  
아래 `avg_len`은 `text_length` 평균, `avg_hanja`는 `hanja_count` 평균.

| era | 전체 row | train row (`is_training_text=True`) | inscription 수 | 실제 결손 row | avg text 길이 | avg hanja_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 조선 | 8,753 | 4,893 | 4,065 | 3,860 | 700.2 | 673.6 |
| 고려 | 2,074 | 728 | 763 | 1,346 | 648.9 | 569.4 |
| 낙랑 | 968 | 346 | 416 | 622 | 147.4 | 44.1 |
| 일제강점기 | 906 | 538 | 437 | 368 | 527.1 | 482.1 |
| 통일신라 | 554 | 77 | 101 | 477 | 2,527.2 | 916.5 |
| 대한제국 | 480 | 307 | 235 | 173 | 483.6 | 456.2 |
| 고구려 | 340 | 123 | 77 | 217 | 1,062.8 | 333.6 |
| 신라 | 336 | 45 | 60 | 291 | 761.2 | 173.6 |
| 백제 | 198 | 67 | 51 | 131 | 934.4 | 336.8 |
| 발해 | 37 | 9 | 22 | 28 | 1,298.7 | 594.2 |
| 가야 | 30 | 14 | 8 | 16 | 29.0 | 15.1 |
| 시대미상 | 19 | 6 | 9 | 13 | 40.7 | 33.8 |
| 고대 | 4 | 4 | 2 | 0 | 5.0 | 3.8 |
| 근대이후 | 2 | 0 | 1 | 2 | 43.0 | 43.0 |
| 현대 | 1 | 1 | 1 | 0 | 309.0 | 309.0 |

**주목 (확인된 사실):**

- **통일신라**는 row 대비 `is_training_text=True` 비율이 낮고(77/554), 평균 길이·결손 row가 많음 → 장문·결손 편중.
- **낙랑**은 avg hanja_count가 낮음(44.1) → 짧은 텍스트 비중 큼.

---

## 4. split별 분포

**입력:** `data/model_inputs/splits.csv` + readings merge

### inscription 수 (splits.csv 기준)

| split | inscription 수 |
| --- | ---: |
| train | 4,258 |
| valid | 912 |
| test | 914 |
| **합계** | **6,084** |

### row 수 (readings × split merge)

| split | row 수 | 그중 `is_training_text=True` | 그중 `has_real_defect=True` |
| --- | ---: | ---: | ---: |
| train | 10,092 | 4,982 | 5,110 |
| valid | 2,225 | 1,110 | 1,115 |
| test | 2,155 | 1,066 | 1,089 |
| no_split | 230 | 0 | 230 |
| **합계** | **14,702** | **7,158** | **7,544** |

### split 미할당

| 항목 | 값 |
| --- | ---: |
| split 없는 inscription | 164 |
| split 없는 row | 230 |

**확인된 사실:** split 미할당 164 inscription은 **전부** `is_training_text=True` row가 없음(defect-only).

### inscription split 중복

| 검사 | 결과 |
| --- | --- |
| 동일 `inscription_id`가 2개 이상 split에 등장 | **0건** |

**확인된 사실:** inscription 단위 leakage는 없음.

### split ↔ training 정합성 (중요)

| 검사 | 결과 |
| --- | ---: |
| `is_training_text=True` inscription 중 split 미포함 | 0 |
| split 포함 inscription 중 training row 없음 | **2,684** |
| `splits.csv` `num_readings` vs readings 실측 일치 | 6,084 / 6,084 |

**확인된 사실:** `splits.csv`는 학습 가능 inscription 3,400건뿐 아니라, **training row가 없는 defect-only inscription 2,684건**도 train/valid/test에 배정되어 있음.

**추측 (코드 대조):** 현재 `src/data/split_by_inscription.py`는 `is_training_text=True` inscription만 split 대상으로 설계되어 있으나, **저장된 `splits.csv`(6,084건)는 그 규칙과 불일치**함. mask 생성(`build_artificial_mask.py`)은 training row만 사용하므로 baseline FT 자체는 training row 기준이지만, **split 메타·시대 분포 해석 시 defect-only inscription 포함 여부를 인지해야 함**.

### split별 시대 분포 (`splits.csv`의 `era`, inscription 단위)

| era | train | valid | test |
| --- | ---: | ---: | ---: |
| 조선 | 2,787 | 585 | 594 |
| 고려 | 534 | 113 | 107 |
| 낙랑 | 290 | 54 | 65 |
| 일제강점기 | 308 | 69 | 55 |
| 대한제국 | 156 | 33 | 41 |
| 통일신라 | 44 | 27 | 12 |
| 신라 | 34 | 8 | 14 |
| 고구려 | 50 | 10 | 13 |
| 백제 | 31 | 7 | 8 |
| 발해 | 10 | 3 | 2 |
| 가야 | 4 | 0 | 3 |
| 시대미상 | 7 | 2 | 0 |
| 고대 | 2 | 0 | 0 |
| 근대이후 | 0 | 1 | 0 |
| 현대 | 1 | 0 | 0 |

**확인된 사실:** test split의 **통일신라** inscription(12)이 valid(27) 대비 적음 — 소수 클래스이므로 시대별 test 지표 분산 클 수 있음 (추측).

---

## 5. truncation 위험 요약

### 참고 리포트

| 파일 | 샘플 수 | truncation 판정 |
| --- | ---: | --- |
| `reports/xls_detail_truncation_compare.json` | 8 | **1건** (`ksm_idx=2866`, XLS 9,782자 vs 상세 18,514자, Δ=+8,732) |
| `reports/xls_detail_truncation_report.json` | 10 | **0건** (집계 방식: inscription 단위 max 비교, 임계치 상이) |
| `reports/xls_detail_truncation_summary.csv` | 10 | CSV 요약, truncated 0 |

**확인된 사실:** XLS 대비 NRICH 상세 페이지 **장문 생략(truncation) 위험은 실재**함(2866 사례). 다만 두 리포트의 샘플·매칭·판정 기준이 달라 **표면 수치가 다름**.

### readings.csv 기반 truncation 프록시

| 지표 | 값 |
| --- | ---: |
| `text_length ≥ 5,000` row | 161 |
| 그중 `is_training_text=True` | 18 |
| `notes`에 `detail_augment` 포함 | 8,000 |
| `notes`에 `detail_augment_truncated` 포함 | **0** |
| `text_length ≥ 5,000` 이면서 augment note 없음 | 22 |
| `text_length ≥ 5,000` + training + augment note 없음 | **9** |

**확인된 사실:**

- 현재 `readings.csv`는 **`source=detail_augment` 전건 보강**(8,000건) 흔적이 있고, 코드상 `--augment-truncated-only`가 남기는 `detail_augment_truncated` 태그는 **0건** → “5,000자 이상만 상세 보강” 정책과 **실제 산출물이 일치하지 않을 수 있음** (당시 실행 플래그는 notes만으로는 확정 불가).
- 장문(≥5,000) 161건 중 22건은 detail augment note가 없음 → XLS 원문 그대로일 가능성.
- **학습 대상 장문 9건**은 augment note 없이 ≥5,000자 — truncation 잔존 위험 **최소 점검 대상**.

**학습 대상 장문 9건 (augment note 없음, 확인됨):**

| reading_id | inscription_id | text_length | era |
| --- | --- | ---: | --- |
| nrich_ksm_011051_x8384 | insc_nrich_011051 | 7,019 | 조선 |
| nrich_ksm_008299_x9354 | insc_nrich_008299 | 7,157 | 조선 |
| nrich_ksm_008550_x9755 | insc_nrich_008550 | 5,202 | 조선 |
| nrich_ksm_014741_x11679 | insc_nrich_014741 | 5,021 | 조선 |
| nrich_ksm_008793_x12030 | insc_nrich_008793 | 5,286 | 대한제국 |
| nrich_ksm_013380_x12585 | insc_nrich_013380 | 5,902 | 조선 |
| nrich_ksm_002201_x12985 | insc_nrich_002201 | 5,516 | 조선 |
| nrich_ksm_001302_x13754 | insc_nrich_001302 | 6,087 | 조선 |
| nrich_ksm_008092_x14619 | insc_nrich_008092 | 5,798 | 조선 |

### 남은 위험 정리

| 위험 | 근거 | baseline 전/후 |
| --- | --- | --- |
| XLS 장문 생략 | compare 리포트 2866 사례 | **baseline 전** 샘플로 위험 확인됨; **전수 검증은 미완** |
| ≥5,000자 training 9건 미보강 | readings notes 집계 | **baseline 전** spot-check 권장 (9건 XLS vs detail 길이) |
| FT 512 토큰 window | smoke FT 설정 (별도) | 학습 입력 truncation — **baseline 설계 이슈**, 본 gate 범위 밖 |
| 전체 14,702건 truncation 전수 스캔 | 미실시 | **baseline 이후** 개선 과제로 적합 (추측) |

---

## 6. 한글 치환·학습 오염 (전제 재확인)

**사용자 전제:** 학습 오염 없음.

**본 점검에서 확인된 근거:**

- `quality_flag=review` 3,637건 전부 `is_training_text=False`
- `is_training_text` ↔ `has_real_defect` 완전 배타
- review row의 `notes`에 `hangul_replaced` 흔적 존재 (예: `hangul_replaced=N (->□, needs_review)`)

→ **전제와 데이터 상태 일치 (확인됨).**

---

## 7. 최종 판정

### 1) baseline 착수 가능 여부

**판정: 조건부 가능 (Proceed with documented caveats)**

| 근거 (확인) | |
| --- | --- |
| ✅ | 학습 텍스트·결손·review 분리 정상; inscription split 중복 없음 |
| ✅ | 시대(`era`)·split 교차 집계 가능 |
| ⚠️ | `splits.csv` 6,084건 중 2,684건은 training row 없음 — split 해석·시대별 eval 정의 시 주의 |
| ⚠️ | truncation: 학습 장문 9건 spot-check 미완 |

### 2) 시대별 평가 가능 여부

**판정: 가능 (`era` 기준)**

- 15개 왕조/시대 값, split별 교차표 산출 완료.
- `year`는 보조·정규화 후보이나, baseline gate에서는 **`era`로 1차 eval 정의 가능** (확인).
- 통일신라·가야 등 **소수 class**는 test inscription 수가 적어 지표 분산이 클 수 있음 (추측).

### 3) truncation 추가 검증 필요 여부

**판정: 최소 추가 검증 권장 (9건 spot-check); 전수 검증은 baseline 이후**

| 구분 | 내용 |
| --- | --- |
| **baseline 전 (권장)** | `is_training_text=True` & `text_length≥5000` & augment note 없는 **9건** XLS vs NRICH detail 길이 대조 |
| **baseline 이후 (개선)** | ≥5,000 전체 161건 또는 전체 corpus truncation 전수 스캔; `augment_truncated_only` 정책과 재빌드 정합 |

---

## 부록: 사용 파일

- `data/processed_layer/readings.csv`
- `data/model_inputs/splits.csv`
- `reports/xls_detail_truncation_compare.json` / `.md`
- `reports/xls_detail_truncation_report.json` / `.md`
- `reports/xls_detail_truncation_summary.csv`
