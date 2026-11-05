# title_key 붕괴 → mask 독립 샘플 영향 감사

- 생성일: 2026-08-02
- 선행 산출물: 없음 (신규 집계)
- 범위: 조회·집계만 (readings/mask/splits 미수정)

## 판정 기준

- 중복 키: XLS meta title_key with >1 row
- readings 연결: XLS reading-sheet rows with title_key in dup set → xls_seq → readings.reading_id suffix _x{seq}
- 빈도 중복 제거: Within dup reading title_key groups, keep one is_training_text reading per (title_key, text_normalized); recount train label_char

## 1. 마스크 생성 단위 (실측·코드)

**판정: readings 행 단위** (`inscription_id` 중복 제거 없음).

- `build_artificial_mask_dataframe` L105–114: `training.iterrows()`로 각 reading 행마다 `_masks_for_reading` 호출
- `_masks_for_reading` L46–91: 해당 행의 `text_normalized`에서 마스크 샘플 생성

## 2. 대표 3사례 (실측)

| 사례 | KSM | readings | 고유 text | 고유 insc | mask n | split 분포 |
|---|---:|---:|---:|---:|---:|---|
| 유명악영세불망비 | 14139 | 14 | 6 | 1 | 14 | `{'train': 14}` |
| 척화비 | 11739 | 12 | 1 | 1 | 12 | `{'train': 12}` |
| 김명진영세불망비 | 13201 | 12 | 2 | 1 | 12 | `{'test': 12}` |

## 3. 전체 규모 (실측)

- 중복 title_key: **97** (메타 초과 행 130)
- 해당 키의 XLS 판독문 행: **410**
- 연결 readings 행: **407** (training 245, 고유 inscription 92)
- 연결 mask 샘플: **587** / 66537 = **0.8822%**
- 연결 readings를 가진 중복 키 중 inscription 1개: 92, >1개: 0

## 4. EXP-013 빈도 구간 변동 (실측)

- train mask before → after dedupe: 46139 → 45949 (dropped reading_ids=158)
- 구간이 바뀐 한자 수: **10**

| char | before | bucket_before | after | bucket_after |
|---|---:|---|---:|---|
| 嵌 | 3 | rare (2-5) | 0 | unseen_in_train |
| 碑 | 100 | very_common (100+) | 97 | common (21-100) |
| 侯 | 21 | common (21-100) | 19 | mid (6-20) |
| 凜 | 3 | rare (2-5) | 1 | hapax (count=1) |
| 嵩 | 1 | hapax (count=1) | 0 | unseen_in_train |
| 橋 | 6 | mid (6-20) | 5 | rare (2-5) |
| 算 | 1 | hapax (count=1) | 0 | unseen_in_train |
| 管 | 21 | common (21-100) | 20 | mid (6-20) |
| 續 | 2 | rare (2-5) | 1 | hapax (count=1) |
| 賣 | 2 | rare (2-5) | 1 | hapax (count=1) |

## 5. Cross-split (실측)

- 마스크가 둘 이상 split에 걸친 중복 title_key 그룹: **0**
  - 그중 inscription 1개인데 cross-split: **0** (기대값 0이면 정상)
  - inscription 2개 이상이라 cross-split 가능: **0**


## 확인하지 못한 항목

- 메타 중복 키 97개 중 판독문 시트에 동일 title_key 행이 없는 키는 readings 연결 0 (판독문 미존재/미매핑). 키별 미연결 목록은 JSON의 `item3_key_collapse` 규모만 집계.
- EXP-013 Exact@k 수치 자체는 재계산하지 않음 (빈도 구간 소속 변화만).

