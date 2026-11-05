# DATA_INTAKE_RUNBOOK.md — 실제 데이터 인테이크 런북

> ⚠️ **2026-07-11 감사 주의 — 이 런북은 미구현 절차의 초안이다.**
> 아래가 지시하는 세 점검 스크립트(`inspect_raw_source.py`, `profile_text_corpus.py`,
> `validate_schema.py`)는 현재 `src/data/`에 **존재하지 않는다.** 새 원천을 인테이크할 때
> 참고할 *설계안*으로만 취급하고, 실행 명령을 그대로 돌리지 말 것.
> **현재 실제로 동작하는 데이터 파이프라인**은 `docs/DATA_STATUS.md` §6 및
> `src/data/build_readings.py` → `reprocess_readings.py` → `build_artificial_mask.py`
> → `split_by_inscription.py` → `build_real_defect_inference.py`이다.

실제 금석문 데이터 파일이 들어왔을 때, **데이터를 만지기 전에** 구조·텍스트·스키마를
확인하는 순서와 명령어를 정리한다. 이 런북은 `src/data/`의 세 점검 스크립트
(`inspect_raw_source.py` → `profile_text_corpus.py` → `validate_schema.py`)를
순서대로 돌리는 절차다(위 주의 참조: 미구현).

> 전제(2026-06-27 기준): **확정 학습 데이터·확정 스키마 없음**(`docs/DATA_STATUS.md`).
> 따라서 아래 컬럼명(`inscription_id`, `original_text`, `label` 등)은 모두 "목표 스키마
> 후보"이며, 실제 파일을 보고 확정한다. 본 런북은 절차만 정리한 것으로, **실데이터가
> 없으므로 아직 실행하지 않았다.**

실행 위치는 항상 **프로젝트 루트**(`구글 ai 캡스톤/`)다. 스크립트는 모듈로 실행한다
(`python -m src.data.<모듈>`). 환경: Python 3.11, `pandas` 필요(xlsx는 `openpyxl`도).

---

## 1. raw 데이터 배치 위치 후보

원본은 **읽기 전용**으로 다룬다(`docs/DATA_PREPROCESSING_RULES.md` 원본 보존 원칙). 점검 스크립트는 입력을 읽기만 하고
원본 경로에 아무것도 쓰지 않는다. 그래도 안전을 위해 **인테이크 전용 폴더에 사본**을 두는 것을
권장한다.

| 후보 경로 | 용도 | 비고 |
|---|---|---|
| `data/raw_intake/` (신규, 권장) | 새로 받은 원천 파일의 **사본**을 두는 곳 | 원본 보존, 가공은 다른 폴더에서 |
| 원천이 준 원본 파일 | 직접 점검 가능(읽기 전용) | 이동·덮어쓰기 금지 |

- 출력(리포트·후보 데이터)은 이 폴더에 쓰지 않는다. 리포트는 `reports/`, 가공 후보는
  `candidates/`로만 나간다.
- `_archive/`, `external/BIRD/`, 원본 PDF/PPTX는 **입력으로 쓰지 않는다.**

> 예시에서는 입력 경로를 `data/raw_intake/<파일명>`으로 가정한다. 실제 받은 경로로 바꿔 쓴다.

---

## 2. 1단계 — `inspect_raw_source.py` (구조 점검)

파일의 기본 구조(컬럼/행/결측/텍스트 컬럼 후보/결손 표기 수)를 본다.
지원 입력: `.csv` / `.xlsx` / `.json` / `.txt`.

```bash
# CSV 예시
python -m src.data.inspect_raw_source \
  --input data/raw_intake/inscriptions.csv \
  --report-name data_inspection_report.md

# XLSX 예시
python -m src.data.inspect_raw_source \
  --input data/raw_intake/inscriptions.xlsx

# JSON / TXT 도 동일하게 --input 만 바꾼다
python -m src.data.inspect_raw_source --input data/raw_intake/source.json
```

- `--report-name`은 `reports/` 아래 저장명(기본 `data_inspection_report.md`).
- 여러 원천을 비교하려면 `--report-name`을 원천별로 다르게 준다
  (예: `data_inspection_nrich.md`).
- **여기서 확인할 것:** 어떤 컬럼이 텍스트 컬럼 후보인지, 결손 표기(□ 등)가 어느 컬럼에
  몇 개 있는지. 이걸 보고 다음 단계의 `--text-column`을 정한다.

---

## 3. 2단계 — `profile_text_corpus.py` (텍스트 프로파일)

1단계에서 정한 텍스트 컬럼을 넣어 문자 수·고유 한자 수·길이 분포·결손 비율·상위/희귀 문자를 본다.

```bash
# 표 형식: --text-column 필수 (1단계에서 고른 컬럼명)
python -m src.data.profile_text_corpus \
  --input data/raw_intake/inscriptions.csv \
  --text-column original_text \
  --report-name text_profile_report.md

# .txt 파일이면 컬럼 개념이 없으므로 --text-column 생략 가능
python -m src.data.profile_text_corpus \
  --input data/raw_intake/corpus.txt
```

- 표 형식인데 `--text-column`을 빼면 의도적으로 에러를 낸다(어느 컬럼인지 추측하지 않음).
- 컬럼명(`original_text`)은 후보다. 1단계 리포트에서 실제 컬럼명으로 바꾼다.
- **여기서 확인할 것:** 결손 비율이 너무 높은지, 길이 분포가 한쪽으로 쏠리는지,
  희귀 한자가 얼마나 되는지(→ 전처리/제외 기준 논의 재료).

---

## 4. 3단계 — `validate_schema.py` (스키마 라이트 검증·누수 점검)

확정 스키마가 없다는 전제에서, **주어졌고 실제로 존재하는 컬럼만** 검사한다.
가장 중요한 건 **라벨이 입력 텍스트에 노출되는지(누수)** 점검이다.

```bash
# 가진 컬럼만 넣는다. 없는 인자는 생략하면 그 검사는 자동 skip.
python -m src.data.validate_schema \
  --input data/raw_intake/inscriptions.csv \
  --id-column inscription_id \
  --text-column masked_input \
  --label-column label
```

- 인자는 모두 선택이다. id만 있으면 id만, text만 있으면 text만 검사한다.
- `--text-column`과 `--label-column`이 함께 있으면 누수 점검을 수행한다.
  - 누수 의심 행이 있으면 메시지를 출력하고 **종료 코드 1**로 끝난다(파이프라인이 멈추게).
- 입력은 표 형식(csv/xlsx)만 지원한다.
- **참고(과검출 가능):** 누수 점검은 substring 방식이라, 정답 글자가 텍스트 내 다른 위치에
  **중복 등장**하면 실제 누수가 아니어도 플래그될 수 있다.
  플래그가 뜨면 마스크 위치를 직접 확인한다.

---

## 5. 결과 reports 확인 방법

모든 리포트는 프로젝트 루트의 **`reports/`** 폴더에 마크다운으로 저장된다(스크립트가
폴더를 자동 생성). `validate_schema`는 리포트 파일이 아니라 **표준출력**으로 결과를 낸다.

| 단계 | 산출물 | 위치 |
|---|---|---|
| 1. inspect | `data_inspection_report.md` | `reports/` |
| 2. profile | `text_profile_report.md` | `reports/` |
| 3. validate | 콘솔 출력(+ 종료 코드) | 터미널 |

확인:

```bash
ls reports/
# 에디터로 열어 본다 (예: VS Code)
code reports/data_inspection_report.md reports/text_profile_report.md
```

> `reports/`는 report 성격 산출물 전용이다. 여기에 확정 데이터(corpus_v1 등)를 만들지 않는다.

---

## 6. 절대 하지 말아야 할 것

- ❌ 원본 데이터 파일을 **이동·삭제·덮어쓰기** 하지 않는다(읽기 전용). 가공은 사본에서.
- ❌ `_archive/`, `external/BIRD/`, 원본 PDF/PPTX를 입력으로 쓰지 않는다.
- ❌ 점검 결과를 **성능 근거**로 쓰지 않는다(구조 확인일 뿐, 모델 성능 아님).
- ❌ `corpus_v1` / `final_dataset` / `train_data` 같은 **확정 데이터명**을 만들지 않는다.
  (가공 후보는 `candidates/candidate_*` 이름으로만.)
- ❌ split을 **랜덤 글자 단위**로 만들지 않는다. split이 필요하면
  `src/data/split_by_inscription.py`로 **유물/문헌 단위**로만(`--id-column` 필수).
- ❌ 누수 점검에서 종료 코드 1이 떴는데 **무시하고 다음 단계로 넘어가지 않는다.**
- ❌ 확인·기록 없이 `git add` / `commit` / `push` 하지 않는다.

---

## 7. 데이터 확인 후 기록할 것

점검이 끝나면 **반드시 문서에 남긴다.** 기록 없는 확인은 재현·발표에서 쓸 수 없다.

### `docs/DECISIONS.md` 에 남길 것 (확정 결정)
- 채택한 **원천**과 파일 형식, 권리/사용 범위(가능하면).
- 확정한 **스키마**: 실제 컬럼명 ↔ 목표 스키마 후보(`original_text`/`masked_input`/`label`/
  `inscription_id` 등) 매핑.
- **결손 표기 정책**: raw 원표기 목록과 가공 레이어 통일 기호(□) 확정,
  유니코드 정규화 방식(NFC/NFKC).
- **split 기준**: split 키로 쓸 식별자 컬럼, train/valid/test 비율.
- (이체자 그룹 매핑 소스를 정했다면 그것도.)

### `docs/EXPERIMENT_LOG.md` 에 남길 것 (실행/실험 기록)
- 데이터 **버전 라벨**과 한 줄 설명(어떤 원천을 어떤 규칙으로 정리했는지).
- 점검 실행 메모: 돌린 명령어, 입력 파일, 생성된 리포트 경로(`reports/...`),
  핵심 수치(행 수, 고유 한자 수, 결손 비율).
- split을 만들었다면 **split 방식**: `--id-column`, seed, 비율(누수 방지 근거).
- 누수 점검 결과(통과/플래그 및 처리).
- 이후 baseline(frequency/ngram) 또는 MLM 실험을 돌리면 `EXP-XXX` 템플릿으로 별도 기록.

> 정리: 확정 결정은 `DECISIONS.md`, 무엇을 어떻게 돌렸는지·수치는 `EXPERIMENT_LOG.md`.

---

*관련 문서: `docs/DATA_STATUS.md`(현재 데이터 상태·실제 파이프라인), `docs/DATA_PREPROCESSING_RULES.md`(전처리·split 규칙), `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md`.*
