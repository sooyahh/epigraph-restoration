# WEEKLY_PLAN.md — 한 달 반 실행 계획

기간: 약 6주(한 달 반). 핵심 원칙은 **"확정 데이터 확인 → 검수 가능한 baseline → 단계적 개선 → 설명 가능한 발표"**.
순서: 문제정의 → 데이터 원천/스키마 확인 → EDA → baseline → feature/model 개선 → 결과 해석 → 발표자료.

> 각 주차는 "끝났다"의 기준(산출물)이 명확하다. 산출물이 없으면 그 주는 끝난 것이 아니다.
> 매주 최소 1개 실험을 `docs/EXPERIMENT_LOG.md`에 남긴다.

---

## 1주차 — 문제정의 + 데이터 구조 파악 + baseline 준비

목표: 무엇을 푸는지 정리하고, 실제 데이터 원천과 목표 스키마 후보를 확인한다.

- 문제정의 정리: 입력/출력 목표 형태, 평가 지표 후보(`Exact@k`, `Group@k`)를 팀이 검토.
- **데이터 구조 파악**: 실제 원천의 컬럼·식별자·결손 표기 방식을 확인. `inscription_id`/`artifact_id`는 아직 후보명이다.
- **전처리 규칙 확정**: `docs/DATA_PREPROCESSING_RULES.md`의 `TODO:` 중 결정 가능한 것 확정 → `docs/DECISIONS.md` 기록.
- **baseline 준비**: 검수 완료 데이터가 생긴 뒤 연결할 수 있도록 baseline/eval 코드(`src/baseline/`, `src/eval/run_fill_mask_eval.py`)를 점검. archive된 smoke 결과는 성능으로 쓰지 않음.

산출물: 문제정의 메모, 실제 데이터 구조 확인표, 전처리 규칙 1차 초안, 데이터 상태 기록.

## 2주차 — EDA + 전처리 데이터 생성

목표: 실제 원천에서 검수 가능한 후보 데이터를 만들고 누수 없는 split 기준을 설계한다.

- EDA: 시대/길이/결손비율 분포, 이상치·중복 점검.
- 목표 스키마 후보의 원문 / 마스킹 입력 / 라벨 분리 방식 설계.
- inscription 단위 split 구현 + 랜덤 split과 누수 비교 실험(EXP).
- 수동 검수 전 데이터는 `candidate`로만 표시하고, 검수 완료 후에만 버전 고정.

산출물: 후보 데이터, split 설계안, EDA 요약, 누수 위험 점검 기록.

## 3주차 — Baseline 완성

목표: 검수 완료 데이터가 확보된 경우에만 끝까지 돌아가는 baseline을 실행한다.

- 검수 완료 데이터와 평가 라벨이 있으면 `bert-base-chinese` 후보 baseline 학습/평가(EXP-001).
- `Exact@1/5/10`, `Group@1/5/10` 측정·기록.
- baseline 복원 예시 몇 개 확보(정성 확인).

산출물: 실제 데이터 기반 baseline 지표, 첫 복원 예시, `docs/EXPERIMENT_LOG.md` 갱신.

## 4주차 — feature/model 개선

목표: baseline보다 나은 결과를 만든다.

- 모델 비교: mBERT, (가능 시) SikuRoBERTa.
- 하이퍼파라미터 조정(lr/batch/epoch/max_len).
- 이체자/glyph 정보 등 feature 보강 검토.

산출물: 모델 비교표, 개선 전후 성능 변화, 채택/보류 결정(`docs/DECISIONS.md`).

## 5주차 — 앙상블 또는 후보 재랭킹

목표: 마지막 성능 끌어올리기. 단, 설명 가능한 범위 안에서.

- 여러 모델 후보 결합(앙상블) 또는 top-k 후보 재랭킹 시도.
- 복잡도 대비 효과 점검(`docs/RISK_CHECKLIST.md` 4번).
- 효과 없으면 과감히 버리고 baseline+개선안으로 확정.

산출물: 최종 모델 후보 확정, 최종 성능 지표.

## 6주차 — 결과 해석 + 발표자료 작성

목표: 결과를 사람이 이해하도록 정리하고 발표를 완성한다.

- 결과 해석: 잘 되는 경우/안 되는 경우 분석, 한계 정리.
- 정성 예시(좋은 복원 사례) 시각화.
- 발표 슬라이드·보고서 작성(`docs/PROMPTS.md` 발표 템플릿).
- 예상 Q&A 준비, 리허설.

산출물: 발표자료, 보고서, 데모/예시, Q&A 노트.

---

## 매주 공통 루틴

- 주 시작: `docs/WEEKLY_PLAN.md`로 이번 주 목표 확인.
- 주중: 실험은 `docs/EXPERIMENT_LOG.md`, 결정은 `docs/DECISIONS.md`에 기록.
- 주 마감: `docs/RISK_CHECKLIST.md`로 위험 점검, 다음 주 우선순위 정리(PM).

> 일정은 진행 상황에 따라 조정될 수 있다. 단, "확정 데이터 확인 먼저, baseline은 그 다음" 순서는 바꾸지 않는다.

---

*관련 문서: `docs/PROJECT_OVERVIEW.md`, `docs/ROLE_GUIDE.md`, `docs/EXPERIMENT_LOG.md`, `docs/RISK_CHECKLIST.md`.*
