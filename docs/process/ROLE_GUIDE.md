# ROLE_GUIDE.md — 팀 역할 및 현재 실행 책임

작은 팀에서는 한 사람이 여러 역할을 겸할 수 있지만, 각 산출물에는 최종 책임자 한 명을 둔다. 이 문서는 도구별 업무가 아니라 **사람의 책임과 완료 조건**을 정의한다. 마지막 갱신일은 **2026-07-12, 2026-09-17(EXP-013~019/DECISIONS [011]~[016] 반영)**이다.

## 1. 데이터 분석 담당

**책임:** 데이터 원천, 전처리, split, EDA, 품질과 누수 위험을 관리한다.

- 원본과 가공 데이터의 출처·버전·이용 조건 기록
- `readings.csv`, `splits.csv`, 인공 마스크 데이터의 규모와 정의 관리
- 실제 결손 `□`와 한글 치환 `◇` 분리 검증
- inscription 단위 split과 cross-split 문맥 중복 감사
- 시대·길이·빈도·결손·UNK 기준 EDA와 층화 분석
- 전수 수동 검수 범위와 표본 검수 결과 기록

**현재 우선 작업:**

1. 14,702행, 7,158 학습 대상, 66,537 마스크 샘플 등 혼재한 수치의 단위 정의를 통일한다.
2. 장문 truncation과 비문별 마스크 샘플 과대표집 여부를 점검한다.
3. 데이터 이용 조건과 전수 수동 검수의 완료 기준을 정한다.
4. **(2026-09-17 추가) cross-split exact duplicate 26건(EXP-019 저장소 감사)의 원인이 DECISIONS [013] title_key 붕괴와 같은지 확인한다 — BM25/ChromaDB retrieval 실험(EXP-019B/C) 착수 전 필수 선행 작업이다(DECISIONS [015]).**

**완료 조건:** 데이터 버전, 생성 코드, split 기준, 품질 통계, 알려진 한계가 문서와 실험 로그에서 추적 가능하다.

## 2. AI 모델 담당

**책임:** baseline, fine-tuning, 평가 및 체크포인트를 재현 가능하게 관리한다.

- validation으로 하이퍼파라미터와 체크포인트 선택
- SillokBERT GPU full-data fine-tuning
- full test 9,719건 평가와 층화 지표 산출
- 후보별 문자·점수·순위 저장
- 필요 시 `bert-base-chinese` 동일 조건 비교
- 실행 환경, seed, 데이터 버전, 명령, 커밋, 산출물 체크섬 기록
- **(2026-09-17 추가) EXP-018 GPU 학습 모니터링·백업, EXP-019 하이퍼파라미터(B1/B2/Lmax 등) 반복 실행을 GrokBot(자율 에이전트)에 위임할 경우, 위임 전 작업 범위·금지 사항을 명시하고, 위임 후에는 자기 완료 보고를 그대로 믿지 않고 실제 체크포인트/로그/수치로 재검증한다(DECISIONS [016], AI_COUNCIL.md §7).**

**현재 상태:** EXP-011 CPU subsample baseline과 **EXP-012 Colab GPU full-data + full test(n=9,719)** 까지 완료했다 (DECISIONS [010]). 공식 full-test 비교 수치는 EXP-012다. 이후 FT 층화 분석(EXP-013), valid 기준선 확보(EXP-015, DECISIONS [012]/[014]), 형식 정합(EXP-016), ZS/FT 보간 기각(EXP-017)까지 진행됐다. **EXP-018(lr 재탐색)은 018A/018B GPU 학습과 post-hoc valid Exact@k 평가(2026-09-17, subsample n=2,000)까지 완료됐으나, 둘 다 DECISIONS [012] gate(≥+0.7pp AND McNemar p<0.05)를 통과하지 못해 공식 후보로 승격하지 않았다 — EXP-015를 계속 공식 valid 기준선으로 유지한다.** EXP-019(ARI-8B 통합)는 설계·저장소 감사만 완료된 상태이며 **현재 최우선 작업**이다(`EXPERIMENT_LOG.md` 참조).

**완료 조건:** 제3자가 같은 코드·데이터·설정으로 실험을 재실행할 수 있고, 최종 모델과 지표가 validation/test 정책을 준수한다.

## 3. 발표·보고서 담당

**책임:** 결과와 한계를 일반 청중도 이해할 수 있게 전달한다.

- 문제, 데이터, 방법, 실험 발전 과정을 한 흐름으로 구성
- EXP-011의 50.45%@5(n=2,000)를 최종·full-test 성능으로 표현하지 않도록 검수
- **공식 full-test 수치는 EXP-012** (Exact@1 33.75% / @5 53.04% / @10 60.47%, n=9,719)
- 인공 마스크 정량평가와 실제 결손 정성 사례를 구분
- 성공 사례뿐 아니라 빈도·UNK·정형구 중복에 따른 실패 사례 제시
- 모든 수치를 `EXPERIMENT_LOG.md`와 대조
- **(2026-09-17 추가) EXP-018을 "학습 완료"와 "Exact@k 평가 완료"를 혼동해 인용하지 않는다. ARI-8B(EXP-019) 관련 수치는 Top-K 규칙 freeze와 EXP-015 대비 paired 비교 전까지 "성능이 좋다/나쁘다"로 서술하지 않는다(DECISIONS [015]).**

**완료 조건:** 슬라이드의 각 수치와 주장이 실험 ID 또는 데이터 근거에 연결되고, 한계가 누락되지 않는다.

## 4. PM·Git 담당

**책임:** 작업 범위, 브랜치, 산출물 보존, 결정 기록을 관리한다.

- 작업 전 `git status`, 현재 브랜치, 수정 허용 파일 확인
- 한 작업당 목표·입력·출력·완료 조건·금지 작업 명시
- AI 작업 후 변경 파일, 실행 명령, 검증 결과를 보고받음
- 모델·예측 파일의 외부 백업 위치와 SHA-256 기록
- `backup/cursor-local-ahead`, stash, legacy 산출물을 파일 단위로 검토
- 중요한 결정은 `DECISIONS.md`, 실험은 `EXPERIMENT_LOG.md`에 기록

**완료 조건:** 최신 상태를 한 문서에서 확인할 수 있고, 어느 AI나 팀원이 이어받아도 브랜치와 다음 작업을 혼동하지 않는다.

## 5. 현재 작업 순서

1. ~~EXP-018 018A/018B post-hoc Exact@5 공식 평가~~ — **완료(2026-09-17).** 018A Δ−1.20pp(gate 미통과), 018B Δ+0.70pp이나 McNemar p=0.350642(비유의, gate 미통과). 공식 후보 승격 없음, EXP-015 유지.
2. **ARI Top-K constrained decoder 구현**(`docs/ARI_TOPK_DESIGN.md` 기준) + GPU 없이 가능한 unit test로 후보 생성 로직 검증(EXP-019A 준비) — **최우선.**
3. **cross-split exact duplicate 26건 원인 확인** → BM25 retrieval corpus leakage audit 설계·실행(EXP-019B 전제조건, DECISIONS [015]).
4. B1/B2/Lmax sensitivity 확인 후 Top-K 규칙 freeze → **EXP-019A full validation** 실행, EXP-015와 paired 비교.
5. `drive_loss_vs_docs.csv` 기준 EXP-018 018A/018B loss 표 정정 수치 확인 및 `EXPERIMENT_LOG.md` 갱신(TODO).
6. **(GrokBot 위임 검토 대상, DECISIONS [016]):** EXP-018 관련 GPU 재학습 모니터링·백업, 논문·2차 원천 조사, EXP-019 하이퍼파라미터 반복 실행 — 위임 시 AI 모델 담당이 사전 범위 승인·사후 검증을 담당한다.
7. 발표·보고서 수치 정합 재확인 — full-test는 EXP-012, valid 기준선은 EXP-015로 통일(EXP-018은 gate 미통과로 미채택 상태임을 명시).
8. (시간이 남으면) EXP-018C(lr=2e-5) 실행, 전문가 사용성 실험(`reports/expert_panel/`), EXP-012 `real_defect` 전수 예측 보관, (선택) EXP-006/007 local CUDA 재실행.

`model_version` 고정 태그: `exp012_sillokbert_gpu_best_valid_loss`

## 6. 공통 규칙

- 원본 데이터와 기존 실험 산출물을 임의로 수정·삭제·이름 변경하지 않는다.
- 기존 사용자 변경을 덮어쓰지 않는다.
- test는 최종 평가용으로 제한하고 모델 선택은 validation으로 한다.
- 실행하지 않은 작업을 완료로 표시하지 않는다.
- CPU subsample 결과와 GPU full 결과를 명확히 구분한다.
- 막히면 추측으로 진행하지 말고 작은 검증 실험 또는 사람의 결정을 요청한다.
- 자율 에이전트(GrokBot 등)에게 위임한 작업도 위 규칙을 동일하게 적용한다(AI_COUNCIL.md §7, DECISIONS [016]).

---

*관련 문서: `PROJECT_OVERVIEW.md`, `DATA_PREPROCESSING_RULES.md`, `RISK_CHECKLIST.md`, `AI_COUNCIL.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`.*
