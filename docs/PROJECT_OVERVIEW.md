# PROJECT_OVERVIEW.md — 프로젝트 개요

이 문서는 프로젝트의 현재 큰 그림과 다음 의사결정 지점을 설명한다. 수치와 완료 상태는 `EXPERIMENT_LOG.md`와 `DECISIONS.md`를 근거로 하며, 마지막 상태 정합화 날짜는 **2026-09-17**이다.

> **현재 상태:** NRICH 금석문 XLS 기반 전처리 v4, inscription 단위 split, 비딥러닝·zero-shot baseline, SillokBERT CPU subsample(EXP-011)과 **GPU full-data fine-tuning + full test(EXP-012)** 를 거쳐, **valid Exact@k 기준선(EXP-015)** 을 확정하고 이후 FT 개선 루프(EXP-016 형식 정합, EXP-017 ZS/FT 보간 — 기각)를 진행했다. **EXP-018(SillokBERT lr 재탐색)은 018A/018B GPU 학습과 post-hoc valid Exact@k 평가(2026-09-17)까지 완료됐으나, 둘 다 DECISIONS [012] gate를 통과하지 못해 공식 후보로 승격되지 않았다 — EXP-015가 계속 공식 valid 기준선이다.** 현재 **최우선 작업은 EXP-019(ARI-8B 통합)** — Top-K 평가 설계와 저장소 leakage 감사까지 마쳤고 실제 구현·GPU 실행·성능 수치는 아직 없다. 전수 수동 검수를 마친 공식 corpus는 아직 없다. 동일 test n=9,719 공식 비교 수치는 EXP-012이며, **일상 비교·신규 모델 비교의 공식 valid 기준선은 EXP-015**다(DECISIONS [014]).

## 1. 프로젝트 목표

훼손되어 읽을 수 없는 금석문의 결손 한자 위치마다 문맥상 가능한 **상위 k개 후보(top-k)** 를 제시해 전문가의 판독·검토를 보조한다.

- 입력: 결손 위치가 `□`로 표시된 금석문 판독문과 식별자 등 메타데이터
- 출력: 각 결손 위치의 한자 후보 top-k와 가능하면 후보별 점수
- 정량 평가: 정답을 아는 글자를 인공적으로 가린 데이터의 `Exact@1/5/10`
- 실제 활용 평가: 정답이 없는 실제 결손 사례에 대한 정성 검토

모델의 출력을 확정 판독으로 간주하지 않는다. 최종 판단은 전문가 또는 검수자가 수행한다.

## 2. 현재 진행 상태

| 단계 | 상태 | 근거·한계 |
|---|---|---|
| 문제 정의 | 완료 | 문자 단위 top-k 복원, DECISIONS [001] |
| 1차 데이터 수집 | 완료 | NRICH XLS 원본과 `readings.csv` |
| 전처리 | 완료 | `gsm_preprocess_v4`; 실제 결손 `□`, 한글 치환 `◇` |
| 전수 수동 검수 | 미완료 | 현재 corpus는 1차 가공본이며 공식 검수 완료 corpus로 부르지 않음 |
| split | 완료 | inscription 단위, `seed=42`; 동일 비문의 split 간 교집합 0건. **단, title_key 붕괴로 인한 동명 비문 병합 이슈(DECISIONS [013])와 cross-split exact duplicate 26건(EXP-019 감사)이 남아 있음** |
| 인공 마스크 데이터 | 완료 | 66,537개: train 46,139 / valid 10,679 / test 9,719 |
| 단순 baseline | 완료 | frequency, n-gram |
| zero-shot baseline | 완료 | `bert-base-chinese`, `ddokbaro/SillokBert`; full test 9,719건 |
| fine-tuning | 완료 | EXP-011: CPU 1 epoch subsample (test n=2,000). EXP-012: Colab GPU full train + full test n=9,719 (DECISIONS [010]) |
| GPU full 학습·평가 | 완료 | EXP-012 Exact@1 33.75% / @5 53.04% / @10 60.47% (test n=9,719) |
| FT 층화 분석 | 완료 | EXP-013: 이득이 고빈도 집중, hapax Exact@5 −2.2pp (DECISIONS [011]) |
| valid Exact@k 기준선 | 완료 | EXP-015: Exact@1 33.78% / @5 53.24% / @10 60.90% (n=10,679) — 이후 모든 비교의 incumbent baseline (DECISIONS [012], [014]) |
| 학습·평가 형식 정합 진단 | 완료 | EXP-016: 정합 확인, context window ±64 유지(±256은 유의 악화) |
| ZS/FT 확률 보간 | 완료·기각 | EXP-017: full valid에서 EXP-015 대비 유의 개선 없음(p=0.917) → FT 단독 유지 |
| SillokBERT lr 재탐색 | 완료(gate 미통과) | EXP-018: 018A/018B GPU 학습 및 post-hoc valid Exact@k 평가(2026-09-17) 완료. 018A Δ−1.20pp(gate 미통과), 018B Δ+0.70pp이나 McNemar p=0.35(비유의, gate 미통과). **공식 후보 승격 없음, EXP-015 유지.** 018C 미실행 |
| ARI-8B 신규 모델 통합 | 설계 완료 | EXP-019: Top-K constrained decoding 설계, 통합 가능성 감사, leakage 감사(26건 duplicate 발견) 완료. **구현·GPU 실행·성능 수치 없음 — 현재 최우선 작업** (DECISIONS [015]) |
| 실제 결손 데모 | 준비됨 | `real_defect_inference.csv`; 정답이 없으므로 정량 성능과 분리 |

## 3. 데이터와 기호

- 원본 데이터는 수정하지 않는다.
- 가공 레이어에서 실제 결손은 `□`, 한글 또는 한글 혼입 치환은 `◇`로 구분한다.
- 학습에는 현재 `is_training_text=True`인 판독문만 사용한다.
- 실제 결손이 포함된 자료는 인공 마스크 정량평가와 분리한다.
- 현재 `readings.csv`는 14,702행이며, 그중 학습 대상은 7,158행으로 기록되어 있다.
- 인공 마스크 데이터는 66,537개지만 독립 비문 수와 샘플 수는 서로 다른 개념이다. 숫자를 보고할 때 반드시 단위를 함께 쓴다.
- title_key 기반 동명 비문 매핑에서 중복 연결(마스크 샘플 587/66,537, 0.88%)이 확인됐다(DECISIONS [013]). retrieval 실험을 위한 corpus 구성 시에는 추가로 cross-split exact duplicate 26건(EXP-019 감사)도 감안해야 한다.

세부 규칙은 `DATA_PREPROCESSING_RULES.md`를 따른다.

## 4. 평가 원칙

1. 모델·epoch·learning rate·체크포인트 선택은 validation 결과로만 한다.
2. test 결과는 최종 비교에 제한적으로 사용하고, 사용 횟수와 목적을 실험 로그에 남긴다.
3. 인공 마스크 성능과 실제 결손 복원 능력을 동일시하지 않는다.
4. 전체 평균 외에 정답 빈도, 시대, 길이, UNK 여부, train과의 n-gram 중복 여부를 층화해 본다.
5. `Group@k`는 이체자 매핑 출처와 규칙이 확정되기 전에는 공식 지표로 보고하지 않는다.
6. FT/신규 모델의 일상 비교는 **EXP-015 valid Exact@k + paired 비교**로 하고, 새 모델 계열도 동일한 원칙을 적용한다(DECISIONS [012], [014]).
7. retrieval(BM25/dense)을 쓰는 실험은 corpus의 exact/near-duplicate leakage audit을 통과한 뒤에만 공식 결과로 보고한다(DECISIONS [015]).

## 5. 현재 모델 판단

- frequency와 n-gram은 파이프라인 및 하한선 확인용 baseline이다.
- zero-shot full test에서 SillokBERT가 `bert-base-chinese`보다 높아 1차 fine-tuning 대상으로 채택됐다.
- EXP-011은 동일한 2,000개 test subsample에서 SillokBERT zero-shot보다 개선됐으나, CPU 1 epoch·부분 데이터 결과다.
- EXP-011의 `Exact@5=50.45%`는 **최종 또는 공식 full-test 성능이 아니라 1차 subsample baseline**이다.
- **공식 full-test 비교 수치**는 EXP-012(Colab GPU, n=9,719): Exact@1 33.75% / @5 53.04% / @10 60.47% (DECISIONS [010]).
- **공식 valid 비교 기준선**은 EXP-015(n=10,679): Exact@1 33.78% / @5 53.24% / @10 60.90% (DECISIONS [012], [014]).
- EXP-013 층화 분석 결과, fine-tuning 이득은 train 빈도에 단조 비례하고 hapax 구간은 오히려 zero-shot보다 낮다(8.2%→6.0%). "전 구간 개선"이라고 서술하지 않는다.
- EXP-017(ZS/FT 확률 선형 보간)은 최적 α(0.8)에서도 EXP-015 대비 유의한 개선이 없어 기각됐다. 공식 후보는 FT(EXP-015) 단독 유지.
- **EXP-018(lr 5e-5/3e-5 재탐색)은 2026-09-17 post-hoc valid Exact@k 평가(subsample n=2,000)까지 완료됐다. 018A(epoch 1)는 EXP-015 대비 Exact@5 −1.20pp로 열세, 018B(epoch 2)는 +0.70pp로 수치상 소폭 우세하나 McNemar p=0.350642로 통계적으로 유의하지 않다. 둘 다 DECISIONS [012] gate(≥+0.7pp AND p<0.05)를 충족하지 못해 공식 후보로 승격하지 않았다 — EXP-015를 계속 공식 valid baseline으로 유지한다.**
- EXP-019(ARI-8B)는 통합·평가 설계만 완료된 상태이며, 성능·GPU 실행 성공 여부에 대해 어떤 주장도 아직 할 수 없다. **다음 최우선 작업이다.**

## 6. 현재 가장 중요한 위험

1. EXP-011·EXP-012·EXP-018(018A/018B) 체크포인트와 상세 예측 파일이 Git 밖의 로컬 경로에만 있어 소실 위험이 있다. (2026-09-17: 018A/018B checkpoint의 Google Drive 저장 경로는 확인됐으나 PC/Git 외부 추가 백업은 별도 확인 필요.)
2. 모델 선택 과정에서 test 결과를 반복 참고해 test에 간접적으로 적응할 수 있다.
3. inscription split 이후에도 test의 59.5%가 train과 2-gram 이상 문맥을 공유해 정형구 효과가 남아 있다.
4. title_key 붕괴로 인한 동명 비문 병합(DECISIONS [013])과 cross-split exact duplicate 26건(EXP-019 감사)이 완전히 해소되지 않았다 — retrieval 실험 leakage 위험으로 이어진다.
5. 인공 마스크 성능이 실제 훼손 위치의 복원 성능을 대표하지 않을 수 있다.
6. tokenizer UNK인 정답은 해당 vocabulary 기반 모델이 직접 출력할 수 없다.
7. ARI-8B의 Top-K 후보 생성 규칙 일부(B1/B2/Lmax 등)는 공개 구현이 없어 자체 설계(OURS)했고 sensitivity 검증 전이다 — 성능 수치를 확정으로 보고하지 않는다(DECISIONS [015]).
8. 데이터 전수 수동 검수와 이용 조건의 최종 확인이 남아 있다.
9. **(2026-09-17 추가) EXP-018 018A/018B의 학습 loss 표 일부가 실제 Drive 산출물(`drive_loss_vs_docs.csv`)과 다를 수 있음이 확인됐으나, 구체 정정 수치는 아직 확보하지 못했다 — 정정 전까지 EXPERIMENT_LOG의 기존 loss 표는 잠정치로 취급한다.**

자세한 대응은 `RISK_CHECKLIST.md`를 따른다.

## 7. 다음 작업 순서

1. **`docs/ARI_TOPK_DESIGN.md` 기준 ARI Top-K constrained decoder를 구현하고 GPU 없이 가능한 unit test로 검증한다(EXP-019A 준비) — 최우선.**
2. cross-split exact duplicate 26건의 원인을 DECISIONS [013]과 대조해 확인하고, BM25 retrieval corpus의 leakage audit을 설계·실행한다(EXP-019B 전제조건).
3. EXP-012·EXP-018 산출물(`notebooks/best/`, `outputs/finetune/exp018/`, Google Drive `exp018/018a·018b/`)을 PC 외부에 백업하고 크기·SHA-256·위치를 기록한다.
4. `drive_loss_vs_docs.csv` 기준 EXP-018 018A/018B loss 표의 실제 정정 수치를 확인해 `EXPERIMENT_LOG.md`를 갱신한다.
5. 발표·보고서 수치를 EXP-012(full-test)/EXP-015(valid)로 정합화하고, EXP-011 subsample·EXP-018(gate 미통과) 상태와 혼동하지 않는다.
6. 실제 결손(`real_defect_inference`) 정성 사례를 별도로 준비한다.
7. (후순위) 018C(lr=2e-5) 실행, B1/B2/Lmax sensitivity 확인 후 Top-K 규칙 freeze → EXP-019A full validation 실행.

## 8. 발표에서 강조할 내용

- 한국 금석문이라는 명확한 문화유산 도메인
- 확정 답을 강요하지 않고 전문가 검토용 top-k 후보를 제공하는 현실적인 목표
- 실제 결손과 인공 마스크 평가를 분리한 엄밀성
- inscription 단위 split과 층화 분석을 통한 과장 방지
- frequency → n-gram → 일반 중국어 BERT → 한국 고전 도메인 BERT → fine-tuning → (설계 중) 신규 LLM 비교로 이어지는 비교 과정
- 최종 수치뿐 아니라 실패 조건(hapax 하락, 보간 기각, **lr 재탐색 018A/018B 모두 gate 미통과**), UNK, 정형구 중복, duplicate 감사, 데이터 검수 한계를 함께 공개하는 신뢰성
- ARI-8B 등 신규 모델을 성급히 "성능이 좋다"고 말하지 않고, 통제된 비교 설계(EXP-015 기준선·leakage audit)를 먼저 세운 과정 자체를 차별성으로 제시

---

*관련 문서: `DATA_PREPROCESSING_RULES.md`, `RISK_CHECKLIST.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md`, `ROLE_GUIDE.md`, `AI_COUNCIL.md`, `CURSOR_HANDOFF.md`.*
