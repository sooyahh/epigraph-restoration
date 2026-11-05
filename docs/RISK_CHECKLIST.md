# RISK_CHECKLIST.md — 프로젝트 위험 점검표

중요한 데이터 생성, 학습, 모델 선택, 발표·제출 전에 확인한다. 질문에 **예 또는 모름**이 나오면 대응을 완료하기 전 결과를 확정하지 않는다. 마지막 갱신일은 **2026-09-17**이다.

## 1. 데이터 누수와 문맥 중복

**점검**

- 같은 비문이 train/valid/test에 걸쳐 있는가?
- 정답 글자가 입력 또는 파생 feature에 남아 있는가?
- test 문맥이 train의 정형구와 과도하게 겹치는가?
- 한 원문의 여러 마스크 샘플이 split을 넘어갔는가?
- title_key 붕괴로 test split 내 near-duplicate 집중이 있는가?
  (동일·유사 텍스트가 여러 샘플로 반복되어 유효 표본 크기를 부풀리는가)
- retrieval(BM25/ChromaDB) corpus를 train-only index로 구성해도, 서로 다른 `inscription_id` 사이의 cross-split exact duplicate로 인해 leakage가 없다고 단정할 수 있는가?

**대응**

- inscription 단위 split 교집합을 자동 검사한다.
- 작은 예시로 입력과 label을 손검증한다.
- n-gram 중복 구간과 비중을 보고하고 중복이 낮은 subset 성능도 확인한다.
- 현재 EXP-009의 2-gram 이상 중복 59.5%를 결과 해석에 포함한다.
- DECISIONS [013] 참조. 587건 중복 연결 샘플의 split별 분포와
  (title_key, text_normalized) 쌍당 반복 건수를 확인 전까지는
  희귀 구간(hapax/rare) 지표의 정밀도를 과신하지 않는다.
- EXP-019 저장소 감사에서 확인된 cross-split exact duplicate 26건(train/valid 간 최대 약 405자 장문 포함)을 DECISIONS [013]의 title_key 붕괴와 같은 원인인지 별개 현상인지 먼저 확인한다. BM25(EXP-019B)·ChromaDB/dense(EXP-019C) retrieval 실험은 exact/near-duplicate leakage audit을 통과하기 전에는 공식 결과로 보고하지 않는다(DECISIONS [015] 게이트).

## 2. test 반복 사용

**점검**

- test 결과를 보고 모델이나 하이퍼파라미터를 바꾸었는가?
- test를 오류 분석과 체크포인트 선택에 사용했는가?
- 동일 test 평가 횟수가 기록되어 있는가?
- EXP-012 이후 **잔여 full-test 예산(2회)** 을 초과하지 않았는가? (DECISIONS [012])

**대응**

- 앞으로 모델·epoch·learning rate 선택은 validation으로만 한다.
- 일상 개선 판단은 **valid Exact@k**(iteration: subsample n=2,000 + paired; 마일스톤: full valid)로 한다.
- test는 최종 확정 모델에 제한적으로 사용한다. 사용 시 목적·후보 고정·누적 횟수를 EXPERIMENT_LOG에 남긴다.
- 기존 test 기반 의사결정은 숨기지 말고 간접적인 test 적응 가능성을 한계로 기록한다.
- **신규 모델(ARI-8B 등) 1차 스크리닝에도 test를 사용하지 않는다 — EXP-015 valid 기준선으로만 비교한다(DECISIONS [014]).**
- **(2026-09-17 확인) EXP-018 018A/018B post-hoc 평가도 test를 사용하지 않고 valid subsample(n=2,000)로만 수행했다 — 원칙이 실제로 지켜졌다.**

## 3. 데이터 품질과 대표성

**점검**

- 현재 corpus를 전수 검수 완료 데이터라고 과장하고 있지 않은가?
- 한글 혼입, 주석, 구조 표기, OCR 오류가 학습에 남아 있는가?
- 시대·유형·길이·비문별 샘플 수가 한쪽으로 쏠리는가?
- 결손이 많은 자료를 인공 마스크 평가와 혼합했는가?

**대응**

- 현재 데이터는 `1차 가공본`으로 표현한다.
- `is_training_text`, `quality_flag`, `□`, `◇` 통계를 검증한다.
- 비문별 샘플 수와 시대·길이 분포를 보고한다.
- 실제 결손 데이터는 정성 데모로 분리한다.

## 4. tokenizer와 truncation

**점검**

- 정답이 UNK인데 오답으로만 집계하고 원인을 숨기고 있지 않은가?
- 장문 truncation으로 마스크 위치가 사라지거나 잘못 이동하는가?
- 문자 위치와 token 위치를 혼동하는가?
- ARI 계열(Qwen tokenizer 등) 후보 생성에서 multi-token/byte 경로, invalid Hanja output, JSON parsing 실패가 BERT의 single-token mask와 다른 방식으로 오답 처리되고 있지 않은가?

**대응**

- UNK 포함/제외 성능을 함께 보고하되 주 지표 정의는 유지한다.
- 최장 문서와 경계 위치에 대한 단위 테스트를 추가한다.
- 원문 위치와 tokenized 위치의 매핑을 검증한다.
- ARI Top-K 후보는 `torch.topk(next_token_logits, k)` 같은 단순 근사 대신, 유효 한자 완성 여부를 확인하는 constrained decoding으로 생성한다(DECISIONS [015], EXP-019).

## 5. 과적합과 불공정 비교

**점검**

- train loss만 보고 최종 체크포인트를 선택하는가?
- 비교 모델의 데이터, 문맥 길이, top-k 처리, 평가 표본이 다른가?
- seed 하나의 결과를 일반화하고 있는가?
- subsample 결과를 full-data 결과와 직접 비교하는가?
- **"학습(GPU 4-epoch) 완료"를 "Exact@k 평가 완료"와 같은 의미로 서술하고 있지 않은가?** (EXP-018 018A/018B가 실제로 이 함정에 해당했다 — 2026-08-30 시점에는 loss만 확보된 상태를 성능 개선으로 인용하면 안 됐다. **2026-09-17부로 Exact@k 평가는 완료됐지만, 이번에는 반대로 "Exact@k가 있으니 채택 가능"으로 서술하지 않도록 주의한다 — gate를 통과하지 못했으므로 여전히 공식 후보가 아니다.**)
- **(2026-09-17 추가) 표본 크기가 다른 두 수치(예: full-valid 53.24% vs subsample 54.10%)를 직접 빼서 "paired 개선량"처럼 서술하고 있지 않은가?**

**대응**

- validation loss·Exact@k와 early stopping을 사용한다.
- **공식 epoch 채택**은 학습 후 저장된 전 epoch에 대한 **valid Exact@5**(동점이면 valid_loss·낮은 epoch)로 한다 (DECISIONS [012]). `checkpoint-epoch-*`를 전부 백업한다.
- 비교표에 데이터·표본·seed·문맥 창·device를 함께 쓴다.
- 가능하면 여러 seed 또는 최소한 seed 고정과 한계를 명시한다.
- EXP-011은 CPU 1 epoch·train 약 8천·test 2천 subsample로만 표기한다.
- 두 모델 Exact@k 비교는 독립 비율 가정이 아니라 **paired**(McNemar)를 쓴다 (`src/eval/paired_exact_compare.py`).
- **(2026-09-17 갱신) EXP-018 018A/018B는 이제 post-hoc valid Exact@k(subsample n=2,000)를 확보했다. DECISIONS [012]의 gate(EXP-015 +0.7pp, McNemar p<0.05)를 018A/018B 모두 통과하지 못했으므로, "부분 완료"가 아니라 "평가 완료·gate 미통과"로 정확히 표기하고 공식 후보로 승격하지 않는다.**
- paired 비교는 반드시 **동일 표본**(같은 subsample 또는 같은 full set) 기준으로만 delta를 계산한다. full-valid 수치와 subsample 수치를 직접 빼지 않는다.

## 6. 평가 방식 오류

**점검**

- `Exact@k` 구현이 top-k 안의 정답 포함 여부로 계산되는가?
- 후보 중복, UNK, 이체자 처리가 모델별로 다른가?
- `Group@k`의 매핑 출처가 확정됐는가?
- 인공 마스크 성능을 실제 결손 정확도라고 부르는가?
- ARI-8B의 raw score(token conditional logprob 평균)를 BERT의 mask softmax와 같은 척도로 직접 비교하고 있지 않은가? (순서만 Exact@k에 쓰고 척도 자체는 비교하지 않는다 — EXP-019 설계 원칙.)

**대응**

- 장난감 예제로 지표를 손계산해 단위 테스트한다.
- 후보 정규화·중복 처리 규칙을 고정한다.
- 이체자 매핑 미확정 상태에서는 `Group@k`를 공식 수치로 쓰지 않는다.
- 실제 결손은 전문가 정성평가 또는 별도 gold set이 생긴 뒤 평가한다.
- ARI 등 새 모델 계열의 후보 생성 규칙은 PAPER/PUBLIC/OURS/UNKNOWN으로 출처를 태깅해 문서화한다(DECISIONS [015]).

## 7. 재현성과 산출물 소실

**점검**

- 모델·예측·검증 파일이 한 PC에만 있는가?
- Git 밖의 파일에 위치·크기·SHA-256 기록이 없는가?
- 실행 코드, 환경, seed, 데이터 버전, 명령이 빠졌는가?
- 브랜치나 stash에만 존재하는 중요 파일이 있는가?

**대응**

- EXP-011·EXP-012·EXP-018 산출물을 PC 외 최소 한 곳에 백업한다.
- 모델 파일을 무조건 일반 Git에 넣지 말고 외부 저장소 또는 팀이 합의한 LFS를 사용한다.
- 문서에 백업 위치·체크섬·생성 실험을 기록한다.
- 브랜치·stash는 삭제 전에 파일 단위로 채택·보류를 결정한다.
- **(2026-09-17 해결) EXP-018 018A/018B GPU 학습 로그·체크포인트의 실제 저장 경로가 확인됐다 — Google Drive `/content/drive/MyDrive/exp018/018a/`, `018b/`, 각 `checkpoint-epoch-1`~`4`에 `model.safetensors`+tokenizer 존재. 기존 "저장 위치 TODO"는 해소됐다. 단, PC/Git 외부 추가 백업 여부는 별도 확인이 남아 있다.**
- **(2026-09-17 추가) `drive_loss_vs_docs.csv` 대조 결과 EXPERIMENT_LOG의 018A/018B loss 표 일부가 Drive 실제 값과 다를 수 있음이 확인됐다. 구체 정정 수치를 확보해 표를 갱신하고, 정정 사유를 문서에 남긴다(과거 기록을 조용히 덮어쓰지 않는다) — 아직 TODO.**

## 8. 구현 범위와 설명 가능성

**점검**

- 일정 내 학습·디버깅·발표가 가능한가?
- baseline 대비 복잡도 증가가 실제 개선으로 검증됐는가?
- 팀원이 방법과 한계를 슬라이드 한 장으로 설명할 수 있는가?
- ARI-8B + retrieval(BM25/ChromaDB) 구조를 담당자 이탈 등 리소스 제약 안에서 critical path(019A→019B)만으로 발표 가능한 형태로 좁혔는가?

**대응**

- SillokBERT full fine-tuning과 평가를 먼저 완성한다.
- 이체자, glyph, 앙상블은 baseline 안정화 뒤 추가한다.
- 설명할 수 없거나 재현되지 않은 방법은 최종 채택하지 않는다.
- 019C(ChromaDB/dense retrieval)는 deferred/optional로 명시하고, 완료되지 않은 부분을 발표에서 숨기지 않는다(DECISIONS [015]).
- EXP-018처럼 "시도했지만 gate를 통과하지 못한" 결과도 실패 사례로서 발표에 포함할 수 있다 — 숨기지 않는다.

## 9. 제출·발표 직전 1분 점검

- [ ] 모든 수치에 실험 ID, 표본 수, 데이터 버전이 있다.
- [ ] EXP-011의 50.45%를 최종 full-test 성능으로 표현하지 않았다.
- [ ] 모델 선택은 validation, 최종 평가는 test라는 원칙을 지켰다.
- [ ] 실제 결손과 인공 마스크 평가를 구분했다.
- [ ] n-gram 중복과 UNK 한계를 공개했다.
- [ ] 전수 수동 검수 미완료를 숨기지 않았다.
- [ ] 모델·예측 산출물이 외부에 백업되어 있다.
- [ ] 발표에 사용한 파일이 공식 브랜치의 검토된 버전이다.
- [ ] **EXP-018을 "GPU 학습 완료"와 "Exact@k 성능 개선"을 혼동해 서술하지 않았다. (2026-09-17: Exact@k는 확보됐으나 gate 미통과 → "성능 개선 성공"으로 서술하지 않는다.)**
- [ ] **cross-split exact duplicate 26건과 leakage audit 필요성을 발표·문서에서 숨기지 않았다.**
- [ ] **ARI-8B·retrieval 관련 수치가 없는데 "성능이 좋다/나쁘다"로 서술하지 않았다(DECISIONS [015]).**
- [ ] **(2026-09-17 추가) full-valid 수치와 subsample 수치를 직접 빼서 paired 개선량처럼 서술하지 않았다(EXP-018B +0.86pp 자동계산 오해 사례 참고).**

---

*관련 문서: `PROJECT_OVERVIEW.md`, `DATA_PREPROCESSING_RULES.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md`.*
