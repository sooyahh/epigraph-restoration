# DECISIONS.md — 의사결정 기록

프로젝트의 중요한 결정을 여기에 남긴다. **기록되지 않은 결정은 없는 결정이다.**
나중에 "왜 그렇게 했지?"라는 질문에 이 문서로 답할 수 있어야 한다(발표 질의응답 대비).

작성 방법: 새 결정을 맨 위(최신순)에 추가한다. 아래 템플릿을 복사해서 채운다.

결정 상태는 `active`(현재 유효), `provisional`(추가 검증 전 임시 채택), `superseded`(후속 결정으로 대체), `completed`(일회성 조치 완료), `rejected` 중 하나로 표시한다. 과거 결정을 삭제하지 않고 대체 관계를 남긴다.

---

## 템플릿 (복사해서 사용)

```
### [번호] 제목 (YYYY-MM-DD)

- 상태: (`active`/`provisional`/`superseded`/`completed`/`rejected`)
- 결정 내용: (무엇을 하기로 했는가, 한 문장)
- 배경/문제: (왜 이 결정을 해야 했는가)
- 대안: (고려한 다른 선택지들)
- 선택 이유: (왜 이 안을 골랐는가)
- 리스크: (이 선택의 위험·한계)
- 검증 방법: (이 결정이 옳았는지 어떻게 확인할 것인가)
- 결정자/참여: (사람 + 참고한 AI)
- 관련 링크: (실험 ID, 문서 등)
```

---

## 결정 기록 (최신순)

### [018] EXP-029 기록 · 발표 S5 불변 · 다음 축은 EXP-030+ (2026-10-01)

- 상태: `active`

- 결정 내용:
  1. EXP-029를 `EXPERIMENT_LOG`에 남긴다. 방법은 one-pass KV cache + `expand_character_candidates`의 **mean token logprob** Top-K(Appendix C–style)다. **공식 ARI Top-K가 아니다.** adapter는 `MyDrive/exp024/full/adapter_final`. 스크립트는 `src/eval/ari/score_exp029.py`(PR #30, `exp024/ari-qlora`).
  2. **발표 S5는 바꾸지 않는다** (`update_presentation_s5 = false`). 잠금 숫자는 DECISIONS [017] 그대로다. Sillok **0.515** / ARI-FT beam5 **0.410** / ZS **0.225** (EXP-025, 동일 n=200).
  3. EXP-029 n=2000 Exact@5 **0.4115**를 S5의 **0.410**과 같게 읽지 않는다. paired Sillok Exact@5 **0.534**(n=2000)를 S5의 **0.515**와 같게 읽지 않는다.
  4. EXP-027 야간 full train은 별개이며 아직 돌고 있을 수 있다. 027 결과를 이 결정에 적지 않는다.
  5. 다음 실험 축은 Claude 검토 이후 **EXP-030+**다.

- 배경/문제:
  n=2000 one-pass Exact@5 0.4115가 S5의 beam5 0.410과 붙어 보여, 슬라이드 칸을 갈아끼울 위험이 있다. 표본(n=2000 vs n=200)과 점수 함수(mean token logprob vs beam5 sum)가 다르다. EXP-027은 아직 별도 학습이다.

- 대안:
  - 0.4115를 S5 ARI-FT 칸에 넣음 (기각 — 프로토콜·표본이 다르다)
  - Sillok 0.534를 S5의 0.515 자리에 넣음 (기각 — n=2000 paired이지 EXP-025 n=200이 아니다)
  - EXP-029를 공식 ARI Appendix C Top-K로 표기 (기각 — Appendix C–style이며 공식 재현이 아니다)
  - EXP-027이 끝났다고 적음 (기각 — 별개 학습이고 결과를 받지 않았다)

- 선택 이유:
  [017]이 S5의 정본이다. EXP-029는 로그용 평가다. 비슷한 소수점(0.4115 vs 0.410)을 같은 숫자로 읽지 않도록 잠금을 한 번 더 적는다.

- 리스크:
  - 0.4115와 0.410, 0.534와 0.515를 한 칸으로 말할 수 있다. 표본 크기와 점수 함수를 붙인다.
  - 027이 끝나도 그 숫자로 S5를 바꾸지 않는다. 027 지표는 완료 산출물이 생긴 뒤에만 별도 절에 적는다.

- 검증 방법:
  S5 문장에 Sillok 0.515 / ARI-FT beam5 0.410 / ZS 0.225(n=200)가 그대로인지 확인한다. EXP-029 절에 `update_presentation_s5 = false`, 점수 함수 이름 **mean token logprob**, "공식 ARI Top-K 아님"이 있는지 확인한다. 027 Exact@k가 비어 있으면 이 결정을 지킨 것이다.

- 결정자/참여: 팀(sooyah). 문서 반영은 Cursor. 다음 축(EXP-030+)은 Claude 이후.

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-029
  `docs/DECISIONS.md` [017]
  PR #30 (`exp024/ari-qlora`, `src/eval/ari/score_exp029.py`)
  Drive: `MyDrive/exp024/full/adapter_final`, `MyDrive/exp029/`

### [017] 발표 Exact@k 고정(EXP-025 근사 Exact) · 실험 순서 028→027 · 비공식 야간 학습 (2026-10-01)

- 상태: `active`

- 결정 내용:
  1. 발표 S5의 Exact@k는 **EXP-025 동일 n=200 근사 Exact**로 고정한다. 공식 ARI Appendix C Top-K도, SOTA도 아니다.
     - Exact@5: Sillok **0.515** > ARI-FT **0.410** > ZS **0.225**. McNemar는 양쪽(Sillok ≻ FT, FT ≻ ZS) 모두 유의하다.
     - Exact@1: Sillok **0.350** / FT **0.220** / ZS **0.105**.
     - Exact@10: Sillok **0.575**. ARI는 **beam5**라 Exact@10이 Exact@5와 같다(FT 0.410, ZS 0.225). 슬라이드에는 **ARI Exact@10 = n.a. (beam5)** 라고 적고, 슬라이드용 beam10 재평가는 하지 않는다.
  2. **금지:** EXP-015 n≈500 Exact@5 **0.504**를 EXP-024 **0.410**과 직접 비교하는 것. 공식 ARI Top-K / Appendix C / SOTA라고 말하는 것.
  3. P1 **EXP-026은 건너뛴다.** 순서는 **EXP-028 정성 평가를 먼저** 하고, 이어서 **EXP-027** 야간 full train이다. **EXP-023은 이미 완료**됐으므로 재실행하지 않는다.
  4. EXP-027·028의 수치는 **S5/S10 슬라이드에 넣지 않는다.** 028은 real_defect 정성 데모이며 Exact@k가 없다. 027은 비공식이다. 해석은 미리 정한다. Sillok Exact@5 0.515보다 낮으면 Sillok 쪽을 보강하는 결과로 읽고, 거의 같으면 동등, 더 높으면 **비공식 참고만** 한다. **2026-10-02 오전까지 끝나지 않으면 중단**한다.
  5. 데모 슬라이드는 사례 선정 전 placeholder였다. **2026-10-01에 검증 사례를 고정**했다. 028은 "복원 성공"이 아니라 **전문가 검토용 불확실 후보**로 말한다.
     - artificial_mask (5). 성공: `mask_00014815` (乙亥年八[MASK]→月), `mask_00003798` (引[MASK]生→門), `mask_00001848` (智[MASK]山→異). 실패: `mask_00005787` (檢校[MASK]子, gold 太 vs 世), `mask_00056976` (大[MASK], gold 干 vs 師).
     - EXP-028 정성 (2, gold 없음): `real_000275` 歲[MASK]戌年 — Sillok 戊 / ARI-FT 甲 (Top-1 불일치). `real_000445` 向[MASK]下二里 — Sillok 山 / ARI-FT 此 (山은 Top-5에 겹침).
     - 028 side-by-side n=25에서 **Top-1 완전 일치 = 0**이므로 "두 모델이 같다"는 데모 사례는 없다.
  6. 저장소 흐름은 그대로다. **Cursor가 commit/push**하고, **GitHub 팀원이 review/merge**한다. Colab 산출물은 Drive에 둔다.
  7. EXP-024 A100 full: train_loss **0.4577**. adapter `MyDrive/exp024/full/adapter_final`. eval n=200 beam5 근사 Exact@1 **0.220** / @5 **0.410**. 같은 표본 EXP-021 ZS 0.105 / 0.225 대비 paired **+11.5 / +18.5 pp**. 공식 ARI Top-K가 아니다.
  8. EXP-027은 `--train-pool -1`이 필요하다(PR #25, `exp024/ari-qlora`에 병합). EXP-028 import 수정은 PR #26(`unique_hanja_candidates`를 `topk`가 아니라 `parse`에서 가져옴)이며 같은 브랜치에 병합됐다.

- 배경/문제:
  발표 슬라이드에 올릴 Exact@k와, 야간에 도는 ARI QLoRA(EXP-027)·정성 데모(EXP-028)를 어디까지 말할지가 겹쳐 있었다. ARI 후보는 beam5 `generate()` 근사라 DECISIONS [015]의 Appendix C Top-K가 아니다. 표본 크기가 다른 Exact@5(EXP-015 n≈500의 0.504와 EXP-024 n=200의 0.410)를 한 줄에 놓으면 비교가 왜곡된다. 027이 아침까지 끝나지 않아도 S5/S10 숫자를 바꾸지 않도록, 해석 규칙과 중단 조건을 미리 고정한다.

- 대안:
  - EXP-024 0.410을 EXP-015 n≈500 Exact@5 0.504와 직접 비교 (기각 — 표본이 다르다)
  - beam10을 다시 돌려 ARI Exact@10을 슬라이드에 채움 (기각 — 슬라이드는 ARI Exact@10 = n.a. (beam5))
  - 공식 Appendix C 재현 또는 SOTA로 표기 (기각 — 이번 수치는 근사 Exact)
  - EXP-026을 P1로 진행하거나 EXP-023을 다시 돌림 (기각 — 026은 건너뛰고, 023은 이미 완료)
  - 027/028 수치를 S5/S10에 반영 (기각 — 028은 Exact@k가 없는 정성 평가, 027은 비공식)
  - 데모 문장을 "복원 성공"으로 씀 (기각 — 전문가 검토용 불확실 후보. n=25에서 Top-1 완전 일치가 0)

- 선택 이유:
  동일 n=200 paired(EXP-025)만 표본과 지표 정의가 맞아 S5에 쓸 수 있다. McNemar가 양쪽 비교에서 유의하므로, 순위(Sillok > ARI-FT > ZS)는 그 표본에 한정해 말한다. 027은 해석만 미리 정해 두고 슬라이드 숫자는 움직이지 않는다. 데모는 2026-10-01에 고른 인공 마스크 5건과 real_defect 2건으로 고정한다.

- 리스크:
  - 근사 Exact를 공식 Top-K로 읽을 수 있다. 슬라이드와 구두 모두 "근사, beam5, n=200, SOTA 아님"을 붙인다.
  - ARI Exact@10을 0.410 / 0.225로 적으면 beam10을 돈 것처럼 보인다. 슬라이드는 n.a.로 둔다.
  - 027이 Sillok 0.515보다 높아도 비공식이다. 그 숫자로 S5를 갈아끼우지 않는다.
  - 027이 2026-10-02 오전까지 미완이면 중단한다. 중간 loss나 미완 체크포인트를 성능으로 적지 않는다.
  - 028은 gold가 없는 real_defect라 Exact@k가 없다. Top-1이 갈리거나 Top-5에만 겹치는 사례를 복원됐다고 말하면 안 된다.
  - 선정 메모 `/workspace/research-notes/2026-10-01_verification-examples.md`는 연구 환경 쪽 파일이며 이 저장소에는 없다.

- 검증 방법:
  S5 숫자는 `reports/exp025/`와 Drive `MyDrive/exp025/paired_exact_compare.json`의 n=200과 맞는지 확인한다. 027은 완료 시각·adapter 경로·요약이 생긴 뒤에만 로그에 지표를 적는다. 028 데모 문장은 side-by-side CSV의 해당 행과 맞는지 확인한다. 슬라이드에 027/028 Exact@k, ARI Exact@10 숫자, Appendix C 재현, SOTA가 없으면 이 결정을 지킨 것이다.

- 결정자/참여: 팀(sooyah). 문서 반영은 Cursor.

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-021, EXP-023, EXP-024, EXP-025, EXP-026, EXP-027, EXP-028
  `docs/DECISIONS.md` [014], [015]
  PR #24 (`main`, EXP-025만), PR #25·#26 (`exp024/ari-qlora`에 병합)
  Drive: `MyDrive/exp024/full/adapter_final`, `MyDrive/exp025/paired_exact_compare.json`, `MyDrive/exp027`, `MyDrive/exp028/exp028_sillok_vs_ari_ft_side_by_side.csv`
  연구 메모(저장소 밖): `/workspace/research-notes/2026-10-01_verification-examples.md`

### [016] AI Council에 자율 에이전트(GrokBot) 도입 — EXP-018 모니터링·백업 / 논문·2차 원천 조사 / EXP-019 하이퍼파라미터 반복 3개 용도로 한정 (2026-09-17)

- 상태: `provisional`

- 결정 내용:
  1. AI Council에 xAI GrokBot(자체 클라우드 컴퓨터에서 여러 앱에 로그인해 다단계 작업을 자율 실행하는 에이전트)을 새 작업 환경으로 추가한다.
  2. 사용 범위는 다음 3가지로 한정한다: (a) EXP-018 GPU 학습 모니터링·체크포인트 백업, (b) 논문·2차 원천 조사 자동화, (c) EXP-019 ARI-8B 하이퍼파라미터(B1/B2/Lmax 등) 반복 실행.
  3. git add/commit/push, `data/raw_layer/` 등 원본 데이터 쓰기, ARI Top-K 설계 규칙(`docs/ARI_TOPK_DESIGN.md`) 자체의 변경은 어떤 경우에도 GrokBot에 위임하지 않는다.
  4. GrokBot의 작업 완료 선언은 그대로 신뢰하지 않는다. 실제 산출물(체크포인트 경로+SHA-256, 실행 로그, 결과 CSV, 조사 자료의 출처 링크+수집일자)로 재확인한 뒤에만 `EXPERIMENT_LOG.md`/`DECISIONS.md`에 반영한다.
  5. EXP-019 하이퍼파라미터 반복 스윕은 사람이 최초 1개 조합의 성공 실행을 직접 확인한 뒤에만 GrokBot에 나머지 반복 실행을 위임한다.

- 배경/문제:
  GPU 장시간 학습 모니터링, 논문/2차 원천 조사, 반복적 하이퍼파라미터 실행에 드는 사람 시간을 줄이기 위해 xAI의 자율 에이전트 GrokBot(https://x.ai/bot) 사용을 검토했다. GrokBot은 기존 AI Council 도구(GPT/Claude/Gemini/Codex/Cursor)와 달리 사람의 매 턴 확인 없이 여러 앱에 로그인해 장시간 자율로 작업을 이어간다는 점에서, 기존 "완료 판정" 기준(`AI_COUNCIL.md` §5)을 그대로 적용하되 추가 안전장치가 필요하다고 판단했다.

- 대안:
  - GrokBot을 전면 도입해 코드 구현·git 작업까지 위임 (기각 — 자율 에이전트의 git/원본 데이터 접근은 검증 사례 없이 위험이 너무 큼)
  - GrokBot 도입을 전면 보류 (기각 — GPU 모니터링·문헌조사 반복 작업의 사람 시간 절감 효과를 포기하게 됨)
  - 3개 용도로 한정하고, git/원본 데이터 접근 금지 + 완료 선언 재검증 의무를 부여해 도입 (채택)

- 선택 이유:
  세 용도 모두 "실행 결과가 명확한 파일·로그·표로 남고, 실패해도 기존 산출물을 훼손하지 않는" 작업이라 `AI_COUNCIL.md`의 완료 판정 기준(실제 산출물 존재 확인)을 그대로 적용할 수 있다. 반대로 코드 리팩터링이나 git 작업은 자율 에이전트의 실수가 기존 산출물을 되돌릴 수 없이 훼손할 위험이 있어 제외했다.

- 리스크:
  - GrokBot이 컴퓨트 예산(시간·비용)을 초과 사용할 수 있다 — 장시간 실행 전 승인 절차 없이는 실행하지 않도록 프롬프트에 명시했다(`PROMPTS.md` GrokBot-A).
  - 논문/2차 원천 조사 결과에 출처·수집일자가 없으면 신뢰할 수 없다 — `PROMPTS.md` GrokBot-B 템플릿의 완료 조건으로 강제한다.
  - "완료했다"는 자기 보고와 실제 완료 사이의 간극은 아직 실사용 사례가 없어 미검증이다 — 초기 사용은 사람이 전 과정을 밀착 확인한다.

- 검증 방법:
  첫 EXP-018 GPU 모니터링 위임 또는 EXP-019 하이퍼파라미터 반복 위임이 실제 산출물(체크포인트/로그/CSV)로 재현 확인되면 상태를 `active`로 승격한다. 원본 데이터 접근 시도나 승인 없는 장시간 실행 등 사고가 발생하면 즉시 사용 범위를 재검토한다.

- 결정자/참여: 팀(sooyah) + Claude(가드레일 문서화)

- 관련 링크:
  `docs/AI_COUNCIL.md` §2, §7
  `docs/ROLE_GUIDE.md` (AI 모델 담당, 현재 작업 순서 5번)
  `docs/PROMPTS.md` §6 (GrokBot-A/B/C)
  https://x.ai/bot

### [015] ARI-8B 통합 평가 설계: EXP-015 고정 비교 · Top-K 자체 설계 채택 · 019C(ChromaDB) 보류 및 leakage audit 게이트 (2026-09-17)

- 상태: `active`

- 결정 내용:
  1. ARI-8B 비교 실험은 **EXP-019A(no-retrieval) → EXP-019B(BM25) → EXP-015와 동일 Exact@k 비교** 순서를 critical path로 고정한다.
  2. **EXP-019C(ChromaDB/dense retrieval)는 담당자 이탈로 인해 `deferred/optional`로 분류**하고, 019A/B 완료 전까지 우선순위에서 제외한다.
  3. ARI의 Top-5 후보 생성 규칙(constrained decoding: first-token branching → 미완 token 확장 → 한자 완성 확인 → 필터링·중복 제거 → 평균 logprob 정렬)은 공개 구현이 없어 우리가 직접 설계·고정한다. 문서에는 규칙별로 `PAPER`(논문 명시)/`PUBLIC`(공개 코드 존재)/`OURS`(우리 결정)/`UNKNOWN`(미확정) 태그를 반드시 표기한다.
  4. **EXP-019B(BM25) 및 향후 EXP-019C 진행 전에 반드시 retrieval corpus의 exact/near-duplicate leakage audit을 통과해야 한다**(게이트 조건). 감사 없이 retrieval 실험 수치를 공식 결과로 보고하지 않는다.
  5. 발표에서는 "Appendix C를 그대로 재현했다"고 서술하지 않고, "핵심 알고리즘 기반, 공개되지 않은 세부 규칙은 명시적으로 고정해 재구현했다"고 표현한다.

- 배경/문제:
  ARI-8B(`DAMI-Lab/ARI-8B`)를 기존 SillokBERT 평가 파이프라인에 통합하려는 과정에서, (a) ChromaDB/dense retrieval 담당자가 이탈해 해당 트랙을 무기한 진행할 수 없게 됐고, (b) ARI의 Top-K 후보 생성 방식이 BERT의 mask softmax와 근본적으로 달라 공개된 참조 구현이 없는 상태에서 순서 있는 서로 다른 한자 Top-5를 만드는 방법을 우리가 직접 정해야 했으며, (c) 저장소 감사 중 서로 다른 `inscription_id` 사이에서도 cross-split exact duplicate 원문이 26건 발견되어, retrieval을 train-only index로 구성해도 leakage가 없다고 단정할 수 없는 상황이 확인됐다.

- 대안:
  - 019A/B/C를 동시에 병행 진행 (기각 — 담당자 부재로 019C 실행 불가, 리소스 분산)
  - ARI 통합을 전면 보류하고 SillokBERT 개선(EXP-018)에만 집중 (기각 — 신규 모델 비교라는 대회 차별성 요소 상실)
  - Top-K 후보 생성 시 `generate()` 단순 반복 호출로 근사 (기각 — 순서 있는 서로 다른 한자 Top-5를 보장하지 못해 EXP-015와 공정 비교 불가)
  - critical path를 019A→019B로 좁히고 019C는 deferred, Top-K는 자체 설계하되 태그로 출처를 구분, retrieval 실험 전 leakage audit 게이트 부여 (채택)

- 선택 이유:
  담당자 이탈이라는 제약이 명확한 상황에서 전체를 멈추기보다 실행 가능한 범위(019A/B)로 scope를 좁히는 편이 "한 달 반 안에 구현 가능한 결과물" 원칙에 부합한다. Top-K 자체 설계는 불가피하지만, PAPER/PUBLIC/OURS/UNKNOWN 태깅으로 재현성·발표 설명 가능성을 확보할 수 있다. leakage audit 게이트는 RISK_CHECKLIST §1(데이터 누수)과 DECISIONS [013]에서 이미 확인된 title_key 중복 문제의 연장선에서, retrieval이라는 새로운 데이터 접근 경로가 열리기 전에 반드시 확인해야 하는 최소 안전장치다.

- 리스크:
  - Top-K 후보 생성 규칙(B1=32, B2=8, Lmax=4)이 전부 OURS이며 sensitivity 검증 전이므로, 이 값에서 나온 Exact@5를 ARI-8B의 "진짜" 성능으로 해석하면 안 된다.
  - 019C가 무기한 보류되면 "dense retrieval이 BM25보다 나은지"는 이번 대회 기간 내 결론 낼 수 없다 — 발표에서 이 부분은 한계로 명시해야 한다.
  - cross-split duplicate 26건의 성격(DECISIONS [013]과 동일 원인인지 별개 현상인지)이 아직 확인되지 않아, leakage audit 설계 자체가 재조정될 수 있다.
  - T4 16GB 4-bit inference의 실제 로딩·속도·OOM 여부가 미검증이라 019A 일정이 지연될 수 있다.

- 검증 방법:
  1. `ARI_TOPK_DESIGN.md` 기준 decoder 구현 후 GPU 없이 가능한 unit test로 후보 생성 로직을 먼저 검증한다.
  2. B1/B2/Lmax sensitivity를 측정해 Top-K 규칙을 freeze한 뒤에만 EXP-019A full validation을 공식 수치로 인정한다.
  3. EXP-019B 진행 전 retrieval corpus의 exact/near-duplicate audit 결과를 `EXPERIMENT_LOG.md`에 별도 기록하고, 감사를 통과한 경우에만 진행한다.
  4. EXP-019A/B의 Exact@1/5/10은 EXP-015와 동일 validation sample_id로 재현해 `paired_exact_compare.py`로 비교한다(DECISIONS [014] 원칙 적용).

- 결정자/참여: 팀(sooyah) + Cursor(저장소 감사·설계 문서 작성) + Claude(문서 정리·검증)

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-019
  `docs/DECISIONS.md` [014] (EXP-015 incumbent baseline 원칙)
  `docs/DECISIONS.md` [013] (title_key 충돌·중복 — 26건 duplicate와의 관계 확인 필요, 미확정)
  `docs/RISK_CHECKLIST.md` §1
  `docs/ARI_INTEGRATION_PLAN.md`, `docs/ARI_TOPK_DESIGN.md`

### [014] EXP-015를 신규 모델 비교의 공식 validation baseline(incumbent)으로 고정 (2026-08-30)

- 상태: `active`

- 결정 내용:
  향후 ARI-8B, retrieval/BM25, reranking 등 **신규 모델**을 비교할 때, **EXP-015**를
  SillokBERT **incumbent validation baseline**으로 고정한다.
  - **고정할 기준 (EXP-015):**
    - 기준 실험: `EXP-015`
    - 모델: `ddokbaro/SillokBert`
    - checkpoint: EXP-012 best checkpoint
    - model_version: `exp012_sillokbert_gpu_best_valid_loss`
    - 평가 데이터: `data/model_inputs/artificial_mask_dataset.csv`
    - split: `valid`
    - validation samples: **10,679**
    - split rule: inscription-level split
    - seed: `42`
    - context window: `±64`
    - Exact@1: **33.78%**
    - Exact@5: **53.24%**
    - Exact@10: **60.90%**
    - 주력 비교 지표: **Exact@5**
    - test: 신규 모델 **선택 단계에서는 사용하지 않음**
  - **앞으로의 비교 원칙:**
    - 신규 모델은 가능한 한 EXP-015와 **동일한 validation `sample_id`**를 사용한다.
    - Exact@k 정의와 candidate normalization 규칙을 모델 간 **동일하게** 유지한다.
    - test는 validation에서 모델이 확정되기 전까지 **사용하지 않는다**.
    - 더 새로운 모델이 나왔다고 해서 EXP-015 기준선을 **삭제하지 않는다** —
      EXP-015는 비교용 **incumbent 기록**으로 계속 보존한다.
    - 단순 전체 점수뿐 아니라 필요 시 **paired comparison**, **구간별 성능**, **runtime**,
      **파싱 실패율**도 함께 비교한다.

- 배경/문제:
  DECISIONS [012]에서 EXP-015를 FT 개선 iteration의 valid 기준선으로 채택했으나,
  이는 SillokBERT 계열 fine-tuning 내부 비교에 한정된 결정이었다.
  앞으로 ARI-8B, retrieval/BM25, reranking 등 **모델 계열 자체가 다른** 신규 접근을
  비교해야 하는데, 비교 기준(어떤 validation 샘플, 어떤 지표 정의, test 사용 시점)이
  명확히 고정되어 있지 않으면 모델마다 조건이 달라져 비교가 왜곡되거나
  발표에서 "무엇 대비 얼마나 개선했는가"를 설명할 수 없게 된다.

- 대안:
  - 신규 모델마다 별도 validation 샘플·지표로 개별 평가 (기각 — 모델 간 비교 불가)
  - 신규 모델 1차 스크리닝에 test set 사용 (기각 — 데이터 누수·test 예산 조기 소진 위험,
    RISK_CHECKLIST §2 위반)
  - EXP-015를 incumbent로 고정하고, 신규 모델은 동일 validation sample_id·동일 Exact@k
    정의로 재현·비교 (채택)

- 선택 이유:
  EXP-015는 이미 EXP-012 best checkpoint 전체 valid(n=10,679)에 대해 Exact@1/5/10을
  확보해 두었고, DECISIONS [012]에서 검증 방법과 함께 채택된 기록이 있다.
  동일한 기준을 모델 계열이 다른 신규 비교로 확장하면 별도의 재검증 없이 바로
  "EXP-015 대비 몇 pp 개선"이라는 발표 가능한 형태로 결과를 제시할 수 있다.

- 리스크:
  - Exact@5 단일 지표만으로 판단하면 특정 구간(빈도대·시대 등)에서는 개선되고 다른
    구간에서는 악화되는 모델을 과대평가할 수 있다 → 구간별 성능·paired comparison 병행 필요.
  - EXP-015는 **CPU 추론** 결과이며 EXP-012 epoch 2 체크포인트 소실로 Exact@k 기준
    재선택이 불가능했던 한계를 그대로 물려받는다(EXP-015/DECISIONS [012] 한계 참조).
    신규 모델의 추론 환경(GPU 등)이 다르면 runtime 비교 시 이 차이를 감안해야 한다.
  - retrieval/BM25, reranking 계열은 SillokBERT와 후보 생성 방식 자체가 달라
    candidate normalization 규칙을 동일하게 맞추는 작업이 추가로 필요할 수 있다
    (리스크로 인지하되, 구체 규칙은 별도 결정에서 확정한다).

- 검증 방법:
  신규 모델 평가 시 EXP-015와 **동일 valid sample_id**, **동일 Exact@k 정의**로 재현하고,
  `src/eval/paired_exact_compare.py` 등으로 EXP-015 대비 **paired 비교**(McNemar 등)를 수행한다.
  test는 validation 단계에서 모델이 최종 후보로 확정된 이후에만, 잔여 test 예산 범위 내에서
  사용한다(DECISIONS [012]).

- 결정자/참여: 팀(sooyah) + Claude(문서 정리)

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-015
  `docs/EXPERIMENT_LOG.md` EXP-012
  `docs/DECISIONS.md` [012]

### [013] title_key 충돌로 인한 동명 비문 판독문 붕괴 (2026-08-02)

- 상태: `active`

- 결정 내용:
  normalize_title_key(한글+한문 제목만 사용) 기반 매핑이 서로 다른 실물 비문을
  하나의 inscription_id로 붕괴시키는 결함을 확인하고 기록한다.
  현 시점에서 corpus 전면 재생성은 하지 않으며, 실측된 영향 범위에 따라 대응을 확정한다.
  향후 어떤 원천을 통합하더라도 제목 단독 매칭을 식별자 생성 규칙으로 사용하지 않는다.

- 배경/문제:
  XLS 메타 6,753행 대비 고유 title_key 6,623개로 130행이 중복이며,
  중복 키 97개 중 93개가 동명·이물(소재지·연대·크기 상이)이다.
  build_readings.py 85-98행에서 상세 판독문 fetch가 ksm_idx 단위 캐시로 1회만
  호출되고, 메타 조인도 {title_key: row} 구조라 마지막 행만 생존한다.
  확인 사례: 척화비(실물 6기 → readings 12행 전부 동일 텍스트),
  유명악영세불망비(실물 7기 → 위치 전부 대구 산격으로 통일),
  김명진영세불망비(실물 6기).

- 손실의 성격 구분:
  - 내용 오염: 영세불망비류. 지역마다 다른 인물을 기린 별개 텍스트가 붕괴 — 실질 손실.
  - 이형 소실: 척화비. 전국 표준 문안이라 내용 손실은 경미하나 이체자·결손 표기 이형 소실.

- 대안:
  - corpus 전면 재생성 후 모든 실험 재실행
  - 영향 그룹만 제외
  - 영향 그룹만 재수집해 복구
  - 현 상태 유지 + 한계 기록 (채택)

- 선택 이유:
  실측 결과 중복 연결 마스크 샘플 587/66,537(0.88%)로 규모가 작고,
  cross-split 0건(단, inscription_id 단위 붕괴로 인한 구조적 필연이며
  독립적 리스크 해소로 해석하지 않음)이 확인되어 즉시 차단할 누수는 없다.

- 검증 결과 (2026-08-02, 실측, scripts/audit_title_key_collapse_mask_impact.py):
  1. 중복 연결 마스크 샘플: 587 / 66,537 (0.88%)
  2. EXP-013 빈도 구간 변동 문자: 10건 (hapax 편입 3, hapax→unseen 이탈 2,
     rare→unseen 이탈 1, common/mid 구간 내 이동 4)
  3. cross-split: 0건 (구조적 필연, 독립적 리스크 해소로 해석하지 않음)
  4. 미확인: 연결 실패 5키(97-92)의 정체, XLS-readings 3행 gap(410-407),
     587건의 train/valid/test 분포 상세, test 내 near-duplicate 집중도
     (김명진 사례: test 12건이 고유텍스트 2종에서 발생)

- 리스크:
  - 학습 corpus의 텍스트 다양성이 실제보다 축소돼 있다.
  - 93그룹의 시대·소재지 메타가 마지막 행 값으로 오염돼 시대별 층화 분석에 영향 가능.
  - EXP-013 hapax 그룹(n=232) 구성원이 재집계로 소폭 바뀐다 — 발표용 수치는
    중복 제거 반영 후 재계산 권장.
  - test split 내 near-duplicate 집중(김명진 사례) — 유효 표본 크기 축소 가능성,
    규모 미확인.

- 수정 시 필수 조건 (누수 방지):
  붕괴를 해소해 동명 비문을 개별 inscription_id로 분리할 경우,
  split 그룹 키는 inscription_id가 아니라 title_key(또는 동등한 동명 그룹 키)를 사용한다.
  텍스트가 거의 동일한 동명 비문이 train과 test로 나뉘면 현재보다 심한
  near-duplicate 누수가 발생한다. 이 조건 없이 분리 작업을 수행하지 않는다.

- 재검토 조건:
  연결 실패 5키 정체 확인, 587건의 split별 분포 확인, test near-duplicate 집중도 확인
  중 하나 이상에서 규모가 커지면 상태를 재검토한다.
  (2026-09-17 추가 메모: EXP-019 저장소 감사에서 발견된 cross-split exact duplicate
  26건이 본 항목과 동일 원인인지 별개 현상인지는 아직 확인되지 않았다 — DECISIONS [015] 참조.)

- 결정자/참여: 팀(sooyah) + Cursor(감사·코드 확인·실측) + Claude(검증·영향 분석)

- 관련 링크:
  src/data/build_readings.py 85-98
  src/data/build_artificial_mask.py 105-114
  scripts/audit_title_key_collapse_mask_impact.py
  reports/title_key_collapse_mask_impact_2026-08-02.md / .json
  docs/RISK_CHECKLIST.md §1
  docs/DECISIONS.md [015] (2026-09-17 cross-split duplicate 26건 관련성 확인 필요)

### [012] FT 개선 루프 평가 프로토콜 — valid Exact@k · paired · test 예산 (2026-08-01)

- 상태: `active`

- 결정 내용:
  이후 fine-tuning 개선 실험의 일상 판단은 **validation Exact@k**로 한다.
  - **기준선:** EXP-012 best의 full valid Exact@k = **EXP-015**
    (Exact@1 33.78% / @5 53.24% / @10 60.90%, n=10,679; 학습 없음).
  - **iteration:** valid subsample **n=2,000**, `seed=42`, Exact@5 + **paired** 비교
    (`python -m src.eval.paired_exact_compare`, McNemar continuity correction).
  - **마일스톤(후보 확정):** full valid **n=10,679** Exact@1/5/10.
  - **다음 FT checkpoint 공식 채택:** 저장된 전 epoch에 대해 **valid Exact@5 최대**;
    동점이면 **valid_loss 최소**, 그다음 **낮은 epoch**.
    학습 중 alias `best/`의 `minimize_valid_loss`는 유지할 수 있으나, 채택·비교에는 post-hoc Exact@5를 쓴다.
  - **epoch 산출물:** `checkpoint-epoch-*` **전부 보존·백업** (best만 받으면 안 됨).
  - **full test:** EXP-012 이후 **잔여 2회**. 매번 목적·고정 후보·누적 횟수를
    `EXPERIMENT_LOG`에 사전/사후 기록. **사용 시점은 팀 별도 논의** (캘린더 잠금 없음).
  - EXP-012 **test** Exact@k(DECISIONS [010])는 발표·공식 비교용으로 **유지**한다.

- 배경/문제:
  EXP-012는 validation에 Exact@k 없이 `valid_loss`만 기록했다. 제품 지표는 Exact@5라
  proxy 불일치가 있고, RISK_CHECKLIST §5는 validation Exact@k를 권고한다.
  best만 백업해 epoch 2 가중치가 소실되어 Exact@k 기준 재선택이 불가능해졌다.

- 대안:
  - valid_loss만으로 iteration 계속
  - 매 실험마다 test를 열어 비교
  - valid Exact@k + paired + test 예산 제한 (채택)

- 선택 이유:
  test 반복 사용(§2) 압력을 줄이고, §5와 제품 지표를 정렬한다.
  paired 비교는 동일 샘플 불일치 쌍만 보므로 subsample에서도 +1~2pp급 차이를 감지하기 쉽다.

- 리스크 / 한계:
  EXP-012는 **loss 기준**으로 선택된 모델이다. 이후 Exact@5로 고른 모델과 비교할 때
  "+Xpp" 중 일부는 **선택 기준 변경 효과**일 수 있다. 불공정 비교는 아니나 한계로 기록한다.
  epoch 2 Exact@k는 복구 불가하다.

- 검증 방법:
  EXP-015 full valid 수치를 HANDOFF·EXPERIMENT_LOG에 고정.
  이후 후보는 paired 도구로 EXP-015(또는 동일 valid 예측)와 비교하고 test는 예산 내에서만 사용.

- 결정자/참여: 팀(sooyah) + Cursor / Claude 검토 수렴

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-015, EXP-012
  `docs/RISK_CHECKLIST.md` §2 · §5
  `src/eval/paired_exact_compare.py`
  `reports/ft_checkpoint_manifest_TEMPLATE.md`

### [011] 로컬 EXP-013 유지 · 클라우드 EXP-014·데모 shortlist 보류 (2026-08-01)

- 상태: `active`

- 결정 내용:
  로컬 **EXP-013**(Phase 4 층화 분석)은 유지한다.
  원격 브랜치 `origin/cursor/demo-cases-hapax-3c60`(`dcae614`)의 **EXP-014**(hapax 원인 가설)와
  real_defect 데모 shortlist는 **지금 merge하지 않고 보류**한다. 원격 커밋·파일은 삭제하지 않는다.

- 배경/문제:
  클라우드 작업이 발표 일정 촉박을 가정하고 진행된 측면이 있다. 발표까지 여유가 있고
  당분간 우선순위는 EXP-013이 보여 준 고빈도 편향·hapax 하락을 반영한 **모델(FT) 개선**이다.
  EXP-014는 EXP-012 full 예측 없이 EXP-011 subsample(n=48) 중심이라 가설 수준이다.
  또한 원격 브랜치가 데모 선별을 **EXP-013**으로 번호 매겨 로컬 층화 EXP-013과 충돌한다.

- 대안:
  - 클라우드 브랜치를 즉시 merge하고 번호를 재부여
  - 원격 브랜치·산출물을 삭제
  - 로컬 EXP-013까지 폐기
  - 로컬 EXP-013 유지 + 클라우드 EXP-014·데모만 보류(미병합)

- 선택 이유:
  층화 EXP-013은 다음 FT 설계(희귀 구간·빈도 재가중 등)의 근거이므로 유지한다.
  EXP-014·데모는 삭제 없이 원격에 두면 재개 가능하고, 지금 merge하면 번호 충돌·발표 작업이
  FT 경로보다 앞서게 된다.

- 리스크:
  보류가 길어지면 원격 브랜치와 `main`이 더 벌어질 수 있다.
  데모 shortlist를 잊으면 발표 직전 다시 선별 비용이 생긴다.

- 검증 방법 / 재개 조건:
  희귀 구간 FT 개선 실험 설계 시 EXP-012 full 예측으로 hapax 분석을 재실행한 뒤,
  번호를 재부여하고 merge 여부를 재검토한다. 데모는 병합 시 로컬 EXP-013과 다른 ID로 기록한다.

- 결정자/참여: 팀(sooyah) + Cursor

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-013, EXP-014(deferred stub)
  `docs/CURSOR_HANDOFF.md` §6
  `origin/cursor/demo-cases-hapax-3c60` (`dcae614`)

### [010] EXP-012를 공식 full-data fine-tune + full test 결과로 채택 (2026-07-12)

- 상태: `active`

- 결정 내용:
  EXP-012(Colab GPU, full train 46,139 + valid_loss best + full test 9,719)를
  SillokBERT fine-tuning의 **공식 full-data 비교 수치**로 채택한다.
  EXP-011은 CPU subsample 1차 baseline으로 유지하며 덮어쓰지 않는다.

- 배경/문제:
  EXP-011은 test n=2,000·부분 train이라 zero-shot full test(n=9,719)와 직접 비교하기 어렵다.
  GPU full-data 학습과 full test 1회 결과가 확보되어 공식 비교 기준을 고정할 필요가 있었다.

- 대안:
  - EXP-011 Exact@5(50.45%)를 계속 발표 주수치로 사용
  - EXP-012를 provisional만 두고 추가 epoch/하이퍼파라미터 탐색 후 채택
  - EXP-012를 즉시 공식 full-test 결과로 채택

- 선택 이유:
  `run_mode=full`, `full_data_no_row_cap=true`, best_epoch=1(valid_loss≈3.6619),
  test 9,719행 예측·summary가 일치하며 Colab smoke 메타와 분리 확인됨.
  동일 n=9,719에서 EXP-007 zero-shot 대비 Exact@1/5/10 상승이 관측됨.

- 리스크:
  전수 수동 검수 미완 corpus. 인공 마스크 ≠ 실제 결손 성능.
  산출물이 로컬 `notebooks/best/`에 있어 Git 밖 백업·소실 위험이 있다.
  Colab `_exp012_smoke` 동명 파일과 혼동하면 잘못된 메타를 인용할 수 있다.

- 검증 방법:
  `notebooks/best/`의 `best_checkpoint.json`, `finetune_metrics.json`, `evaluation_summary.csv`,
  예측 CSV 행 수(9,719)를 교차 확인. smoke 경로·`run_mode=smoke` 메타는 기각.

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-012
  `notebooks/best/`
  `reports/colab_exp012_runbook.md`

### [009] EXP-011을 1차 fine-tuning baseline으로 채택 (2026-07-08)

- 상태: `provisional`

- 결정 내용:
  EXP-011을 SillokBERT CPU subsample fine-tuning의 1차 baseline으로 채택한다.
  최종 또는 공식 full-data 성능으로는 사용하지 않는다.

- 배경/문제:
  zero-shot 대비 fine-tuning 효과를 동일한 test subsample n=2,000 조건에서 검증할 필요가 있었다.

- 대안:
  - zero-shot 결과만 유지
  - GPU full train 완료 후에만 baseline 확정
  - CPU subsample 결과를 1차 baseline으로 채택하고 이후 GPU 결과로 개선

- 선택 이유:
  모델 파일, 학습 metrics, 예측 CSV 2,000행, split 무중복 검증 파일이 존재하며,
  예측 CSV 재집계 결과가 문서 수치와 일치했다.

- 리스크:
  전체 train과 전체 test 결과가 아니며 CPU 1 epoch 실험이다.
  산출물 내부 legacy ID는 EXP-006으로 남아 있다.
  test subsample을 1차 효과 확인에 사용했으므로, 이후 설정 선택은 validation으로만 해야 한다.
  체크포인트와 상세 예측·검증 파일은 Git 밖의 로컬 경로에 있어 백업 전 소실 위험이 있다.

- 검증 방법:
  validation으로 설정과 체크포인트를 선택한 전체 GPU 학습 후 test 9,719건으로 최종 평가한다.
  산출물을 PC 외부에 백업하고 위치·크기·SHA-256을 기록한다.

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-011

### [008] SillokBert를 Phase 4 primary fine-tuning 대상으로 확정 (2026-07-06)

- 상태: `active`

- 결정 내용:
  Phase 4 supervised fine-tuning의 **1차( primary ) 대상 모델**을 `ddokbaro/SillokBert`로 확정한다.
  `bert-base-chinese` fine-tuning은 **비교 실험(secondary)** 으로 유보하거나 소규모로만 수행한다.

- 배경/문제:
  Phase 3 zero-shot(EXP-006/007)에서 SillokBert가 전체 Exact@5 43.17%로 bert-base-chinese 34.98%를 상회했다.
  다만 ngram baseline(EXP-004)이 고빈도 정답 글자에 성능이 집중(very_common 47.1%)되어,
  SillokBert 우위가 **동일한 편향**인지 **희귀자·도메인 이해**인지 구분되지 않았다.

- 대안:
  - (A) bert-base-chinese를 primary로 fine-tune (일반 중국어 BERT)
  - (B) SillokBert를 primary로 fine-tune (한국 고전/한문 사전학습)
  - (C) 둘 다 동등하게 full fine-tune (리소스 2배)

- 선택 이유:
  EXP-008 층화 분석(`reports/phase3_stratified_analysis_2026-07-06.md`)에서 SillokBert가
  **train에서 한 번 이상 등장한 빈도 구간**에서 bert-base-chinese를 Exact@5 기준 상회:
    - very_common: 58.0% vs 54.6%
    - common: 44.9% vs 32.7%
    - mid: 27.6% vs 16.5%
    - rare: 20.8% vs 11.3%
    - hapax: 8.2% vs 7.8%
  희귀·hapax 구간에서도 우위가 있어, 단순 고빈도 암기에만 국한되지 않은 **도메인 특화 zero-shot 이점**으로 해석한다.
  단, unseen 구간은 SillokBert 3.7%로 bert-base-chinese 4.1%보다 0.4%p 낮아 미출현 글자까지 우위라고 주장하지 않는다.
  tokenizer coverage(EXP-010)에서 SillokBert UNK **occurrence** 2.51% < bert 3.42% (학습 corpus 기준).

- 리스크:
  - EXP-006/007은 CPU에서 실행됐다. GPU 호환성 확인은 필요할 수 있으나 과학적 우선순위는 full fine-tuning보다 낮다.
  - cross-split n-gram overlap(EXP-009): test 59.5%가 train과 2-gram 이상 문맥 키 공유 →
    ngram·일부 쉬운 BERT 케이스는 정형구 memorization 영향 가능. inscription split만으로는 불충분.
  - fine-tuning 후 overfitting·valid/test gap 모니터링 필요.

- 검증 방법:
  - Phase 4: SillokBert를 artificial_mask train으로 학습하고 validation으로 early stopping·설정을 선택한 뒤 test Exact@k를 기록한다.
  - bert-base-chinese는 동일 하이퍼파라미터 소규모 비교 또는 생략(리소스에 따라).
  - GPU 호환성은 full fine-tuning smoke 단계에서 확인한다. zero-shot CUDA 재실행은 선택 작업이다.

- 결정자/참여:
  팀 + Cursor (EXP-008~010 분석 기반)

- 관련 링크:
  `docs/EXPERIMENT_LOG.md` EXP-006~010,
  `reports/phase3_stratified_analysis_2026-07-06.md`,
  `reports/ngram_cross_split_overlap_2026-07-06.md`,
  `reports/tokenizer_vocab_coverage_2026-07-06.md`,
  `src/data/finetune_sillokbert.py`

### [007] 결손(□)과 한글 치환(◇) 기호 분리 — gsm_preprocess_v4 (2026-07-06)

- 상태: `active`

- 결정 내용:
  `gsm_preprocess_v4`에서 실제 결손 마커는 `□`, 한글 치환은 `◇`로 분리한다.
  `has_real_defect`와 `defect_count`는 원문 결손 마커에서 온 `□`만 집계한다.
  가공 산출물(`readings.csv`, `artificial_mask_dataset.csv`, `real_defect_inference.csv`, `defect_markers_inventory.json`)을 v4 기준으로 재생성한다.

- 배경/문제:
  v3에서 결손과 한글 치환이 모두 `□`로 통일되어, `has_real_defect`가 한글-only 행(2,258건)까지 결손으로 분류되고 `real_defect_inference` 데모에 한글 위치가 혼입(807건)되었다.

- 대안:
  - (A) v3 유지 + `real_defect_inference` 필터만 강화
  - (B) 기호 분리(□/◇) + 파생 데이터 재생성
  - (C) 위치별 메타데이터 JSON 컬럼 추가(기호는 □ 유지)

- 선택 이유:
  (B)는 `text_normalized`에서 위치별로 결손과 한글을 구분할 수 있고, 혼합 행에서도 `□` 위치만 실제 결손 데모에 사용 가능하다. (A)는 행 단위 필터만 가능하고 혼합 행 문제가 남는다. (C)는 구현·소비 복잡도가 높다.

- 리스크:
  기존 v3 기준 결손 통계(47만 개)와 비교 불가. `◇`가 원문에 드물게 존재할 경우 오인 가능(현재 corpus에서 미확인).

- 검증 방법:
  `is_training_text=True` 7,158행 유지, 한글-only 2,258행 `has_real_defect=False`, `real_defect_inference`에 한글-only 0건, `python -m unittest discover -s tests` 통과.

- 결정자/참여:
  팀 + Cursor 구현

- 관련 링크:
  `src/preprocess/gsm_v4.py`, `src/data/reprocess_readings.py`, `docs/EXPERIMENT_LOG.md` EXP-002

### [006] 판독문 corpus의 한글 치환 및 학습 대상 무결성 검증 (2026-07-03)

- 상태: `active`

- 결정 내용:
  `readings.csv`에서 한글이 포함된 텍스트는 `quality_flag=review` 또는 `is_training_text=False`로 격리하고, 현재 baseline 학습에는 `is_training_text=True` 행만 사용한다.

- 배경/문제:
  `defect_markers_inventory.json` 및 전처리 결과에서 한글 치환량이 약 322,804자로 확인되어, 한글/OCR/한글 비문이 학습 데이터에 섞였을 가능성을 점검했다.

- 검증 결과:
  `readings.csv` 14,702행 중 학습 대상(`is_training_text=True`)은 7,158행이며, 이 영역의 한글 총량은 0자였다.
  한글 322,804자는 전부 `is_training_text=False` 영역에만 존재했다.
  CSV UTF-8 인코딩과 열 구조도 정상이며, 열 수 어긋남은 0건이었다.

- 해석:
  한글 치환은 학습 신호를 오염시키지 않았고, 한글 비문·판독불가 표기·한글 주석 혼입 행은 의도대로 학습에서 제외되었다.
  따라서 한글 치환 문제는 baseline 학습을 막는 위험이 아니다.

- 리스크:
  한자 20자 이상이면서 한글 1~5자 때문에 제외된 행이 1,628행 확인되었다.
  이는 데이터 무결성 문제는 아니지만, 데이터 효율 문제다.
  baseline 이후 `전면`, `후면`, `출전` 등 구조 표기 제거를 통해 일부 회수할 수 있다.

- 후속 조치:
  현재 baseline은 `is_training_text=True` 7,158행만 사용한다.
  실제 결손이 있는 3,907행은 정량 평가가 아니라 실제 복원 데모 후보로 분리한다.
  1,628행 회수는 baseline 이후 전처리 개선 실험으로 검토한다.

- 결정자/참여:
  데이터 분석 담당 + Claude/Cursor/GPT 검토

- 관련 링크:
  `data/processed_layer/readings.csv`,
  `data/processed_layer/defect_markers_inventory.json`,
  `docs/DATA_PREPROCESSING_RULES.md`,
  `docs/EXPERIMENT_LOG.md`

### [005] 기존 자동 생성 산출물 초기화 (2026-07-02)

- 상태: `completed`

- 결정 내용: 검수되지 않은 가공 산출물과 리포트 산출물을 삭제하고 프로젝트를 정리한다.
- 배경/문제: 데이터 원천과 확정 스키마가 아직 불안정하여 기존 산출물을 성능 근거로 쓰기 어렵다.
- 선택 이유: 원본/문서/코드는 보존하고, 검수되지 않은 산출물만 제거해 재현성을 높인다.
- 리스크: 일부 임시 결과가 사라질 수 있다.
- 검증 방법: git status와 git log로 삭제 대상 및 커밋 결과 확인.

### [004] 전처리 v3: 글자 추정 보정 금지, 한글→□ 치환, 결손 텍스트는 학습 제외 (2026-07-01)

- 상태: `superseded` — 한글 치환 기호는 DECISIONS [007]의 v4(`◇`)가 대체했다. 원문에 없는 글자 추정 금지와 실제 결손 텍스트 학습 제외 원칙은 유지한다.

- 결정 내용: `preprocess_gsm.py`를 `gsm_preprocess_v3`으로 개정한다. (a) 원문에 없는 글자를 삽입·삭제·치환하는 수동 보정을 전면 제거한다. (b) 잔존 한글(OCR 노이즈)은 삭제하지 않고 `□`로 치환한 뒤 record를 `quality_flag="review"`로 격리한다. (c) `is_training_text`를 `quality=="ok" and not has_real_defect`로 정의해 실제 결손(`□`)이 있는 텍스트는 학습/마스킹 원천에서 제외한다.
- 배경/문제: 데이터 감사에서 v2의 세 가지 문제가 확인됨. (1) `內진晨 → 內間晨`처럼 원문에 없는 한자 `間`을 삽입(데이터 발명) → 인공 마스킹 시 틀린 정답 라벨이 될 위험. (2) 한글을 무조건 삭제해 글자 수·위치 인덱스가 밀림(`mask_position` 어긋남). (3) 결손 17개인 ksm_3가 `is_training_text=True`로 표시되어 실제 결손 텍스트가 학습 대상으로 오인될 위험.
- 대안:
  - (A) 안전한 음운 치환(`씨→氏`)만 코드에 남긴다.
  - (B) 모든 수동 보정 제거 + 한글→`□` + `review` 격리 + 결손 텍스트 학습 제외.
  - (C) 한글을 그대로 두고 `review`만 건다.
- 선택 이유: (A)도 코드가 글자를 추정하는 판단을 포함해 재현성·발표 설명에서 약점이다. (C)는 정규화 텍스트에 한글이 남아 글자 단위 처리에 부적합하다. (B)는 코드가 어떤 글자도 추정하지 않고 위치를 보존하며, 실제 한자는 사람이 `review` 단계에서 확정하므로 가장 방어적이고 설명 가능하다.
- 리스크: `班씨詩 → 班氏詩`처럼 명백히 옳던 치환도 `□`+`review`로 바뀌어 사람 검토 부담이 늘어난다. 결손 텍스트를 학습에서 전면 제외하므로, 이후 "`□` 위치만 피해 인공 마스킹"하는 파이프라인이 생기면 `is_training_text` 정의를 재검토해야 한다.
- 검증 방법: v3 로직을 실제 raw 2건으로 실행해 불변식(`text_length==len`, `defect_count==□ count`, 잔존 결손기호 0, 한글 잔존 0)을 통과 확인. v2/v3 diff로 변경이 의도한 3곳(`씨지진→□`)에 한정됨을 확인. raw 파일 SHA256은 처리 전후 동일(원본 무수정).
- 결정자/참여: 데이터 분석 담당(사람) + Claude(Cowork) 감사·구현.
- 관련 링크: `preprocess_gsm.py`(`PREPROCESS_RULE_VERSION="gsm_preprocess_v3"`), `data/processed_layer/readings.csv`(v3 승격), `data/processed_layer/defect_markers_inventory.json`(`hangul_replaced_total` 추가), `docs/DATA_PREPROCESSING_RULES.md` §2·§3. v2 결과는 diff 검토 후 제거(git 미추적).


### [003] NRICH OpenAPI `md_data` 필드는 실제 XML 역매핑 기준으로 파싱한다 (2026-06-29)

- 상태: `provisional` — 2건 외 추가 샘플 검증 전까지 잠정 적용한다.

* 결정 내용: NRICH OpenAPI 목록 XML의 `md_data1`~`md_data22`는 공공데이터포털 공식 명세서 순서가 아니라, 실제 XML 샘플을 역추적해 확인한 매핑 기준으로 파싱한다.
* 배경/문제: 공식 명세서의 `md_data1`~`md_data22` 순서와 실제 XML 응답의 의미가 일치하지 않았다. 예를 들어 명세서상 `md_data1`은 `DATA_LINK`로 되어 있으나, 실제 XML에서는 목록 순번 `ROW_NUM`이었다. 반대로 실제 상세 URL인 `DATA_LINK`는 `md_data22`에 들어 있었다.
* 대안:

  * (A) 공공데이터포털 공식 명세서의 `md_dataN` 정의를 그대로 사용한다.
  * (B) 실제 XML 응답 샘플을 기준으로 `md_dataN`의 의미를 역매핑한다.
  * (C) `md_dataN` 필드를 사용하지 않고 상세 페이지에서만 데이터를 수집한다.
* 선택 이유: (A)를 따르면 `KSM_IDX`, 제목, 시대, 개관, 상세 URL 등 핵심 메타데이터가 잘못 파싱된다. (C)는 목록 메타데이터를 버리게 되어 수집·검수 효율이 떨어진다. 따라서 이계신도비(`KSM_IDX=3`)와 이소한신도비(`KSM_IDX=5`) 2건으로 교차 검증한 (B)를 채택한다.
* 리스크: 현재 검증 샘플은 2건뿐이므로, 시대·유형·지정종목·이용조건이 다른 레코드에서도 동일 매핑이 유지되는지 추가 확인이 필요하다. 또한 이 매핑은 NRICH 목록 XML 기준이며, JSON 응답이나 상세 API에서는 구조가 다를 수 있다.
* 검증 방법: 수집 파이프라인 구현 전 3~8건의 추가 XML 샘플을 확인한다. 특히 `KSM_IDX`와 `DATA_LINK`의 `ksm_idx` 일치 여부, 코드·명칭 쌍(`KSM_EAR`/`KSM_EAR_NM`, `KSM_TYPE`/`KSM_TYPENM`, `FORMAL_CD`/`FORMAL_CDNM`)을 점검한다.
* 결정자/참여: 데이터 분석 담당(사람) + GPT/Cursor 검토.
* 관련 링크: `docs/DATA_SCHEMA.md` §18.3, §18.5, §18.6.


### [002] 랜덤 문자 단위 split은 데이터 누수 위험이 있으므로 보류한다 (2026-06-26)

- 상태: `active`

- 결정 내용: 글자를 무작위로 섞어 train/valid/test로 나누는 랜덤 문자 단위 split은 당분간 쓰지 않는다.
- 배경/문제: 같은 비문의 인접 글자가 train과 test에 동시에 들어가면, 모델이 문맥을 "외워서" 맞히게 되어 점수가 부풀려진다.
- 대안:
  - (A) 랜덤 문자 단위 split — 구현은 가장 쉬움.
  - (B) inscription_id(유물/문헌) 단위 split — 누수 방지에 강함.
  - (C) 시대/도메인 단위 분리 split — 일반화 성능 측정에 유리.
- 선택 이유: 누수가 발생하면 모든 성능 수치를 신뢰할 수 없게 된다. 대회/발표에서 가장 치명적인 약점이므로 (A)는 보류하고 (B)를 우선 검토한다.
- 리스크: inscription 단위로 나누면 split별 데이터 양이 불균형해질 수 있다. → 분포를 점검하고 필요 시 (C)를 병행한다.
- 검증 방법: 같은 데이터로 (A)와 (B)를 각각 돌려 성능 차이를 비교한다. (A)가 비현실적으로 높으면 누수가 입증된다.
- 결정자/참여: 데이터 분석 담당(사람) + GPT/Claude 검토.
- 관련 링크: `docs/DATA_PREPROCESSING_RULES.md` 4절, `docs/EXPERIMENT_LOG.md`(누수 비교 실험 예정).

### [001] 복원 단위는 우선 글자(문자) 단위로 한다 (2026-06-26)

- 상태: `active`

- 결정 내용: 결손 복원의 기본 단위를 단어가 아닌 **글자(character) 단위**로 한다.
- 배경/문제: 금석문 결손은 글자 단위로 발생하고, 한문/고전중국어는 단어 분절 기준이 불안정하다.
- 대안:
  - (A) 단어(word) 단위 복원 — 분절기 필요, 한문에서 불안정.
  - (B) 글자(character) 단위 복원 — 분절 불필요, 평가 명확.
- 선택 이유: (B)는 분절 오류가 없고, `Exact@k`로 평가가 깔끔하며, 참조 모델 BIRD가 character-level이라 baseline 이식이 쉽다.
- 리스크: 글자 단위만으로는 더 긴 문맥/구절 복원에 약할 수 있다. → 이후 단계에서 문맥 강화·재랭킹으로 보완 검토.
- 검증 방법: EXP-003~011의 문자 단위 인공 마스크 평가에서 `Exact@1/5/10`을 일관되게 산출했다. `Group@k`는 이체자 매핑 확정 후 별도로 검증한다.
- 결정자/참여: 팀 전체 + GPT/Claude 검토.
- 관련 링크: `PROJECT_OVERVIEW.md`, `EXPERIMENT_LOG.md` EXP-003~011.



---

*관련 문서: `AI_COUNCIL.md`(결정 절차), `EXPERIMENT_LOG.md`(검증 결과), `PROJECT_OVERVIEW.md`(현재 상태).*
