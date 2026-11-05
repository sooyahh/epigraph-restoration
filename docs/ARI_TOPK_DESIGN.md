# ARI Top-K Design (Appendix C → Exact@5)

상태: **design only** (2026-09-16)  
범위: Appendix C 재구성 + 우리 single-mask Exact@5 대응. 구현·모델 로드·GPU 실행 없음.

출처 태그:

| 태그 | 의미 |
|---|---|
| **PAPER** | 논문 본문·Appendix C·Fig. 12에 명시 |
| **PUBLIC** | `EvelynKimm/ARI` 또는 HF `DAMI-Lab/ARI-8B` 카드/예시에 존재 |
| **OURS** | 공개 코드에 없어 우리 평가에 맞게 고정하는 구현 결정 |
| **UNKNOWN** | 논문/코드 모두 침묵. 구현 전에 한 번 더 고정해야 함 |

관련: `docs/ARI_INTEGRATION_PLAN.md` §7. 이 문서가 Top-K의 정본이다.

---

## 1. 한 줄 결론

ARI 후보의 점수와 순위는 BERT의 `softmax(mask logits)`가 아니다.

```text
PAPER: 한 후보 한자 = tokenizer가 그 한자를 이루는 토큰 시퀀스
       score(한자) = 그 토큰들의 log p(token | prefix)의 산술평균
       순서      = score 내림차순
       확장      = 첫 토큰에서 갈래를 나눈 뒤, 유효 문자열이 될 때까지 계속 생성
```

우리 Exact@5는 이 순서로 만든 **서로 다른 한자 5개**를 `rank_of_gold()`에 넣는다.  
`torch.topk(next_token_logits, 5)`는 이 절차가 아니다.

공개 repo에는 이 디코더가 **없다**. 본평가 Exact@5는 논문 알고리즘의 **자체 재구현**이 필요하다.

---

## 2. 논문이 실제로 말한 것

### 2.1 Appendix C (PAPER, 거의 전부)

ACL 2026 Findings Appendix C, Fig. 12:

1. Qwen 계열은 byte-level BPE (Wang et al., 2020).
2. 빈도 높은 중국어 한자는 **토큰 1개**.
3. vocab에 없는 한자는 보통 **바이트 토큰 2–3개**로 쪼개진다.
4. i번째 손상 글자를 복원할 때 **constrained decoding**:
   - 첫 바이트 토큰을 낸 뒤
   - **유효한 문자 시퀀스가 형성될 때까지** 생성을 계속한다.
5. 후보 점수 = 그 문자를 구성하는 바이트 토큰들의 **log probability 평균**.
6. 그 점수로 Top-K를 사용자에게 보여 준다.
7. `[D(i+1)]`은 기본으로 **i번째의 Top-1**에 조건 부여.
8. 사용자가 c등 후보를 고르면 그 글자로 context를 갱신.
9. Top-K 추출에 **vLLM** (Kwon et al., 2023)을 쓴다.

### 2.2 Fig. 12가 보여주는 트리 (PAPER, OCR 한계 있음)

손상문 예: `吏曹假[D1][D2]來言…`  
gold `[D1] = 郞`, 그림에 `n=2` (2토큰 한자).

한 레벨이 아니라 **두 단계**다.

```text
prefix (i번째 글자 직전)
        │
        ├─ 1st tokens (완전 한자)     郎  logprob -6.877
        │                             朗  logprob -7.752
        │                             堂  logprob -9.065
        │
        └─ 1st tokens (미완 바이트)    <byte>  logprob -0.002  등
                 │
                 └─ 2nd tokens로 문자 완성
                       郞   (구성 토큰 평균)
                       郔
                       郘  …
```

읽을 수 있는 함의:

- 1스텝 후보 중 일부는 이미 한 글자 한자 → 그 자체가 후보.
- 일부는 한 글자가 아님 → 다음 토큰을 붙여 한자를 완성.
- gold `郞`(n=2)과 한 토큰짜리 `郎`은 **다른 글자**다. next-token Top-5만 보면 gold가 빠질 수 있다.

그림의 개별 숫자·토큰 문자열은 PDF OCR이 깨져 있어 **알고리즘 증거로만** 쓴다. 하이퍼파라미터(1st beam 폭 등)는 그림에서 확정하지 않는다.

### 2.3 논문 평가에서 Top-K가 쓰인 곳 (PAPER)

- Fig. 11: 학습 중 DRand / DNE의 Top-1 / Top-5 / Top-10.
- §5.1: test에서 top-K accuracy.
- §5.2: 전문가 UI에 ARI·BERT-Res 후보를 보여 줌. Gemini는 logits가 없어 Appendix C를 못 씀.

자동 Exact@k와 전문가 후보 리스트가 **같은 Appendix C 절차**를 쓴다고 보는 것이 타당하다.

### 2.4 우리 평가와 다른 점 (PAPER vs 우리 데이터)

논문은 문서당 평균 여러 `[Dn]`. `[D2]`는 `[D1]` Top-1에 조건.  
우리는 sample당 `[D1]` 하나 (`single_hanja_mask_v1`).  
**OURS:** `[D2]` 연쇄는 구현하지 않는다. 한 sample = 한 글자 Top-K.

---

## 3. 공개 코드에 있는 것 / 없는 것

`EvelynKimm/ARI` main tree (README, `scripts/`, `src/crawl`, `src/data_preprocessing`, `src/hanja_llm`) 전수 기준.

### 3.1 있는 것 (PUBLIC)

| 파일 | Top-K와의 관계 |
|---|---|
| `src/hanja_llm/prompt.py` | JSON 출력 형식을 학습/추론에 요구. 후보는 JSON 값이다. |
| `src/hanja_llm/hanja_restoration_dataset.py` | 응답 타깃 = `mask_label` 문자열 |
| `src/data_preprocessing/build_dataset.py` | `mask_label = json.dumps(map_info, ensure_ascii=False)` |
| `src/data_preprocessing/mask.py` | `map_info`: `{"[D1]": "原", ...}` 글자 하나 ↔ 마커 하나 |
| HF 카드 예시 | `model.generate(do_sample=False, max_new_tokens=128)` 후 `decode`. **Top-1 문자열 하나** |

`json.dumps` 기본 구분자는 `", "` 와 `": "` 이다. SFT 타깃 예:

```text
{"[D1]": "某"}
```

콜론 뒤 공백이 있다. HF SYSTEM_PROMPT 예시는 공백 유무가 섞여 있다 (**PUBLIC 내부 불일치**). Top-K prefix는 SFT 타깃을 따른다 (**OURS**, §6.3).

### 3.2 없는 것 → 전부 재구현 (PAPER 근거 + OURS)

다음 파일/함수는 공개 tree에 없다.

- vLLM 호출
- constrained decoding
- 토큰 단위 logprob 평균
- Top-K 한자 리스트
- JSON 파서
- Exact@k 평가
- “valid character sequence” 판정 함수

HF 예시는 Appendix C가 아니다. smoke test 전용이다.

---

## 4. 점수(score)가 무엇인가

### 4.1 정의 (PAPER)

한자 `c`가 토큰 시퀀스 `t_1, …, t_n` (n은 보통 1–3)으로 생성되었을 때, 각 스텝의 조건부 로그확률

```text
ℓ_j = log p(t_j | prompt + prefix + t_1…t_{j-1})
```

에 대해

```text
score(c) = (ℓ_1 + … + ℓ_n) / n
```

이것이 논문이 말한 “averaging the log probabilities of their constituent byte tokens”이다.

- n=1 (vocab 한자): `score = ℓ_1`. BERT의 단일 토큰 logprob과 형태는 비슷하나, 위치·모델이 다르다.
- n=2 또는 3: **합이 아니라 평균**. 긴 분해를 길이로 나눈다.

### 4.2 BERT baseline 점수와의 차이 (PUBLIC 우리 코드)

`FillMaskPredictor._predict_batch_model_scored()`:

```text
score_BERT = softmax(mask_position_logits)[token_id]   # 확률, 합 아님, 평균 아님
```

`compute_exact_metrics()`는 점수를 쓰지 않고 **리스트 순서만** 본다.  
그래도 순서를 만드는 점수가 다르면 Top-5 집합이 달라진다.

| | EXP-015 BERT | ARI Appendix C |
|---|---|---|
| 위치 | 유일한 `[MASK]` 토큰 | JSON 값에서 i번째 한자가 시작되는 지점 |
| 원자 | 토큰 1개 (길이 1만 채택) | 한자를 이루는 1–3 토큰 |
| 점수 | softmax **확률** | 토큰 logprob의 **산술평균** |
| 순서 | 확률 내림차순 | 평균 logprob 내림차순 |
| CSV | `top5_scores`에 확률 | **OURS:** 평균 logprob를 그대로 기록 (확률로 변환하지 않음) |

CSV 스키마의 `top*_score`는 “상대 선호 점수”이지 보정 확률이 아니다 (`DATA_SCHEMA.md` §9.3). ARI 점수를 `exp(mean logprob)`로 바꾸지 않는다. 그건 논문에 없다.

### 4.3 점수가 아닌 것 (명시적 배제)

구현 시 아래를 점수라고 부르지 않는다.

| 잘못된 점수 | 이유 |
|---|---|
| `topk(next_token_logits, 5)` 의 softmax | 미완 바이트와 완결 한자를 같은 1스텝에서 섞음 |
| 토큰 logprob의 **합** | 논문은 평균. 합은 긴 분해에 불리 |
| JSON 전체 시퀀스의 logprob | 마커 키·따옴표 확률까지 섞임 |
| `generate()` 한 샘플의 문장 점수 | 후보가 1개뿐 |
| BERT와 맞추려고 `exp(score)` | 논문 외 변환 |

### 4.4 평균이 만드는 편향 (PAPER 리스크, 고치지 않음)

UTF-8 선행 바이트는 `ℓ_1 ≈ 0`에 가까울 수 있다 (Fig. 12의 `-0.002`).  
2토큰 한자의 평균은 1토큰 한자보다 **높아질 수** 있다.  
논문이 평균을 선택했으므로 1차 구현은 평균을 유지한다. 합(joint logprob)은 ablation 후보로만 적고, 공식 Exact@5 기본값이 아니다.

---

## 5. 서로 다른 5개 한자를 어떻게 얻는가

목표는 `["字a","字b","字c","字d","字e"]`처럼 **문자 집합 크기 5**, 순서 있음.

### 5.1 왜 next-token 5개가 아닌인가 (PAPER + OURS)

Qwen이 `郞`을 `(t1, t2)`로 쪼개면, 1스텝 Top-5에는 `t1`(미완 바이트)과 완결 토큰 `郎`이 섞인다.  
`t1`을 한자로 디코드하면 글자가 아니거나 다른 글자다.  
길이-1 필터를 치면 gold `郞`은 사라지고 Exact@5가 구조적으로 불리해진다.

그래서 후보는 **토큰이 아니라 완결 한자**다.

### 5.2 알고리즘 (PAPER 골격 + OURS 살)

Fig. 12는 1st-token에서 갈래를 나누고, 미완이면 2nd-token을 붙인다.  
1스텝 greedy 다음 토큰만 이어 가는지, 2스텝도 Top-M인지는 **UNKNOWN**.  
그림에 2nd tokens가 여러 개이므로 **OURS 기본은 두 레벨 모두 분기**이다.

```text
입력:  prompt_ids,  forced_prefix_ids = tokenize('{"[D1]": "')
      k = 5
      B1 = 32          # 1st-token beam  (UNKNOWN → OURS 기본값)
      B2 = 8           # 이후 스텝 분기  (UNKNOWN → OURS 기본값)
      Lmax = 4         # 한 글자당 최대 토큰 (논문은 전형 2–3, +1 여유)

상태: prefix = prompt_ids + forced_prefix_ids
frontier = [ (prefix, tokens=[], logprobs=[]) ]
completed = {}     # 스텝을 넘어 누적. 매 step 리셋 금지.

for step in 1..Lmax:
    새 frontier = []
    for 각 가설 h in frontier:
        logits = model(h.prefix).last_position
        ℓ = log_softmax(logits)
        take = B1 if step==1 else B2
        for t among top-`take` tokens:
            ids = h.tokens + [t]
            lp  = h.logprobs + [ℓ[t]]
            s   = decode(ids)          # special token 제외
            status = classify(s)       # §5.3
            if status == COMPLETE_HANJA:
                completed[s] = score 최대 경로만 남김   # UNKNOWN → OURS
            elif status == INCOMPLETE:
                새 frontier.append(가설(prefix+[t], ids, lp))
            else:  # INVALID
                drop

    unique_hanja 가 k개 이상이고 더 이상 INCOMPLETE 가 없으면 break

한자 목록을 score 내림차순 정렬
상위 k개 반환 (부족하면 있는 만큼)
```

`completed`는 루프 밖에서 유지한다. 매 step `completed = []`로 비우면 Fig. 12의 1토큰 한자(郎 등)가 미완 바이트 분기가 이어지는 순간 사라지고, Exact@5가 논문 순위와 달라진다.

`B1=32`인 이유: BERT도 `k*3`을 뽑아 길이-1만 남긴다. ARI는 미완 바이트·비한자·JSON 잔여를 버려야 해서 더 넓게 뽑는다. 값이 논문에 없으므로 run_meta에 기록한다.

### 5.3 “valid character sequence” (PAPER 용어, 정의는 OURS)

논문은 정의를 안 준다. 우리 판정은 기존 `is_cjk_ideograph()` (`src/nrich/text_utils.py`)와 같은 문자 집합으로 맞춘다.

디코드 문자열 `s` (토큰 시퀀스만, prefix JSON 제외):

| `s` | status | 이유 |
|---|---|---|
| 빈 문자열, 잘못된 UTF-8, 치환문자 `�`만 | INCOMPLETE | 바이트 미완 |
| 길이 1 이고 `is_cjk_ideograph(s)` | COMPLETE_HANJA | 후보 |
| 길이 1 이지만 한자 아님 (`"`, `{`, 한글, 숫자, 공백) | INVALID | 분기 종료 |
| 길이 ≥ 2 | INVALID | 한 마커 = 한 글자 (PAPER+PUBLIC 프롬프트) |

**OURS:** NFC 정규화 후 길이/한자를 본다. 이체자 통합은 하지 않는다 (Group@k 미사용과 동일).

### 5.4 중복 한자 (UNKNOWN → OURS)

같은 글자가 (a) 단일 vocab 토큰과 (b) 바이트 2–3토큰 두 길로 나오면, **평균 logprob이 큰 경로만** 남긴다. 리스트에 같은 글자를 두 번 넣지 않는다.  
`rank_of_gold`는 첫 일치만 보므로 중복은 Exact@5를 왜곡한다.

### 5.5 5개가 안 채워질 때 (OURS, BERT와 동일 철학)

유효 한자가 5개 미만이면 짧은 리스트를 그대로 둔다. Exact@5는 그 안에서만 hit를 본다 (`rank_of_gold`가 `None`이면 miss). 패딩 한자를 넣지 않는다.

---

## 6. 순서를 무엇으로 정하는가

### 6.1 공식 순서 (PAPER)

```text
order = sort_desc_by score(c) = mean_j log p(t_j | …)
```

Exact@1 = 이 리스트의 `[0]`.  
Exact@5 = gold가 인덱스 0..4에 있는지 (`rank <= 5`).

2–5등 상대 순서는 Exact@5 값에는 없고 Exact@1·`rank_of_gold`에는 있다. EXP-015와 같다.

### 6.2 타이브레이크 (UNKNOWN → OURS)

평균 logprob이 같으면:

1. 토큰 수 `n`이 더 작은 쪽 (단일 토큰 한자 우선)
2. 그래도 같으면 유니코드 코드포인트 오름차순 (재현성)

논문에 타이브레이크는 없다. 이 규칙은 run_meta에 적는다.

### 6.3 JSON prefix와 순서 (PUBLIC 타깃 + OURS 디코딩)

Fig. 12는 `{`가 아니라 한자 토큰부터 그린다.  
SFT 타깃은 `json.dumps` JSON이다. 제약 없이 1스텝 Top-K를 하면 `{`가 1등일 수 있다.

**OURS (본평가 Top-K):** teacher-force prefix를 SFT와 맞춘다.

```text
forced_prefix = '{"[D1]": "'
```

그 다음 토큰부터 §5.2를 적용한다. 완성 따옴표 `"}` 는 점수에 넣지 않는다.

**OURS (smoke):** prefix 없이 `generate()` → JSON 파싱 → 글자 1개. 순서가 아니라 파이프라인 검증.

sanity (구현 후, 아직 실행하지 않음): 같은 샘플에서 constrained Top-1과 `generate()` 파싱 Top-1의 불일치율을 기록한다. 불일치가 커도 EXP-015 비교는 **constrained 리스트만** 쓴다.

SYSTEM_PROMPT 예시의 `{"[D1]":"李"}` (공백 없음)는 prefix로 쓰지 않는다. 학습 타깃은 `json.dumps`다.

---

## 7. 우리 Exact@5 파이프라인에 끼우는 방법

기존 evaluator는 바꾸지 않는다.

```text
valid row
  original_text, mask_position, label_char, sample_id
        │
        ▼
  build_masked_window(..., mask_token="[D1]")     # 기존 함수, ±64
        │
        ▼
  ARI chat prompt (019A: related docs 없음)
        │
        ▼
  teacher-force  '{"[D1]": "'
        │
        ▼
  Appendix C 재구현  (§5.2)     ← 공개 코드 없음
        │
        ▼
  top_chars: list[str]   길이 ≤ 5, 원소는 한 글자 한자
  scores:    list[float]  평균 logprob, 같은 순서
        │
        ▼
  rank = rank_of_gold(top_chars, label_char)       # src/eval/metrics.py
  Exact@1 / @5 / @10 는 기존 compute_exact_metrics
  CSV는 PREDICTION_COLUMNS 그대로
```

`label_char`는 데이터셋 컬럼이다. 모델 JSON에서 읽지 않는다.

### 7.1 BERT 리스트와의 계약

`FillMaskPredictor`가 지키는 것:

- 원소는 길이 1
- 앞에서 뒤 순서가 순위
- 중복 없음
- k개 이하 가능

ARI 리스트도 이 계약을 지키면 `run_fill_mask_eval.py`를 수정하지 않아도 된다.

### 7.2 Exact@10

기존 CSV는 top-10을 저장한다. 알고리즘은 k=10과 동일하고 `B1`만 넉넉히 둔다.  
공식 비교 주력은 Exact@5 (DECISIONS [014]).

### 7.3 smoke vs 본평가

| | smoke | 본평가 (EXP-015 비교) |
|---|---|---|
| 방법 | `generate()` Top-1 | Appendix C Top-5 |
| Exact@5 보고 | 금지 (`comparable=false`) | 허용 |
| 구현 출처 | PUBLIC HF 예시 | PAPER + OURS |

---

## 8. 구현 함수

GPU-free 코어는 `src/eval/ari/topk.py`. `predictors.py`는 수정하지 않는다. `predict_topk_hanja`는 평가 진입점이며 아직 없다.

| 함수 | 책임 | 근거 |
|---|---|---|
| `json_value_prefix_d1()` | `'{"[D1]": "'` | PUBLIC `json.dumps` + OURS |
| `classify_decoded_span(s)` | INCOMPLETE / COMPLETE_HANJA / INVALID | OURS, `is_cjk_ideograph` |
| `expand_character_candidates(...)` | §5.2 트리 | PAPER + OURS |
| `score_mean_logprob(logprobs)` | `mean(ℓ)` | PAPER |
| `dedupe_keep_best(chars)` | 같은 글자 최고점 | OURS |
| `order_candidates(chars)` | score desc + tie-break | PAPER + OURS |
| `predict_topk_hanja(row, k=5)` | 행 → `(chars, scores)` | 평가 진입점 |

로그확률은 `log_softmax`여야 한다. raw logit 평균은 논문과 다르다 (**OURS** 고정).

엔진: 1차는 **transformers forward** (vLLM은 논문 최적화). 알고리즘이 같으면 Exact@k는 같아야 한다. T4 사용 여부는 이 문서에서 확정하지 않는다.

---

## 9. 테스트 전략 (구현 단계, 지금 실행하지 않음)

GPU 없이 tokenizer fixture로 가능한 것:

1. 단일 토큰 한자: 완결 1스텝, `score == ℓ_1`.
2. 2–3 토큰 한자: 1스텝 topk 리스트에 gold 글자가 없고, 트리를 타면 gold가 산다.
3. 모든 출력 글자 `len==1` and `is_cjk_ideograph`.
4. 중복 없음.
5. 정렬이 `mean logprob` 내림차순.
6. `rank_of_gold(["字1","字2","字3","字4","字5"], "字3") == 3` → Exact@1=0, Exact@5=1.

가짜 점수 예 (단위 테스트용, 실모델 아님):

```text
candidates ordered by mean logprob:
  字1  -0.40
  字2  -1.10
  字3  -1.50
  字4  -2.00
  字5  -2.40
gold = 字3
Exact@1 = 0
Exact@5 = 1
```

---

## 10. 미결정 항목 (UNKNOWN)

구현 착수 전에 기본값을 run_meta에 박고 바꾼다.

| 항목 | 기본 (OURS) | 바꾸면 생기는 일 |
|---|---|---|
| 1st-token beam `B1` | 32 | 너무 작으면 다토큰 gold 누락 |
| 이후 스텝 `B2` | 8 | 2nd-token 다양성 |
| 최대 토큰 `Lmax` | 4 | 3을 넘기는 한자 절단 |
| 2nd 레벨 분기 vs greedy 연장 | 분기 | greedy는 빠르지만 Fig. 12와 덜 닮음 |
| 동일 한자 다중 경로 | max score | 합치면 순위 변동 |
| JSON prefix 공백 | `": "` (`json.dumps`) | prefix가 틀리면 Top-1이 `{` 또는 공백 |
| vLLM vs transformers | transformers 1차 | 수치 엔진 차이 가능 → 기록 |
| 평균 vs 합 | 평균 (PAPER) | 합은 ablation만 |

이 값들을 논문 재현값이라고 쓰지 않는다. 공개 구현이 없다.

---

## 11. Go / No-Go (Top-K만)

EXP-015 Exact@5와 나란히 두려면:

1. 후보가 **한 글자 한자** k개(또는 그 이하), 토큰 조각 없음.
2. 점수가 **구성 토큰 평균 logprob**.
3. 순서가 그 점수 내림차순.
4. gold가 2–3 토큰인 fixture에서 next-token Top-5와 결과가 달라질 수 있음을 테스트로 확인.
5. `generate()`-only 리스트를 Exact@5로 보고하지 않음.

하나라도 아니면 Exact@1만 비교하거나, 비교를 보류한다.

---

## 12. 출처 요약표

| 내용 | 태그 |
|---|---|
| 미완 바이트 후 문자가 끝날 때까지 생성 | PAPER |
| 점수 = 구성 토큰 logprob **평균** | PAPER |
| 순서 = 그 점수 내림차순 | PAPER |
| 다음 마커는 Top-1에 조건 (우리는 불필요) | PAPER |
| vLLM로 Top-K 추출 | PAPER, 코드 없음 |
| JSON `{"[Dn]": "X"}` 학습/출력 | PUBLIC |
| `json.dumps`가 타깃 문자열 | PUBLIC |
| `generate()` Top-1 예시 | PUBLIC |
| constrained decoding 소스 | **없음 → OURS 재구현** |
| JSON value prefix teacher-force | OURS |
| valid = `is_cjk_ideograph` 한 글자 | OURS |
| beam 폭 B1/B2, Lmax, 타이브레이크 | OURS (UNKNOWN 채움) |
| Exact@k = 기존 `rank_of_gold` | 우리 evaluator, 변경 없음 |
