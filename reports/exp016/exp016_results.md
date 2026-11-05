# EXP-016 — 학습·평가 입력 형식 정합 진단 + context window 추론 스윕

> **상태:** completed · 학습 없음 · DECISIONS [012] iteration 프로토콜  
> 체크포인트: `notebooks/best/` (EXP-012 best, 변경 없음) · 실행일 2026-08-02

## 1. 선행 확인

| 항목 | 결과 |
|---|---|
| 최신 EXP | EXP-015 (2026-08-01). **EXP-016 사용 가능** (충돌 없음) |
| 실제 학습 코드 | `src/data/finetune_sillokbert.py` (`notebooks/exp012_colab.ipynb`는 이 모듈 CLI 호출) |
| 체크포인트 | `notebooks/best/` 존재 |

## 2. 코드 조사 판정

**종합: (a) 정합**

- **(b) 해당 없음** — 학습은 표준 `DataCollatorForLanguageModeling`이 아니라 **커스텀 단일-마스크**. 다중 vs 단일 마스크 불일치 가설 기각.
- **(c) 해당 없음(기본 조건)** — 학습·평가 모두 **context window ±64**. `max_seq_len=512`는 학습 시 초과 샘플 스킵 상한이며, eval tokenizer truncation 상한과 대응. EXP-012 로그에 둘을 병기한 것은 잘못 섞인 eval 설정이 아니라 **학습 파이프라인 상수**.

### 2.1 마스킹 로직

저장소에 `DataCollatorForLanguageModeling` / `mlm_probability` **없음**.

```150:193:src/data/finetune_sillokbert.py
def _build_mlm_examples(df: pd.DataFrame, tokenizer, *, label: str = "") -> list[dict]:
    ...
        left = max(0, mask_pos - CONTEXT_WINDOW)
        right = min(len(original), mask_pos + CONTEXT_WINDOW + 1)
        ...
        masked_window = window[:local_mask] + tokenizer.mask_token + window[local_mask + 1 :]
        ...
        labels = [-100] * len(input_ids)
        labels[mask_index] = label_id
        if len(input_ids) > MAX_SEQ_LEN:
            continue
```

```212:245:src/data/finetune_sillokbert.py
def _collate_batch(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    # padding only — no 15%/80%/10%/10% random MLM corruption
```

### 2.2 문맥 길이

```17:18:src/data/finetune_sillokbert.py
MAX_SEQ_LEN = 512
CONTEXT_WINDOW = 64
```

```5:15:src/eval/mask_window.py
CONTEXT_WINDOW = 64
MAX_SEQ_LEN = 512
def build_masked_window(..., context_window: int = CONTEXT_WINDOW):
```

- `run_fill_mask_eval.py`에 `--context-window` CLI **없음** → EXP-012/015는 기본 ±64.
- `notebooks/best/finetune_metrics.json`: `"max_seq_len": 512`, `"context_window": 64`.

## 3. Context window 실험 결과

- split: valid subsample **n=2,000**, seed=**42** (test 미사용)
- 체크포인트: EXP-012 best 고정
- ±64 기준선: EXP-015 full-valid 예측을 동일 subsample로 필터 (EXP-015와 동일 ±64·정규화)
- ±128/±256: 기존 코드 미수정, `outputs/exp016/run_context_window_sweep.py`에서 `_window_for_row`만 monkeypatch
- 후보 정규화: 단일 문자·`##` 제외, top-k=10 (기존 `FillMaskPredictor`와 동일)

| 조건 | Exact@1 | Exact@5 | Exact@10 |
|---|---:|---:|---:|
| ±64 (기준) | **34.30%** | **53.40%** | **60.70%** |
| ±128 | 33.30% | 52.65% | 60.75% |
| ±256 | 27.10% | 44.55% | 50.90% |

### 기준선(±64) 대비 paired (McNemar, continuity correction, n=2000)

| 비교 | k | Δpp (B−A) | only_64 / only_alt | p-value | 유의? |
|---|---:|---:|---|---:|---|
| ±128 vs ±64 | 1 | −1.00 | 64 / 44 | 0.068 | 아니오 (α=0.05) |
| ±128 vs ±64 | 5 | −0.75 | 64 / 49 | 0.188 | 아니오 |
| ±128 vs ±64 | 10 | +0.05 | 51 / 52 | 1.000 | 아니오 |
| ±256 vs ±64 | 1 | −7.20 | 209 / 65 | 5.7e−18 | **예 (악화)** |
| ±256 vs ±64 | 5 | −8.85 | 251 / 74 | 1.6e−22 | **예 (악화)** |
| ±256 vs ±64 | 10 | −9.80 | 248 / 52 | 2.1e−29 | **예 (악화)** |

보조 진단: 동일 subsample에서 tokenizer `max_length=512` truncation 시  
`over512` = ±64: 0, ±128: 0, ±256: **877/2000**. 마스크 토큰 소실은 0건(중앙 마스크라 truncate 후에도 잔존). ±256 하락은 마스크 유실이 아니라 **학습 분포(±64)와 어긋난 장문맥 + 비대칭 truncate**로 해석.

## 4. 결론

**재학습 없이 추론 context window를 키워 성능을 개선할 여지는 없다.**

- 학습·평가 형식은 이미 **(a) 정합**. 다음 단계 HP 튜닝의 전제로 “마스킹 방식 불일치 먼저 고치기”는 **불필요**.
- ±128은 Exact@5 −0.75pp, 비유의. ±256은 Exact@5 −8.85pp로 **유의하게 악화**.
- **유지:** 추론·평가 context window **±64** (학습과 동일).
- **EXP-017 (ZS/FT 보간):** 입력 창은 ±64로 고정해 비교. 창 확대는 후보에서 제외.
- **EXP-018 (HP 재탐색):** 마스킹은 현 단일-마스크 유지가 타당. 문맥 길이를 바꿀 경우 **학습·평가를 함께** 바꿔야 하며, 추론만 바꾸는 건 이번에 기각.

## 5. 확인하지 못한 항목

| 항목 | 이유 |
|---|---|
| ±64를 본 드라이버로 재추론한 수치 | 시간·정합을 위해 EXP-015 예측 재사용. 동일 코드 경로·동일 subsample이므로 paired에는 문제 없으나, 드라이버 자체 재현 수치는 별도 미측정 |
| GPU 추론 재현 | 로컬 CUDA 미가용 (cpu). 상대 비교에는 영향 적음 |
| Colab 세션의 런타임 패치 여부 | `exp012_colab.ipynb`는 `finetune_sillokbert` CLI 호출. 세션 중 임시 패치는 로그상 근거 없음 |
| 학습 시 ±64 창의 문자→토큰 경계 상세 | 창은 문자 기준; 토큰 경계 불일치는 학습·평가 공통 |

## 6. 산출물

| 경로 | 내용 |
|---|---|
| `outputs/predictions/exp016_cw64_valid_sub2000_predictions.csv` | ±64 기준 (EXP-015 필터) |
| `outputs/predictions/exp016_cw128_valid_sub2000_predictions.csv` | ±128 추론 |
| `outputs/predictions/exp016_cw256_valid_sub2000_predictions.csv` | ±256 추론 |
| `reports/exp016/exp016_context_window_results.json` | 요약·paired 전부 |
| `reports/exp016/paired_cw*_vs_cw64_k*.json` | k별 paired |
| `outputs/exp016/run_context_window_sweep.py` | 기존 src 미수정 드라이버 |
