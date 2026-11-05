# 발표 슬라이드 골격 (v0.3)

> **번호 체계 안내:** 이 문서의 일부 EXP-XXX는 2026-07-04 노트북 작업 시점 번호다.
> - 노트북 **EXP-006 Sillok FT** = 공식 **EXP-011** (CPU subsample, test n=2,000)
> - 노트북 **GPU EXP-007 FT** = 공식 **EXP-012** (Colab GPU full-data, test n=9,719, **완료 · DECISIONS [010]**)
> - 공식 **EXP-007** = SillokBERT zero-shot (이미 완료; GPU FT가 아님)

**현재 확정 표:** §4 = subsample (EXP-011). **§5 = full-test 공식 비교 (EXP-012).**  
경량 스냅샷: `reports/exp012_results.md`

---

## 1. 문제 정의 (1장)

- 훼손 금석문(비석·탁본)의 결손 한자 `□` → AI top-k 후보
- 목표: **완전 자동 복원 X**, 검수 가능한 후보 추천
- 문자(한자) 단위 MLM

## 2. 데이터 (1~2장)

- NRICH 판독문 → `readings.csv` 14,702행
- 학습: `is_training_text=True` 7,158행, 실제 결손 제외
- 인공 마스크: **66,537** samples (single-char)
- Split: **inscription 단위**, train∩test = **0** (PASS)
- UNK 정답: test **2.4%** (SillokBERT vocab)

## 3. 방법 (1장)

- Masked LM, ±64자 window, max 512 tokens
- Baseline ladder: frequency → n-gram → BERT zs → SillokBERT zs → **fine-tune**
- 평가: Exact@1/5/10 (인공 마스크 test만)
- EXP-012 선택: validation MLM loss 최소 → `best/` 후 test 1회

## 4. 결과 — n=2000 subsample (1장) **현재 확정 = 공식 EXP-011**

| EXP (노트북 / 공식) | Exact@1 | Exact@5 | Exact@10 |
| --- | ---: | ---: | ---: |
| EXP-002b frequency | 2.3% | 6.8% | 10.8% |
| EXP-003b ngram | 15.7% | 28.7% | 34.8% |
| EXP-004 BERT zs | 19.8% | 35.6% | 42.5% |
| EXP-005 Sillok zs | 23.7% | 42.6% | 51.0% |
| **EXP-006 / EXP-011 Sillok FT** | **31.3%** | **50.5%** | **57.8%** |

§4는 test n=2,000. §5(n=9,719)와 직접 병기·동일시하지 말 것.

## 5. 결과 — full test 9,719 (1장) **확정 = 공식 EXP-012**

| EXP | Exact@1 | Exact@5 | Exact@10 |
| --- | ---: | ---: | ---: |
| EXP-002 frequency | 2.3% | 6.6% | 10.3% |
| EXP-003 ngram | 14.5% | 28.1% | 34.7% |
| EXP-004b BERT zs | 19.7% | 35.0% | 42.4% |
| EXP-005b / EXP-007 Sillok zs | 24.9% | 43.2% | 51.2% |
| **EXP-012 Sillok FT (Colab GPU)** | **33.75%** | **53.04%** | **60.47%** |

## 6. 시대별 (조선 중심) (1장)

- EXP-011(노트북 EXP-006): 조선 n=1,546, Exact@1 **33.3%** (see `baseline_results_2026-07-03.md`) — subsample
- **EXP-012:** 조선 n=7,425, Exact@1 **35.87%** / @5 **55.39%** / @10 **62.67%** (`evaluation_summary.csv`)
- 두 표의 조선 n이 다르므로 직접 동일시하지 말 것. 통일신라 등 소수 class → 분산 주의

## 6-b. 정답 글자 빈도별 — fine-tuning이 어디를 올렸나 (1장, EXP-013)

Exact@5, test n=9,719:

| 구간 | n | EXP-007 zs | EXP-012 FT | 차이 |
| --- | --: | --: | --: | --: |
| very_common (100+) | 3,364 | 58.0% | **71.7%** | +13.7pp |
| common (21-100) | 3,690 | 44.9% | **55.5%** | +10.6pp |
| mid (6-20) | 1,588 | 27.6% | **34.1%** | +6.5pp |
| rare (2-5) | 602 | 20.8% | **22.1%** | +1.3pp |
| hapax (1회) | 232 | 8.2% | 6.0% | **−2.2pp** |
| unseen | 243 | 3.7% | 3.7% | 0.0pp |

- 이득이 **train 빈도에 비례**. hapax는 오히려 하락, unseen은 변화 없음.
- **"모든 구간이 개선됐다"고 말하지 말 것.** 정직하게 제시하는 편이 Q&A 방어에 유리하다.
- 근거: `reports/phase4_stratified_analysis_2026-08-01.md`

## 7. 데모 — 실제 결손 `□` (1~2장) **EXP-012 출처**

- Source: EXP-012 `exp012_sillokbert_gpu_best_valid_loss`
  - 전수 예측: `outputs/predictions/exp012_real_defect_predictions.csv` (3,704건, clean_only)
  - 1차 풀: `reports/expert_panel/expert_review_panel.csv` real_defect 25건
  - 발표 후보 7건: `reports/exp012_demo_cases.md` (**팀 검토 후 5건 확정**)
- **옛 EXP-011** `real_defect_demo_30.csv` 사례는 사용하지 않음 (모델·수치가 발표 본문과 불일치)
- **정량 평가 아님** (gold 없음). 점수는 상대 선호도. “맞았다” 서술 금지.

## 8. 한계 & Future work (1장)

- EXP-011: train 8k subsample / test 2k — **subsample baseline** (공식 full-test 아님)
- EXP-012: full train 46k + full test 9,719 **완료** (공식 full-test 비교 수치)
- **fine-tuning 이득은 고빈도 한자에 집중** — hapax 구간은 zero-shot 대비 하락 (EXP-013)
- test가 조선에 편중 (7,425 / 9,719 = 76%); 가야 n=2·신라 n=10은 수치 보고 부적합
- 단일 글자 마스크만; 연속 결손 미반영
- Future: glyph, Group@k, reranking, UI, 실제 결손 정성 확대

---

*수치 근거: `docs/EXPERIMENT_LOG.md` EXP-012, `reports/exp012_results.md`, `notebooks/best/`.*
