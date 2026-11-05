# 발표 Q&A 초안

> **번호 체계:** 노트북 EXP-006 FT = 공식 **EXP-011**. 노트북 GPU EXP-007 FT = 공식 **EXP-012** (Colab, **완료 · DECISIONS [010]**). 공식 EXP-007은 SillokBERT **zero-shot**(완료).

## 데이터·split

**Q: train/test에 같은 비문이 섞이지 않았나요?**
A: inscription_id 단위 split. mask dataset 기준 train∩test=0 확인 (`work/step1_verification.json`).

**Q: 실제 결손 `□`로 성능을 재었나요?**
A: 아니요. 정답이 없어 **인공 single-char mask**만 Exact@k. 실제 결손은 데모(정성)만.

**Q: 한글/OCR 노이즈는?**
A: `is_training_text=True` 영역 한글 0자. review/결손 row는 학습 제외 (DECISIONS [006]).

## 평가·숫자

**Q: n=2000과 n=9719 차이?**
A: EXP-011(CPU FT) 비교표는 subsample 2000. frequency/ngram·zero-shot·**EXP-012 FT** full은 9719.
**EXP-011 Exact@5 50.45%를 EXP-012 full-test(53.04%@5)와 동일시하지 말 것** — 표본이 다르다.
EXP-012: Exact@1 **33.75%** / @5 **53.04%** / @10 **60.47%** (n=9,719).

**Q: Exact@k 정의?**
A: 정답 한자가 모델 top-k 후보에 포함된 비율.

**Q: UNK 2.4%는?**
A: SillokBERT vocab이 1토큰으로 못 쓰는 test 정답 비율. 학습은 UNK skip, eval은 포함.

## 모델

**Q: 왜 SillokBERT가 BERT보다 좋나요?**
A: 실록 continued pretraining + fine-tune. 동일 n=9,719에서 EXP-007 zs(24.9%@1) → EXP-012 FT(33.75%@1).
다만 정형구 overlap으로 **일부는 암기**일 수 있음 — 한계로 명시. unseen 구간까지 우위라고 주장하지 않음.

**Q: fine-tuning이 희귀 한자도 개선했나요?**
A: **아닙니다.** EXP-013 층화 기준 Exact@5 개선폭은 very_common +13.7pp / common +10.6pp / mid +6.5pp /
rare +1.3pp로 빈도에 비례하고, **hapax(1회 등장)는 8.2% → 6.0%로 하락**, unseen은 3.7%로 변동 없습니다.
학습 분포에 모델이 맞춰진 결과로 해석하며, 희귀자 일반화는 이번 범위에서 해결하지 못한 한계입니다.
(`reports/phase4_stratified_analysis_2026-08-01.md`)

**Q: 그럼 전체 +9.9pp는 과장 아닌가요?**
A: 전체 평균은 test 분포(고빈도 한자 비중이 큼)를 반영한 값이라 정확합니다. 다만 "모든 구간 개선"은 사실이 아니므로
구간별 표를 함께 제시합니다. 실제 판독 현장에서 정형구·빈출자 후보 제시가 우선순위라면 실용적 이득은 유지됩니다.

**Q: smoke 결과는?**
A: **성능 근거로 사용하지 않음.** (`_exp012_smoke` 메타·동명 파일 혼동 주의)

**Q: EXP-011 vs EXP-012?**
A: EXP-011 = CPU subsample baseline (train ~8k, test 2k). EXP-012 = Colab GPU full train (46k) + full test 9719 — **별도 완료 실험**, EXP-011 덮어쓰기 안 함. 공식 full-test 비교는 EXP-012.

## 범위

**Q: 연속 결손 `○○○`?**
A: 이번 baseline은 **한 글자 마스크만**. 연속 결손은 future work.

**Q: Group@k / 이체자?**
A: 매핑 소스 미확정 → 미구현. 후보 지표.

---

*EXP-012 full-test 수치 반영 완료. 제출 전 `docs/RISK_CHECKLIST.md` 점검.*

## 희귀자·hapax (Q&A 보강)

**Q: fine-tuning 후 희귀 글자(hapax) 성능이 떨어지지 않나요?**
A: 전체 Exact@k는 올랐지만 hapax는 표본이 작아 단정 불가. EXP-014(가설): EXP-011 subsample hapax n=48에서 ZS·FT 모두 거의 실패(both_miss5 46/48), ZS→FT 명확 하락은 1건. UNK는 hapax 정답의 18.5%(43/232)만 설명하며 FT 전후 동일. 자세한 내용: `reports/exp014_hapax_decline.md`.
