# EXPERIMENT_LOG.md — 실험 기록

**돌린 실험은 전부 여기에 기록한다. 기록 없는 실험 결과는 성과로 인정하지 않는다.**
이 로그가 있어야 (1) 재현이 가능하고, (2) 발표에서 성능 향상 과정을 보여줄 수 있다.

> 현재 상태(2026-09-17 갱신 · EXP-018 GPU Phase B 실행 확인은 2026-08-30 · EXP-018 post-hoc Exact@k 평가는 2026-09-17): Phase 1–3 baseline, EXP-011 subsample FT, **EXP-012** full-data FT+test,
> **EXP-013** 층화, **EXP-015** valid Exact@k 기준선, **EXP-016** 형식 정합, **EXP-017** ZS/FT 보간까지 완료.
> **EXP-018** Phase A(코드·Colab kit) 완료 / GPU Phase B: **018A·018B 학습(4 epoch) 완료** 및
> **post-hoc valid Exact@k 평가 완료**(2026-09-17, subsample n=2,000, seed=42, 전체 checkpoint-epoch-1~4 평가).
> 018A·018B 모두 DECISIONS [012] gate(subsample Exact@5 ≥ EXP-015 +0.7pp AND McNemar p<0.05)를
> **통과하지 못했다** — **공식 incumbent로 승격하지 않는다**. **018C는 여전히 미실행/결과 미확보.**
> **EXP-019**는 ARI-8B 통합·Top-K 평가의 설계 및 저장소(cross-split duplicate) 감사까지 완료됐으며,
> 실제 코드 구현·모델 다운로드·GPU inference·전체 validation은 **아직 수행하지 않았다.**
> 동일 test n=9,719 공식 비교는 EXP-012. 일상 FT iteration은 EXP-015 valid + paired.
> 현재 공식 valid 비교 기준선은 여전히 **EXP-015 valid**(n=10,679; Exact@1/5/10 = 33.78 / 53.24 / 60.90)이다.
> EXP-018은 이제 Exact@k 수치를 확보했으나 gate 미통과로 EXP-015를 대체하지 못한다.
> EXP-019는 아직 Exact@k 수치가 없어 여전히 비교 baseline이 아니다.
> EXP-017 결론: 선형 보간은 FT(EXP-015) overall을 유의 개선하지 못함 → 공식 후보는 FT 유지.
> **EXP-015는 신규 모델(ARI-8B, retrieval/BM25, reranking 등) 비교의 공식 incumbent validation
> baseline으로 고정됨(DECISIONS [014]).**
> EXP-019의 critical path는 **019A no-retrieval → 019B BM25 → EXP-015와 동일 Exact@k 비교**다.
> 019C ChromaDB/dense retrieval은 담당자 이탈에 따라 `deferred / optional`이다.
> **다음 최우선 실험은 EXP-019(ARI Top-K decoder 구현 → EXP-019A)다. EXP-018 계열의 추가 조치(018C 등)는 후순위다.**

> **2026-10-01 추가 (위 9월 17일 문단은 그때의 스냅샷):** 발표 S5 Exact@k는 **EXP-025 동일 n=200 근사 Exact(beam5)** 로 고정한다(DECISIONS [017]). 공식 ARI Appendix C Top-K도 SOTA도 아니다. EXP-023은 이미 완료(재실행 없음). EXP-026은 건너뜀. EXP-028 정성 비교는 완료이며 Exact@k가 없다. EXP-027 A100 full train은 야간 진행 중이고 비공식이다. 027·028 수치는 S5/S10에 넣지 않는다.
> **EXP-029**는 one-pass KV cache + `expand_character_candidates`의 **mean token logprob** Top-K(Appendix C–style, 공식 ARI Top-K 아님)를 로그에만 남긴다. `update_presentation_s5 = false`. S5 잠금(Sillok **0.515** / ARI-FT beam5 **0.410** / ZS **0.225**, n=200)은 그대로다. 0.4115와 Sillok 0.534를 S5와 같게 읽지 않는다. EXP-027은 별개이며 아직 돌고 있을 수 있어, 이 갱신은 027 결과를 적지 않는다(DECISIONS [018]).

> **2026-10-02 갱신:** **EXP-027** 완료(10-02 01:59 KST, [017] 기한 안). train 전체 1 epoch, beam5 n=200 Exact@1/5 **0.350 / 0.525** — 같은 200에서 EXP-024(0.220 / 0.410)보다 유의하게 높고 Sillok(0.350 / 0.515)과는 유의차 없음 → [017]대로 **동등, 비공식**. **EXP-030** 세 variant 학습 완료(압축 프롬프트가 약 4.1배 빠름), 채점 전. S5 잠금은 그대로다.

작성 방법: 새 실험은 고유 번호로 추가한다. 실행일과 검증·채택일이 다르면 둘 다 기록한다. 아래 템플릿을 복사해서 채운다.

---

## 평가 지표

| 지표 | 의미 | 비고 |
|---|---|---|
| `Exact@1` | top-1 후보가 정답과 정확히 일치하는 비율 | 가장 엄격 |
| `Exact@5` | 정답이 top-5 후보 안에 있는 비율 | 현재 주력 지표 |
| `Exact@10` | 정답이 top-10 후보 안에 있는 비율 | 검토 보조 |
| `Group@1/5/10` | 정답과 같은 이체자 그룹이 top-k 안에 있는 비율 | 매핑 출처·규칙 확정 전에는 공식 보고 금지 |

> 모델·하이퍼파라미터·체크포인트 선택은 validation으로 수행한다. test는 최종 비교에 제한적으로 사용하고 평가 목적과 횟수를 기록한다. 실제 결손은 정답이 없으므로 인공 마스크 `Exact@k`와 별도로 정성 평가한다.
>
> **FT 개선 루프 (DECISIONS [012]):** 일상 iteration은 valid subsample n=2,000(`seed=42`) Exact@5 + **paired** 비교(`src/eval/paired_exact_compare.py`). 마일스톤은 full valid n=10,679. 기준선은 EXP-015(EXP-012 best valid Exact@k). EXP-012 이후 full test 잔여 예산 **2회**(사용 시점은 팀 논의). 발표·공식 비교용 test 수치는 EXP-012 유지.

## 템플릿 (복사해서 사용)

```
### EXP-XXX (YYYY-MM-DD) — 한 줄 제목

- 유형/상태: (`DATA`/`AUDIT`/`EVAL`/`TRAIN`/`SMOKE`; `planned`/`completed`/`invalidated`/`superseded`)
- 목적/가설: (이 실험으로 무엇을 확인하려 하는가)
- 데이터 버전: (예: corpus_v1)
- split 방식: (예: inscription 단위, seed=42, 70/15/15)
- 모델: (예: bert-base-chinese)
- 코드/환경: (Git commit, Python·라이브러리, CPU/GPU)
- 하이퍼파라미터: (lr, batch size, epochs, max_len 등)
- 모델 선택 기준: (validation 지표·체크포인트 선택 규칙)
- test 사용: (사용 여부, 목적, 표본 수, 누적 평가 횟수)
- 평가 지표 결과:
    - Exact@1:
    - Exact@5:
    - Exact@10:
    - Group@1/5/10:
    - 기타:
- 결과 요약: (숫자로 한두 줄)
- 해석: (왜 이런 결과가 나왔는가, 무엇을 알게 됐는가)
- 다음 실험: (이 결과를 바탕으로 다음에 무엇을 할 것인가)
- 산출물 경로: (검수 완료 데이터 기반 결과 csv/로그 위치)
- 보존 상태: (Git 추적 여부, 외부 백업 위치, SHA-256)
- 한계: (subsample, UNK, 중복 문맥, 미검증 조건 등)
```

---

## 실험 기록 (최신순)

### EXP-030 (2026-10-02 새벽 학습 · 요약 저장 01:28 / 02:04 / 02:30 KST) — FT 압축 프롬프트 ablation (학습 완료, 채점 전)

- 유형/상태: `TRAIN` / **학습 완료, 평가 미실행**. Exact@k 없음. 채택 판정 전.
- 목적/가설: EXP-024와 같은 5,000행·1 epoch·하이퍼파라미터에서 system prompt만 바꿔, few-shot을 뺀 압축 프롬프트가 성능을 유의하게 잃지 않는지 본다(`docs/ARI_FINETUNE_PLAN.md` P1, 규칙 `noninferior`).
- 데이터/split: `_select_rows(train, 5000, seed=42)`, 세 variant 모두 `--cover-max-length`(PR #35) — `data.cover_max_length=true`, `pool_dropped_over_length=[]`, `n_train_unique=5000`. 학습 행이 같다.
- 모델/환경: ARI-8B NF4 + LoRA r=16/α=32, 1 epoch, lr 2e-4, effective batch 16. **A100-SXM4-40GB**, transformers 5.16.1, peft 0.20.0.
- 학습 결과:

    | variant | max_seq_len | per-device × accum | step당 | 학습 시간 | peak GiB | train_loss |
    |---|---|---|---|---|---|---|
    | (a) `ari` | 1,192 | 8 × 2 | 7.60초 | 2,408초 (40.1분) | 34.2 | 0.4568 |
    | (b) `compact` | 288 | 16 × 1 | 1.81초 | 581초 (9.7분) | 17.2 | 0.4644 |
    | (c) `compact_era` | 288 | 16 × 1 | 1.82초 | 583초 (9.7분) | 17.2 | 0.4630 |

    - 세 run 모두 313/313 step, loss 유한, adapter 저장.
    - 압축 프롬프트는 학습 시간이 **약 4.1배 짧다**(2,408 → 581초; step당 7.60 → 1.81초).
    - train_loss는 프롬프트가 달라 직접 비교하지 않는다(loss는 target 토큰에만 걸리지만 조건부 분포의 맥락이 다르다).
- 평가 지표 결과: *(미실행 — 기입하지 않음)*. one-pass n=2,000 채점 후 `judge_ablation.py --rule noninferior`로 판정한다.
- 결과 요약: 학습만 끝났다. 압축 프롬프트의 비용 절감(약 4배)은 확인됐고, 성능 판정은 남았다.
- 다음 실험: (a)(b)(c) adapter를 `score_exp029 --prompt-variant <v> --n 2000`으로 채점 → `judge_ablation --rule noninferior` → 채택 variant를 EXP-031/032에 쓴다. EXP-027 결과(학습량 효과)를 보고 이후 실험을 train 전체 기준으로 옮길지 팀이 정한다.
- 산출물 경로: Drive `MyDrive/exp030/{ari,compact,compact_era}/full/` (`adapter_final`, `exp024_full_summary.json`), `MyDrive/exp030/compact/compact_train.log`, zip `MyDrive/exp030/exp030_colab_sooyah_min_cover.zip`.
- 한계: 5,000행. (a)는 이 브랜치 트레이너로 다시 학습한 것으로 EXP-024 adapter와 별개다(train_loss 0.4568 vs EXP-024 0.4577). GPU가 EXP-024/027(80GB)과 다르다(40GB) — 시간 비교는 세 variant 사이에서만 한다.

### EXP-029 (2026-10-01) — ARI-8B QLoRA one-pass mean token logprob Top-K (n=200 pilot → n=2000)

- 유형/상태: `EVAL` / `completed`. Appendix C–style. **공식 ARI Top-K가 아니다.** `update_presentation_s5 = false`.
- 목적/가설: EXP-024 QLoRA adapter를 beam5 `generate()` 근사 대신, one-pass KV cache + `expand_character_candidates`로 Top-K를 채점한다. 순위 함수는 **mean token logprob**다.
- 데이터/split: valid. 표본 순서는 `_select_rows(valid, 2000, seed=42).head(n)` (EXP-021/024와 동일). test 사용 없음.
- 모델: ARI-8B (`DAMI-Lab/ARI-8B`) QLoRA. adapter `MyDrive/exp024/full/adapter_final` (EXP-024).
- 코드/환경: `src/eval/ari/score_exp029.py`. PR #30이 `exp024/ari-qlora`에 병합됨. Colab 산출물은 Drive에 둔다.
- 프로토콜:
    - 방법: **mean token logprob** Top-K. one-pass KV cache + `expand_character_candidates`(k=10, b1=32, b2=8, lmax=4). Appendix C–style.
    - 순위는 mean token logprob. sum logprob는 raw jsonl에만 남긴다.
    - **공식 ARI Top-K가 아니다.** beam5 `generate()` 근사(EXP-021/024/025)와도 다른 점수다.
- 평가 지표 결과:
    - n=200 pilot: Exact@1 / Exact@5 / Exact@10 = **0.205 / 0.385 / 0.445**. beam5 top1_agreement **0.715**.
    - mismatch review: mean token logprob와 beam(sum)의 **예상된 차이**이며 버그가 아니다. **GO** for n=2000.
    - n=2000 FT: Exact@1 **0.239** / Exact@5 **0.4115** / Exact@10 **0.4875** (n_scored=2000).
    - multi-token gold n=144: Exact@5 **≈0.361**.
- paired vs EXP-015 Sillok valid (동일 n=2000):
    - Exact@1: Sillok **0.343** vs ARI **0.239** (Δ−10.4pp, only_S 345 / only_ARI 137, McNemar p≈4e-21).
    - Exact@5: Sillok **0.534** vs ARI **0.4115** (Δ−12.3pp, only_S 417 / only_ARI 172, McNemar p≈9e-24).
    - Exact@10: Sillok **0.607** vs ARI **0.4875** (Δ−12.0pp, only_S 405 / only_ARI 166, McNemar p≈2e-23).
- 결과 요약: 동일 n=2000에서 Sillok(EXP-015)이 이 one-pass FT보다 Exact@1/5/10 모두 유의하게 높다. 발표 S5 숫자는 아니다.
- 해석: n=200 beam5 top1_agreement 0.715는 mean token logprob와 beam(sum)의 예상 divergence다. n=2000 Exact@5 **0.4115**를 S5의 ARI-FT beam5 **0.410**과 같게 읽지 않는다. paired Sillok Exact@5 **0.534**(n=2000)를 S5의 **0.515**(n=200)와 같게 읽지 않는다.
- 발표 잠금: **S5는 바꾸지 않는다** (`update_presentation_s5 = false`, DECISIONS [017]·[018]). 잠금값은 그대로 Sillok **0.515** / ARI-FT beam5 **0.410** / ZS **0.225** (EXP-025, n=200). SOTA가 아니고, 공식 ARI Appendix C Top-K도 아니다.
- 다음 실험: EXP-027 야간 full train은 **별개**이며 아직 돌고 있을 수 있다. 이 절은 027 결과를 적지 않는다. 다음 축은 Claude 이후 **EXP-030+**.
- 산출물 경로: Drive `MyDrive/exp029/` — summary, n=2000 predictions (`exp029_qlora_onepass_n2000_predictions.csv`), raw jsonl (`exp029_qlora_onepass_raw.jsonl`), mismatch_review.
- 보존 상태: Colab 산출물은 Drive. 이 로그에 적은 숫자 밖에는 지어 넣지 않는다.
- 한계: Appendix C–style 구현이며 공식 ARI Top-K가 아니다. valid subsample n=2000(seed=42). multi-token gold는 n=144. EXP-027 지표 없음.

### EXP-028 (2026-10-01) — real_defect 정성 비교 (Sillok vs ARI-FT, n=25)

- 유형/상태: `EVAL` qualitative / `completed`
- 목적/가설: 실제 결손(real_defect)에서 SillokBERT와 ARI-FT 후보가 어디서 갈리는지 본다. Exact@k는 계산하지 않는다.
- 데이터: real_defect **n=25**. gold 없음. test 사용 없음.
- 모델: SillokBERT(EXP-012) 기존 예측 재사용 + ARI-FT(EXP-024 `MyDrive/exp024/full/adapter_final`) beam5.
- 코드/환경: 브랜치 `exp024/ari-qlora`. import 수정 PR #26 병합 — `unique_hanja_candidates`는 `topk`가 아니라 `parse`에서 가져온다. Colab 산출물은 Drive에 둔다.
- 평가 지표 결과:
    - Exact@1 / Exact@5 / Exact@10: **없음.** 계산·보고하지 않는다.
    - 기타: side-by-side n=25에서 **Top-1 완전 일치 = 0**. "두 모델이 같다"는 데모 사례는 없다.
- 결과 요약: 정성 비교만 끝났다. 이 결과를 S5/S10 숫자에 넣지 않는다.
- 해석: **전문가 검토용 불확실 후보**이다. 복원 성공으로 말하지 않는다.
- 발표 데모로 고정 (2026-10-01, DECISIONS [017]). 028 자체의 Exact@k가 아니다.
    - EXP-028 정성 2건 (gold 없음): `real_000275` 歲[MASK]戌年 — Sillok 戊 / ARI-FT 甲 (Top-1 불일치). `real_000445` 向[MASK]下二里 — Sillok 山 / ARI-FT 此 (山은 Top-5에 겹침).
    - 같은 날 고정한 artificial_mask 검증 5건: 성공 `mask_00014815` (乙亥年八[MASK]→月), `mask_00003798` (引[MASK]生→門), `mask_00001848` (智[MASK]山→異). 실패 `mask_00005787` (檢校[MASK]子, gold 太 vs 世), `mask_00056976` (大[MASK], gold 干 vs 師).
- 다음 실험: EXP-027 야간 full train(비공식). EXP-026은 건너뜀. EXP-023은 재실행하지 않는다.
- 산출물 경로: Drive `MyDrive/exp028/exp028_sillok_vs_ari_ft_side_by_side.csv` 및 같은 폴더의 raw·summary.
- 보존 상태: Colab 산출물은 Drive. 선정 메모는 저장소 밖 `/workspace/research-notes/2026-10-01_verification-examples.md` (이 저장소에 파일을 만들지 않음).
- 한계: gold가 없어 Exact@k가 없다. n=25. beam5 근사이며 공식 ARI Appendix C Top-K가 아니다.

### EXP-027 (2026-10-01 착수 · 2026-10-02 01:59 KST 학습 완료) — ARI-8B QLoRA train 전체 1 epoch (비공식)

- 유형/상태: `TRAIN`+`EVAL` / `completed` (비공식 근사 Exact, beam5 n=200)
- 목적/가설: EXP-024(5,000행)와 같은 트레이너·하이퍼파라미터로 train split 전체를 학습해 데이터 양의 효과를 본다. 결과는 비공식이다. S5/S10 슬라이드 숫자를 바꾸지 않는다.
- 데이터/split: train **46,139행 전체**(`_select_rows(train, None, seed=42)`, `pool_limit=null`), 길이 초과 제외 0, invalid window 0. 평가는 valid EXP-021/024 순서 head(200), beam5. test 사용 없음.
- 모델/환경: ARI-8B NF4 + LoRA r=16/α=32, 프롬프트 `ari`(긴 SYSTEM_PROMPT, era unknown). A100-SXM4-80GB, torch 2.11.0, transformers 4.57.6, peft 0.20.0. 실행 `--mode full --train-pool -1 --save-steps 200`.
- 하이퍼파라미터: 1 epoch, lr 2e-4 cosine, warmup 3%, effective batch 16 (per-device 16 × accum 1), max_seq_len 1,200, seed 42.
- 학습 결과: global_step **2,884/2,884**, train_loss **0.3956** (EXP-024 0.4577). loss 로그 0.75 → 0.34, 모두 유한. step당 약 7.1초, 학습 **약 5.7시간**(20,569초), peak reserved 73.3 GiB. adapter `MyDrive/exp027/full/adapter_final`.
- 중단 조건([017], 2026-10-02 오전까지 미완이면 중단): **충족.** 학습 요약은 2026-10-01T16:59Z(= 10-02 01:59 KST), 평가는 17:32Z(= 02:32 KST)에 저장됐다.
- 평가 지표 결과 (n=200 beam5 근사 Exact, EXP-024와 같은 표본·같은 순서):
    - Exact@1: **0.350** / Exact@5: **0.525** / Exact@10: beam5라 Exact@5와 같다(0.525). 의미 없음.
    - parse: 출력 1,000개 중 993 ok(`not_single_hanja` 7), 표본당 후보 평균 4.965.
    - paired vs EXP-024 (5,000행, 같은 200): Exact@1 0.220 → 0.350 (+13.0pp, McNemar p≈1.2e-4), Exact@5 0.410 → 0.525 (+11.5pp, discordant 5/28, p≈1.3e-4, bootstrap 95% CI +6.5 ~ +17.0pp).
    - paired vs Sillok EXP-015 (같은 200): Exact@1 0.350 vs 0.350 (p≈0.87), Exact@5 0.515 vs 0.525 (+1.0pp, discordant 19/21, p≈0.874, 95% CI −5.5 ~ +7.0pp).
    - 계산: `src/eval/ari/judge_ablation.py --rule superior --expected-n 200` (EXP-025 same-200 CSV와 위 예측 CSV).
- 결과 요약: train 전체로 학습하면 같은 200표본에서 EXP-024보다 Exact@1·@5가 유의하게 높고(+13.0 / +11.5pp), Sillok과는 유의한 차이가 없다.
- 해석 (사전 합의 [017] 적용): Sillok 0.515와 "거의 같음" → **동등**으로 읽는다. 0.525가 0.515보다 1pp 높지만 n=200에서 유의하지 않으므로 "Sillok을 넘었다"고 말하지 않는다. 5,000행 결과(EXP-024/029)와의 차이 대부분은 학습량에서 온 것으로 보인다(같은 하이퍼파라미터, 행 수만 다름).
- 발표: S5 잠금값(Sillok 0.515 / ARI-FT beam5 0.410 / ZS 0.225)은 바꾸지 않는다. 이 절의 0.525는 S5에 넣지 않는다.
- 다음 실험: 같은 adapter를 EXP-029 one-pass n=2,000으로 채점해 Sillok n=2,000(0.534)과 paired 비교한다. EXP-030~032를 5,000행 기준으로 계속할지, train 전체 기준으로 옮길지 팀이 정한다.
- 산출물 경로: Drive `MyDrive/exp027/full/` (`adapter_final`, `exp024_full_summary.json` — 파일 이름은 트레이너 그대로), `MyDrive/exp027/eval/` (`eval_exp027_beam5_n200.log`, `exp024_qlora_beam5_n200_predictions.csv`, `exp024_qlora_eval_summary.json`, raw jsonl).
- 보존 상태: Colab 산출물은 Drive. 저장소에는 이 기록만 둔다.
- 한계: 비공식. beam5 근사(Appendix C 아님), n=200. 평가를 `eval_exp027` 래퍼가 아니라 `eval_exp024`로 돌려 예측 CSV·요약의 라벨이 `EXP-024`/`qlora_exp024`로 찍혀 있다(로그의 adapter 경로 `exp027/full/adapter_final`로 EXP-027임을 확인). 요약 JSON에는 `prompt_variant` 필드가 없다(EXP-030 이전 코드) — 기본 `ari`다.

### EXP-026 — 건너뜀

- 유형/상태: `skipped`
- 결과 요약: P1로 진행하지 않았다. 순서는 EXP-028 정성 평가 다음 EXP-027 야간 full train이다(DECISIONS [017]).
- 평가 지표 결과: 없음.

### EXP-025 (2026-10-01) — 동일 200표본 paired Exact + McNemar (EXP-015/021 vs EXP-024)

- 유형/상태: `EVAL` / `completed` (기존 예측 CSV inner join; 학습·추론 없음)
- 목적/가설: EXP-024 eval의 `sample_id`에 EXP-015(SillokBERT) · EXP-021(ARI-8B ZS beam5) · EXP-024(ARI-8B QLoRA FT beam5)를 맞춰, 동일 표본에서 Exact@k와 McNemar를 비교한다.
- split: valid, EXP-024 `sample_id` inner join, **n=200** (same-sample). test 사용 없음.
- 평가 지표 결과 (n=200, Exact@1 / Exact@5):
    - Sillok (EXP-015): 0.350 / 0.515
    - ARI-ZS (EXP-021): 0.105 / 0.225
    - ARI-FT (EXP-024): 0.220 / 0.410
- Exact@10 (n=200): Sillok **0.575**. ARI-FT 0.410 · ARI-ZS 0.225는 beam5라 Exact@5와 같다. 발표 슬라이드는 **ARI Exact@10 = n.a. (beam5)** 이며, 슬라이드용 beam10 재평가는 하지 않는다.
- McNemar (continuity correction): **FT ≻ ZS 유의**. **Sillok ≻ FT는 k=1과 k=5에서 유의**. 양쪽 비교 모두 유의하다.
- 결과 요약: 동일 200에서 QLoRA FT는 ZS보다 Exact@1/@5가 유의하게 높고, SillokBERT는 FT보다 Exact@1/@5가 유의하게 높다.
- 발표 잠금: S5 Exact@k의 근거는 이 절이다(DECISIONS [017]). SOTA가 아니고, 공식 ARI Appendix C Top-K도 아니다. EXP-027·028 수치로 이 표를 바꾸지 않는다.
- 한계: **공식 ARI Top-K가 아님.** Top-5는 EXP-021/024와 같은 단순화 beam search(Appendix C constrained decoding 아님). beam5라 ARI Exact@10 ≈ Exact@5. EXP-015 전체 valid나 n≈500 Exact@5 0.504를 EXP-024 n=200의 0.410과 직접 비교하지 말 것.
- 산출물 경로: `reports/exp025/` (`exp025_summary.json`, `paired_exp015_vs_exp024_k{1,5,10}.json`, `paired_exp021_vs_exp024_k{1,5,10}.json`, same-200 CSV 3종). Drive `MyDrive/exp025/paired_exact_compare.json`. 전체 valid 입력 CSV는 커밋하지 않음. PR #24가 `main`에 병합됨(EXP-025만).

### EXP-024 (2026-10-01) — ARI-8B QLoRA full train + n=200 beam5 평가

- 유형/상태: `TRAIN`+`EVAL` / `completed` (비공식 근사 Exact)
- 목적/가설: ARI-8B를 QLoRA로 학습한 뒤, EXP-021과 같은 beam5 근사로 n=200 Exact@k를 본다. 공식 ARI Appendix C Top-K가 아니다.
- 데이터/split: valid eval **n=200**, beam5. test 사용 없음.
- 모델/환경: ARI-8B QLoRA, A100 full. adapter `MyDrive/exp024/full/adapter_final`.
- 학습 결과: train_loss **0.4577**.
- 평가 지표 결과 (n=200, beam5 근사 Exact):
    - Exact@1: **0.220**
    - Exact@5: **0.410**
    - Exact@10: beam5라 Exact@5와 같다(0.410). 슬라이드에는 n.a. (beam5).
    - paired vs EXP-021 ZS(0.105 / 0.225): **+11.5 / +18.5 pp**
- 결과 요약: 같은 n=200에서 ZS보다 Exact@1·Exact@5가 높다. 발표에 쓰는 순위·유의성은 EXP-025(Sillok 포함)가 정본이다.
- 해석: 공식 Top-K·SOTA로 읽지 않는다. EXP-015 n≈500 Exact@5 0.504와 이 0.410을 직접 비교하지 않는다(DECISIONS [017]).
- 다음 실험: 동일 표본 paired는 EXP-025로 완료. 이어서 EXP-028 정성, 그다음 EXP-027(비공식, `--train-pool -1`).
- 산출물 경로: Drive `MyDrive/exp024/full/adapter_final`. 코드·노트북은 브랜치 `exp024/ari-qlora`.
- 보존 상태: adapter 등 Colab 산출물은 Drive. 이 로그에 적은 숫자 밖에는 지어 넣지 않는다.
- 한계: beam5 근사. Appendix C constrained decoding 아님. n=200.

### EXP-023 — 층화 커버리지 (이미 완료, 재실행하지 않음)

- 유형/상태: `ANALYSIS` / `completed`
- 결과 요약: 빈도 구간별 층화 커버리지는 이미 끝나 있다. **다시 돌리지 않는다.** 이 갱신에서는 지표를 다시 적지 않는다.
- 산출물 경로: `reports/exp023/` (정본).
- 다음 실험: 재실행 없음. 이후 순서는 DECISIONS [017] (028 다음 027).

### EXP-022 (2026-10-01) — 선택적 예측 임계값 test 재현성 확인 (EXP-020 valid 임계값 고정, EXP-012 test 후처리)

- 유형/상태: `ANALYSIS` / `completed` (기존 test 예측 CSV 재집계; 학습·추론·모델 로딩 없음)

- 목적/가설:
  EXP-020은 임계값 선택과 성능 측정을 같은 valid에서 해 낙관 편향 가능성이 있었다.
  valid에서 확정한 임계값을 **고정**한 채 test 예측에 적용해 coverage·Exact@k가 재현되는지 확인한다.
  test에서 임계값 재탐색·최적화는 하지 않는다(test coverage는 결과이지 조정 대상이 아님).

- 데이터 버전:
  test `notebooks/best/exp012_sillok_ft_full_predictions.csv` (EXP-012 기존 산출, n=9,719, gitignore 로컬),
  valid 참조 `reports/exp020/exp020_results.json`·`exp020_sample_confidence.csv`,
  빈도 구간용 `data/model_inputs/artificial_mask_dataset.csv` train label 빈도(EXP-013/020과 동일 정의)

- 모델:
  EXP-012 best(`exp012_sillokbert_gpu_best_valid_loss`) 예측 재사용. 재추론 없음.

- 코드/환경:
  `python scripts/analyze_selective_prediction_test.py --predictions <test 예측 CSV>`, CPU.
  confidence·Wilson·정렬 함수는 `scripts/analyze_selective_prediction.py`(EXP-020)에서 import, 수정 없음.

- 고정 임계값:
  top-1 확률 ≥ **0.617386** (EXP-020 `max_coverage.top1_score["exact_at_5>=90%"].fine.threshold`를 JSON에서 그대로 읽음).
  같은 규칙으로 valid를 다시 선택하면 n=2,683으로 EXP-020과 일치함을 스크립트에서 확인.
  반올림값 0.617을 쓰면 test 선택 집합이 동일(n=2,401)하여 결과 차이 없음.

- test 사용:
  **신규 test 추론 없음.** EXP-012에서 이미 산출된 test 예측의 재집계(EXP-013 층화 분석과 같은 성격).
  DECISIONS [012] test 예산 관점의 해석은 **팀 판단 미정**이며 두 해석을 모두 기록한다:
  - 해석 A — 재집계(예산 미소모): 모델·가중치·예측이 EXP-012에서 고정됐고 새 test forward pass가 없으므로
    EXP-013과 동일하게 잔여 예산 2회 유지.
  - 해석 B — 예산 1회 사용: 이번에는 test 결과를 보고 "선택적 예측 주장"의 채택 여부를 판단하는
    **새 결정 규칙(임계값)의 평가**에 test를 썼으므로, 의사결정 목적의 test 열람 1회로 보고 잔여 1회.
  어느 쪽이든 임계값은 valid에서 사전 고정했고 test 결과로 수정하지 않았다.

- sanity check (실측): **통과**.
  행 9,719, notes split=test. 재계산 Exact@1/5/10 = 0.337483 / 0.530404 / 0.604692 = EXP-012 공식값과 일치
  (저장 `is_exact_at_k`·`rank_of_gold` 불일치 0). score 범위 [0, 0.999998], 샘플별 top-10 합 최대 1.0000 → 확률(EXP-020과 동일 형식).

- 평가 지표 결과 (고정 임계값 0.617386, 실측; 괄호는 95% Wilson):

  | split | 커버리지 | 선택 n / 전체 | Exact@1 | Exact@5 | Exact@10 |
  |---|--:|--:|--:|--:|--:|
  | valid (EXP-020) | 25.1% | 2,683 / 10,679 | 80.5% (78.9–81.9) | 90.0% (88.8–91.1) | 91.9% (90.8–92.9) |
  | **test (EXP-022)** | **24.7%** | **2,401 / 9,719** | **80.2%** (78.5–81.7) | **89.4%** (88.1–90.6) | **91.5%** (90.3–92.6) |

  test − valid (독립 두 비율 z 검정, 참고): Exact@5 −0.6pp (95% CI −2.3~+1.1, p=0.49),
  Exact@1 −0.3pp (p=0.79), Exact@10 −0.4pp (p=0.60), coverage −0.4pp (p=0.49). 모두 유의하지 않음.

- coverage-accuracy 곡선 (참고용, 5% 단위; 주력 수치 아님):
  test Exact@5 커버리지 10/20/30/50%: 96.1% / 91.9% / 86.5% / 76.1% (valid 96.7 / 92.6 / 86.7 / 77.1).
  전 구간에서 test가 valid보다 0~1pp 낮은 수준으로 곡선이 거의 겹침.
  test 곡선에서 "Exact@5 90% 만족 최대 커버리지"는 test 튜닝이므로 산출·보고하지 않는다.

- 빈도 구간 교차 (선택 구간, test n=2,401; valid 대비):
  선택 비중 very_common 48.3%(valid 47.3%) · common 38.7%(38.9%) · mid 9.3%(9.8%) · rare 2.7%(2.0%) ·
  hapax 0.7%(1.0%) · unseen 0.4%(0.9%).
  선택 Exact@5: very_common 95.1%(95.1%) · common 88.5%(90.7%) · mid 81.7%(81.4%) · rare 54.7%(51.9%, n=64) ·
  hapax 23.5%(50.0%, n=17) · unseen 22.2%(20.0%, n=9).
  고빈도 쏠림과 "rare 이하는 90%에 크게 못 미침" 경향은 일치. hapax는 50.0→23.5%로 낮지만 n=17(CI 9.6–47.3%)로 표본이 작다.

- 결과 요약:
  valid 고정 임계값을 test에 적용하자 커버리지 24.7%(2,401건)에서 Exact@5 89.4%(95% CI 88.1–90.6%).
  valid 대비 −0.6pp로 소폭 낮고 90%를 근소하게 밑돌지만, 차이는 통계적으로 유의하지 않고 두 CI가 크게 겹친다.

- 해석:
  EXP-020의 "확신 상위 약 25%에서 Exact@5 약 90%"는 split을 바꿔도 거의 재현됐다 → 낙관 편향은 1pp 이내로 작다.
  다만 test 점추정치는 90% 미만이므로 발표에서는 "약 25% 커버리지에서 Exact@5 약 89–90% (valid 90.0%, test 89.4%)"처럼
  두 수치를 함께 쓰고 "90% 이상 보장"으로 표현하지 않는다. 고빈도·중빈도가 평균을 끌어올린다는 EXP-020 단서도 유지.

- 다음 실험:
  test 예산 해석(A/B)을 팀이 확정해 DECISIONS/EXPERIMENT_LOG 반영. 희귀 구간용 빈도 기반 보류 규칙 검토(valid에서 설계).

- 산출물 경로:
  `reports/exp022/exp022_results.md` / `.json` / `exp022_test_coverage_accuracy.csv` /
  `exp022_test_sample_confidence.csv` / `exp022_coverage_accuracy_valid_vs_test.png`(1920×1080) / `.pdf`(Noto Sans KR 임베드),
  `scripts/analyze_selective_prediction_test.py`

- 보존 상태:
  미커밋(로컬, 브랜치 `exp022/selective-test-fixed-threshold`, worktree `../구글 ai 캡스톤-exp022`). 입력 test 예측 CSV는 gitignore 로컬.

- 한계:
  test 예측은 EXP-012 GPU 추론, valid(EXP-015)는 CPU 추론 결과라 추론 환경이 다르다(score 6자리 반올림 영향은 미측정).
  valid·test 모두 artificial mask 기준이라 실제 결손 분포와 다를 수 있다. hapax·unseen 선택 표본이 작다(17건·9건).
  valid−test 비교는 독립 표본 z 검정 근사(동일 샘플 paired 아님).

### EXP-021 (2026-10-01) — ARI-8B zero-shot beam5 예비 평가 (n=200, 근사 Exact)

- 유형/상태: `EVAL` preliminary / `completed` (근사 Exact. 공식 EXP-019A 아님)
- 목적/가설: ARI-8B no-retrieval을 beam5로 돌려, 같은 valid 목록에서 예비 Exact@k를 본다.
- 데이터/split: valid 목록 **n=200**. test 사용 없음.
- 프로토콜: `generate()` beam search **num_beams=5**. Appendix C constrained decoding이 아니다. DECISIONS [015]의 "generate() 근사는 공정 비교가 아니다"를 뒤집지 않는다.
- 평가 지표 결과 (n=200, beam5 근사):
    - Exact@1: **0.105**
    - Exact@5: **0.225**
    - Exact@10: 후보가 beam5라 슬라이드에서는 **n.a. (beam5)**. 수치를 Exact@10으로 보고하지 않는다.
- 결과 요약: n=200 근사 Exact@1 0.105 / Exact@5 0.225. 발표 비교의 ZS 칸은 이 값이며, 정본 paired는 EXP-025다.
- 해석: 예비 실행, 단순화 Top-5, constrained decoding 미적용, n=200. 공식 ARI Top-K·SOTA가 아니다.
- 다음 실험: 같은 프로토콜의 QLoRA는 EXP-024. 동일 표본 McNemar는 EXP-025.
- 산출물 경로: EXP-025 same-200 비교에 사용. 이 갱신에서 확인한 Drive 경로는 따로 적지 않는다.
- 한계: beam5 근사. n=200. n≈500 수치(EXP-015 Exact@5 0.504 포함)를 EXP-024 n=200과 직접 비교하지 말 것(DECISIONS [017]).

### EXP-020 (2026-10-01) — 선택적 예측(coverage-accuracy) 분석 (EXP-015 valid 후처리)

- 유형/상태: `ANALYSIS` / `completed` (예측 CSV 후처리; 학습·추론·모델 로딩 없음)

- 목적/가설:
  무조건부 Exact@5 90%는 UNK 천장·희귀자 격차(EXP-013) 때문에 어렵다.
  "확신 있는 결손만 후보를 제시하고 나머지는 판독 보류로 전문가에게 넘기는" 선택적 예측 프레이밍에서
  Exact@5 ≥ 90%를 만족하는 최대 커버리지를 찾는다. 발표(2026-10-07) 주장 가능 여부 확인용.

- 데이터 버전:
  `outputs/predictions/exp015_sillok_ft_valid_predictions.csv` (valid 10,679행),
  빈도 구간용 `data/model_inputs/artificial_mask_dataset.csv` train label 빈도

- 모델:
  EXP-015 예측 재사용 (`exp012_sillokbert_gpu_best_valid_loss`). 재추론 없음.

- 코드/환경:
  `python scripts/analyze_selective_prediction.py`, CPU, 구간 정의는 `src/eval/stratified_analysis.py` 재사용

- confidence 지표:
  score는 전체 vocab softmax **확률**(logit 아님; 단일 한자 토큰만 남긴 top-10, 6자리 반올림).
  (a) `top1_score` (b) `margin` = top-1 − top-2 (c) `entropy` = top-10 확률을 합 1로 재정규화한 엔트로피(top-10 근사).
  참고로 softmax(score) 그대로의 엔트로피(`entropy_softmax_literal`)도 병기.

- test 사용:
  **없음** (valid only)

- 평가 지표 결과 (valid n=10,679, 재계산 전체 33.78 / 53.24 / 60.90 = EXP-015와 일치):

  | 지표 | Exact@5≥90% | Exact@5≥80% | Exact@5≥70% | Exact@1≥90% | Exact@1≥80% |
  |---|--:|--:|--:|--:|--:|
  | top1_score | **25.1%** (n=2,683) | 43.1% | 64.3% | 16.2% | 25.4% |
  | margin | 23.6% (n=2,524) | 38.3% | 57.1% | 15.8% | 25.4% |
  | entropy | 24.5% (n=2,613) | 42.1% | 62.2% | 16.2% | 25.7% |

  top1_score 커버리지 10/20/30%의 Exact@5: 96.7% / 92.6% / 86.7%. 곡선 평균 Exact@5: top1 77.1% > entropy 76.5% > margin 75.4%.

- 빈도 구간 교차 (top1_score 상위 25.1%, n=2,683):
  선택 비중 very_common 47.3%(전체 34.1%) · common 38.9%(38.9%) · mid 9.8%(16.2%) · rare 2.0%(6.4%) ·
  hapax 1.0%(2.0%) · unseen 0.9%(2.4%). 구간 내 선택률 very_common 34.9% → rare 7.9% · unseen 9.8%.
  선택 구간 Exact@5: very_common 95.1% · common 90.7% · mid 81.4% · rare 51.9% · hapax 50.0% · unseen 20.0%.

- 결과 요약:
  top-1 확률 상위 **25.1%(2,683건)**에서 Exact@5 90.0% (95% Wilson 88.8–91.1%). 80%는 43.1%, 70%는 64.3% 커버리지.
  세 지표 차이는 작고 top1_score가 근소 우위.

- 해석:
  확신 상위 구간은 고빈도자 쪽으로 기울지만(very_common 비중 +13pp), 고빈도자만 고르는 것은 아니다.
  다만 rare·hapax·unseen은 상위 구간에 들어와도 Exact@5가 20~52%로 90%에 못 미치므로,
  "커버리지 25%에서 Exact@5 90%"는 고빈도·중빈도자가 끌어올린 평균임을 발표에서 명시해야 한다.

- 다음 실험:
  임계값(top1 ≥ 0.6174)을 별도 split에서 재검증하지 않았으므로, 필요 시 valid 내 교차검증 또는
  test 예산 사용 여부를 팀이 결정. 희귀자 구간용 별도 보류 규칙(빈도 기반 abstain) 검토.

- 산출물 경로:
  `reports/exp020/exp020_results.md` / `.json` / `exp020_coverage_accuracy.csv` /
  `exp020_sample_confidence.csv` / `exp020_coverage_accuracy_top1_score.png`,
  `scripts/analyze_selective_prediction.py`

- 보존 상태:
  미커밋(로컬). 입력 예측 CSV는 gitignore 로컬.

- 한계:
  entropy·margin은 저장된 top-10만으로 계산한 근사. 임계값이 같은 valid에서 정해져 낙관 편향 가능(단일 split,
  재검증 없음). artificial mask 기준이라 실제 결손 분포와 다를 수 있음. 희귀 구간 표본이 작다(선택 hapax 28건).

### EXP-019 (2026-09-17) — ARI-8B 통합 및 공정 Top-K 평가 설계

- 유형/상태: `AUDIT`+`EVAL` design / **design completed; implementation and execution pending**

- 목적/가설:
  ARI-8B를 기존 금석문 artificial single-mask validation과 동일한 Exact@k 프로토콜로 평가할 수 있는지,
  그리고 retrieval 유무의 효과를 leakage를 통제한 조건에서 검증할 수 있는지 사전 감사한다.
  첫 연구 질문은 "BM25 기반 retrieval을 추가하는 것이 ARI-8B의 금석문 복원 성능을 개선하는가?"이다.
  넓은 의미의 모든 RAG 또는 최적 retriever 우위를 묻는 실험으로 해석하지 않는다.

- 데이터 버전:
  기존 `data/model_inputs/artificial_mask_dataset.csv` validation **10,679**행을 유지한다.
  평가 단위는 금석문 전체가 아닌 artificial single-mask sample 1건이며, gold는 `label_char` 한 글자다.
  기존 split은 inscription 단위, `seed=42`이며 train / valid / test의 `inscription_id` 교집합은 0이다.

- 기존 BERT 평가 구조 확인:
  `XLS` → `gsm_preprocess_v4` → inscription split (`seed=42`) → artificial single-mask →
  ±64 `[MASK]` window → SillokBERT MLM → 한 글자 Top-K → `src/eval/metrics.py` 순서다.
  예측 `['字1', '字2', '字3', '字4', '字5']`, gold `'字3'`이면 Exact@1=0, Exact@5=1이다.
  따라서 ARI도 **순서가 있는 서로 다른 한자 Top-5**를 만들어야 EXP-015와 공정 비교가 가능하다.

- 모델 및 최소 통합 경로:
  `DAMI-Lab/ARI-8B`를 별도 ARI runner로 연결한다.
  `기존 validation sample` → `[D1]` marker 변환 → ARI inference → output parsing →
  기존 prediction CSV 스키마 변환 → 기존 `src/eval/metrics.py`의 흐름을 사용한다.
  기존 BERT evaluator를 수정하지 않는 것이 최소 통합 원칙이다.

- 코드/환경 (감사·설계만):
  Cursor가 저장소·기존 EXP 결과·평가 코드를 분석해
  `docs/ARI_INTEGRATION_PLAN.md`, `docs/ARI_TOPK_DESIGN.md`를 작성했다.
  이 단계에서는 기존 코드·EXP 결과·git을 변경하지 않았고, ARI 모델 다운로드·모델 로딩·GPU 실행도 하지 않았다.

- Top-K 설계:
  ARI의 후보 점수는 BERT mask softmax와 다르다. 한자 `c`가 tokenizer에서 `t1 ... tn`으로 분해되면,
  후보 점수는 constituent token conditional logprob의 **산술평균**이다.
  raw score 척도는 BERT와 직접 비교하지 않으며, 이 점수로 정렬된 후보 순서만 기존 Exact@1/5/10에 사용한다.
  `torch.topk(next_token_logits, 5)`는 byte/token fragment를 포함하고 여러 token으로 구성된 한자를 빠뜨릴 수 있으므로 본평가에 쓰지 않는다.
  Appendix C 기반 constrained decoding은 first token 분기 → 미완 token 확장 → 한자 완성 확인 →
  유효 한자 필터링·중복 제거 → 평균 logprob 정렬 → Top-5의 순서로 설계했다.

- JSON prefix 및 재현성 표기:
  공개 target 형식은 예를 들어 `{'[D1]': '某'}`이다.
  본평가에서는 `{'[D1]': '`까지 teacher-force한 뒤 후보 한자를 탐색하도록 설계했다.
  이 prefix 처리와 valid character 판정, beam 폭, 최대 token 길이, 중복 처리, tie-break는 논문에 직접 명시되지 않은 **OURS**다.
  문서 태그는 `PAPER`(논문 직접 명시), `PUBLIC`(공개 코드 존재), `OURS`(우리 구현 결정), `UNKNOWN`(미확정)을 사용한다.
  발표에서는 "Appendix C를 그대로 재현했다"가 아니라, "Appendix C의 핵심 알고리즘을 기반으로
  공개되지 않은 세부 규칙을 명시적으로 고정해 재구현했다"라고 표현한다.

- 하이퍼파라미터 (설계상 임시값, 미실행):
  `B1=32` (first-token branching), `B2=8` (subsequent-token branching), `Lmax=4`.
  모두 논문 값이 아닌 **OURS**이며, B1=16/32/64 등의 sensitivity를 확인해 후보와 Exact@5 수렴 여부를 검증해야 한다.

- split / leakage 감사:
  서로 다른 `inscription_id` 사이에도 cross-split exact duplicate 원문이 **26건** 발견됐다.
  대부분 짧은 명문이지만 train/valid 사이에 약 405자 장문 중복도 있다.
  따라서 retrieval에서는 train-only index만으로 충분하다고 단정할 수 없다.
  EXP-019B/C 전에 exact duplicate, near-duplicate, retrieval corpus leakage audit이 필수다.

- 실험 구조 및 의사결정:
  - **EXP-019A:** ARI-8B, no retrieval. ARI 언어모델 자체와 EXP-015의 비교.
  - **EXP-019B:** ARI-8B + BM25. 동일 corpus·query·k·제외 규칙·중복 필터에서 retrieval 효과를 검증.
  - **EXP-019C:** ARI-8B + ChromaDB/dense retrieval. ChromaDB 담당자 이탈에 따라 **`deferred / optional`**.
  ARI 원 논문이 BM25 retrieval을 사용했으므로, 우리 contribution은 "ARI+BM25라는 새 방법"이 아니라
  금석문 도메인 적용, SillokBERT single-mask baseline과의 동일 Exact@k 비교, no-retrieval vs retrieval ablation,
  leakage 통제, 실제 복원 서비스와의 연결에 둔다.
  서비스의 ChromaDB 근거문헌 retrieval과 모델 실험의 BM25 retrieval은 별도로 설명한다.

- 모델 선택 기준 (예정):
  EXP-015와 동일 validation 10,679건에서 Exact@1/5/10을 비교한다.
  EXP-019A 전체 평가는 Top-K decoder 검증과 평가 규칙 freeze 뒤에만 진행한다.
  EXP-019B는 EXP-019A 평가 규칙이 freeze되고 BM25 leakage audit을 통과한 뒤 진행한다.

- test 사용: **없음**. 현 단계는 validation 설계·감사이며 full test 예산을 사용하지 않는다.

- 평가 지표 결과:
  - Exact@1: *(미실행 — 수치 없음)*
  - Exact@5: *(미실행 — 수치 없음)*
  - Exact@10: *(미실행 — 수치 없음)*
  - 기타: *(모델 로딩·GPU inference·BM25 index·ChromaDB 실험 모두 미실행)*

- 결과 요약:
  ARI-8B는 기존 validation 및 evaluator를 유지한 채 별도 runner로 통합할 수 있다는 설계 결론을 얻었다.
  그러나 이는 **통합 가능성 및 평가 설계의 확인**일 뿐 성능 향상이나 GPU 실행 성공의 증거가 아니다.

- 해석:
  smoke는 `generate()` 기반 Top-1으로 pipeline 동작만 확인할 수 있다. `generate()` 한 번의 결과를 Exact@5로 보고해서는 안 된다.
  공식 비교는 constrained Top-K 후보를 기존 `rank_of_gold()`와 `compute_exact_metrics()`에 넣어 수행한다.
  constrained Top-1과 일반 `generate()` parsing Top-1의 일치율도 sanity check로 측정한다.

- 다음 실험:
  1. `ARI_TOPK_DESIGN.md` 기준 decoder 구현
  2. GPU 없이 가능한 unit tests
  3. 소량 sample model-load / `generate()` smoke
  4. constrained Top-1 vs `generate()` Top-1 sanity check
  5. B1/B2/Lmax sensitivity → Top-K 규칙 freeze → EXP-019A full valid
  6. BM25 retrieval corpus leakage audit → EXP-019B

- 산출물 경로:
  `docs/ARI_INTEGRATION_PLAN.md`, `docs/ARI_TOPK_DESIGN.md`.
  향후 ARI runner·prediction CSV·run metadata·decoder unit test는 EXP-015/018과 분리된 EXP-019 경로에 보존한다.

- 보존 상태:
  설계 문서 작성 완료. 구현 코드·모델 가중치·GPU 로그·prediction CSV·공식 ARI Exact@k 결과는 아직 없다.

- 한계:
  Top-K 공개 구현 부재로 constrained decoder의 세부 규칙은 일부 OURS다.
  beam 폭 민감도, tokenizer byte 경로, JSON parsing, invalid Hanja output, 4-bit T4 16GB OOM 가능성은 미검증이다.
  cross-split duplicate는 retrieval contamination 위험이며, EXP-019C 부재 상태에서는 dense retrieval과 BM25의 우열 또는 최적 retriever를 주장할 수 없다.

### EXP-018 (Phase A 2026-08-02 · GPU Phase B 018A/018B 실행 확인 2026-08-30 · post-hoc valid Exact@k 평가 완료 2026-09-17) — SillokBERT 학습률 재탐색 (018A/018B Exact@k 평가 완료, 둘 다 gate 미통과; 018C 미실행)

- 유형/상태: `TRAIN` 018A/018B **GPU 학습 완료**(loss) · `EVAL`(Exact@k) **완료(2026-09-17), 둘 다 gate 미통과** · 018C **미실행/결과 미확보**
  (Phase A 코드·테스트·CPU smoke·Colab kit 기록은 아래에 원본대로 보존)
- 목적/가설:
  EXP-017 이후 공식 후보는 FT 유지. EXP-018은 SillokBERT full-data FT의 **학습률 재탐색**
  (5e-5 / 3e-5 / 2e-5 × 4 epoch)으로 valid Exact@5를 EXP-015 대비 개선하는지 확인한다.
  Phase A 기록은 **코드·테스트·CPU smoke·Colab kit** 준비 단계만 해당한다.
  2026-08-30 업데이트는 **GPU Phase B 실제 실행 결과**(018A/018B 학습 완료, epoch별 loss)를 반영했다.
  **2026-09-17 업데이트는 018A/018B의 post-hoc valid Exact@k 공식 평가 결과**를 반영한다:
  DECISIONS [012]에서 정한 채택 절차(`select_checkpoint_by_exact.py` → `run_ft_eval_chain.py`)를
  실행해 018A/018B 모두 gate를 통과하지 못했음을 확인했다. 018C(lr=2e-5)는 여전히 실행 결과가
  확인되지 않아 미실행 또는 결과 미확보 상태로 남는다.
- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` · inscription-level split · seed=42
  (train 46,139 / valid 10,679). **test 미사용.**
- 모델: `ddokbaro/SillokBert` · single-mask · context ±64 · max_seq_len 512
- 코드/환경 (Phase A, 원본 유지):
  - 브랜치: `cursor/exp018-phase-a-prep-41a8`
  - `src/data/finetune_sillokbert.py` — git commit/seed/data/HP → `run_meta.json`+`finetune_metrics.json`;
    path guards (refuse `notebooks/best`, EXP-012/011 paths); default out `outputs/finetune/exp018/...`
  - `src/eval/select_checkpoint_by_exact.py` — post-hoc Exact@5 → `best_exact5/`
  - `src/eval/run_ft_eval_chain.py` — subsample 2000 + McNemar + stratified + gate
  - Colab: `notebooks/exp018_colab.ipynb`, `reports/colab_exp018_runbook.md`
  - CPU smoke only (`outputs/finetune/exp018/_smoke_*`); **smoke ≠ official**
  - Phase A verification: unit tests PASS; CPU smoke pipeline PASS; tiny Exact@5-select smoke PASS.
    (Phase A 시점 기준 기록. 이후 GPU Phase B·post-hoc Exact@k 평가 결과는 아래 참조)
- 코드/환경 (2026-09-17 post-hoc Exact@k 평가):
  - repo commit: `c6df2c2d4c6e215c850df66e1fa0d36c9200895e`
  - GPU: **NVIDIA L4**
  - `select_checkpoint_by_exact.py` / `run_ft_eval_chain.py`를 018A/018B 각각의
    `checkpoint-epoch-1`~`checkpoint-epoch-4` 전부에 대해 실행했다.
  - checkpoint 선택 규칙: **valid Exact@5 최대 → 동점이면 valid_loss 최소 → 그래도 동점이면 낮은 epoch**
    (DECISIONS [012]과 동일 원칙).
  - 평가 데이터: valid subsample **n=2,000**, `seed=42`. **test 사용 없음.**
  - 정본 산출물(Google Drive): `/content/drive/MyDrive/exp018/eval_results/20260917T090850Z/`
    - `exp018_eval_summary.md`
    - `checkpoint_selection.csv`
    - `epoch_valid_exact_metrics.csv`
    - `paired_vs_exp015.csv`
    - `gate_results.csv`
    - `drive_loss_vs_docs.csv`
- 하이퍼파라미터 및 GPU Phase B 실행 결과:
  - **018A** — planned_experiment_id `EXP-018A` · lr=5e-5, batch=16, epochs=4, seed=42,
    context_window=64, max_seq_len=512, single_mask=true, test 미사용. GPU full-data 학습 **완료**.

    | epoch | train loss | valid loss |
    |---|---|---|
    | 1 | 4.065657206680426 | 3.7285900998591646 |
    | 2 | 2.419724188071553 | 3.8853376706441245 |
    | 3 | 1.1177933866973448 | 4.253182492864114 |
    | 4 | 0.4733658146882078 | 4.387591689596161 |

    valid loss 기준 best epoch: **epoch 1**.

    > **2026-09-17 추가 메모 (수치 정정 보류):** `drive_loss_vs_docs.csv` 대조 결과, Google Drive의
    > 실제 `best_checkpoint.json` loss 값과 위 표의 일부 값이 다른 것으로 확인됐다. 실제 Drive
    > 산출물 값이 정본이 되어야 하나, 본 갱신에서는 `drive_loss_vs_docs.csv`의 구체 수치를
    > 확보하지 못해 표를 직접 정정하지 못했다. **위 표는 잠정 유지하며, 실제 값 확인 후 정정
    > 사유와 함께 갱신 필요(TODO — Cursor/팀 확인 필요).** 과거 기록을 조용히 덮어쓰지 않는다.
  - **018B** — planned_experiment_id `EXP-018B` · lr=3e-5, batch=16, epochs=4, seed=42,
    context_window=64, max_seq_len=512, single_mask=true, test 미사용. GPU full-data 학습 **완료**.

    | epoch | train loss | valid loss |
    |---|---|---|
    | 1 | 3.972521441093119 | 3.6232217716914352 |
    | 2 | 2.5826241617525176 | 3.757688502287535 |
    | 3 | 1.558894082066003 | 4.029885791780029 |
    | 4 | 0.9665334310528434 | 4.141098001036227 |

    valid loss 기준 best epoch: **epoch 1**. (아래 post-hoc Exact@5 기준 best epoch은 **epoch 2** —
    두 기준이 서로 다르다. 상세는 아래 "2026-09-17 post-hoc valid Exact@k 평가 결과" 참조.)

    > **2026-09-17 추가 메모 (수치 정정 보류):** 018A와 동일하게, `drive_loss_vs_docs.csv` 대조 결과
    > 일부 loss 값이 Drive 실제 산출물과 다른 것으로 확인됐다. 구체 수치를 확보하지 못해 표는
    > 잠정 유지하며 정정은 TODO로 남긴다.
  - **018C** — planned_experiment_id `EXP-018C` · lr=2e-5 예정. **실행 결과 미확인 — 미실행 또는
    결과 미확보 상태.** 완료로 표시하지 않는다. (2026-09-17 시점에도 변동 없음 — 후순위.)
  - Drive: `/content/drive/MyDrive/exp018/{018a,018b,018c}/`
    (2026-09-17 확인: `018a/`, `018b/` 각각에 `checkpoint-epoch-1`~`checkpoint-epoch-4`가 실제로
    존재하며, 각 checkpoint에 `model.safetensors`와 tokenizer 파일이 확인됐다. `018c/`는 미확인 —
    018C가 미실행이므로 산출물이 없는 것으로 추정된다.)
- 모델 선택 기준 (2026-09-17 실제 적용 결과):
  학습 중 `best/` = minimize_valid_loss (유지, 위 표 참조).
  공식 채택 기준 = post-hoc valid Exact@5 max → tie min valid_loss → lower epoch (`best_exact5/`).
  Gate: subsample Exact@5 ≥ EXP-015 +0.7pp AND McNemar p<0.05 → full valid;
  full valid +1.0pp → adoption candidate. EXP-015 공식 full-valid baseline Exact@5 = 53.24%
  (n=10,679). **이번 subsample paired 비교의 EXP-015 Exact@5는 53.40%**(동일 2,000개 subsample
  기준, `paired_vs_exp015.csv`) — 아래에서 이 두 수치를 명확히 구분해 기록한다.
- test 사용: **없음** (잔여 test 예산 미사용, 018A/018B 평가 전 구간)
- 평가 지표 결과 — 2026-09-17 post-hoc valid Exact@k 공식 평가 (valid subsample n=2,000, seed=42, checkpoint-epoch-1~4 전체 평가, `epoch_valid_exact_metrics.csv`/`checkpoint_selection.csv` 근거):

  **018A (post-hoc Exact@5 기준 best epoch: epoch 1)**
  - Exact@1: **34.05%**
  - Exact@5: **52.20%**
  - Exact@10: **59.70%**

  **018B (post-hoc Exact@5 기준 best epoch: epoch 2 — valid_loss 기준 best epoch 1과 다름)**
  - Exact@1: **35.65%**
  - Exact@5: **54.10%**
  - Exact@10: **61.55%**

  **두 종류의 EXP-015 기준선을 구분해 기록한다 (직접 빼서 비교 금지):**
  - 공식 full-valid incumbent: **EXP-015 Exact@5 = 53.24%** (n=10,679, DECISIONS [012]/[014])
  - 이번 subsample paired 비교의 EXP-015: **Exact@5 = 53.40%** (동일 2,000개 subsample, `paired_vs_exp015.csv`)

  **paired 비교 (동일 2,000개 subsample 기준, `paired_vs_exp015.csv`):**

  | 후보 | Exact@5 | vs EXP-015(subsample 53.40%) | McNemar p | gate 통과 |
  |---|---:|---:|---:|:---:|
  | 018A (epoch 1) | 52.20% | **−1.20pp** | 0.093455 | 미통과 |
  | 018B (epoch 2) | 54.10% | **+0.70pp** | 0.350642 | 미통과 |

  - full valid(n=10,679) 승격: **018A/018B 모두 승격하지 않음.**

  > **수치 불일치 메모 (2026-09-17):** 자동 생성된 `gate_results.csv`의 `baseline_exact@5`는
  > 공식 full-valid EXP-015 값(0.5324, 53.24%)을 사용해 018B의 delta를 **+0.86pp**로 계산했다.
  > 그러나 이는 **표본 크기가 다른(n=10,679 vs n=2,000) 두 수치를 직접 뺀 값**이며,
  > `paired_vs_exp015.csv`가 보여주는 **동일 2,000개 subsample 기준** 차이는 **+0.70pp**
  > (EXP-015 53.40% vs 018B 54.10%)다. 본 로그에서는 **+0.86pp를 paired 개선량으로 기록하지
  > 않으며**, 동일 표본 기준인 **+0.70pp**만 공식 paired 비교 결과로 사용한다.
- 결과 요약:
  Phase A 준비 완료. GPU Phase B: 018A·018B는 학습(4 epoch) 완료, loss 확보.
  **2026-09-17: 018A/018B post-hoc valid Exact@k 평가 완료.** 018A(epoch 1)는 EXP-015 대비
  Exact@5 −1.20pp로 gate 미통과. 018B(epoch 2)는 Exact@5 +0.70pp로 수치상 소폭 개선됐으나
  McNemar p=0.350642로 통계적으로 유의하지 않아 gate 미통과. **018A/018B 어느 쪽도 공식
  incumbent로 승격하지 않으며, EXP-015를 계속 공식 valid baseline으로 유지한다.** 018C는
  여전히 실행 결과가 확인되지 않았다.
- 해석:
  018A·018B 모두 **train loss는 epoch가 진행될수록 계속 감소**하지만 **valid loss는 epoch 1
  이후 악화**된다. 현재 조건(해당 lr들, 4 epoch, single-mask)에서는 **빠른 과적합이 반복적으로
  관찰된다**는 정도까지만 기록한다.
  018B epoch 1의 valid loss(**3.6232217716914352**)는 018A epoch 1(3.7285900998591646)보다 낮고
  EXP-012 기존 best valid loss보다도 낮다. 그러나 **2026-09-17 평가 결과, 018B의 valid_loss 기준
  best epoch(epoch 1)와 valid Exact@5 기준 best epoch(epoch 2)가 서로 다르다는 것이 실제로
  확인됐다.** 이는 valid_loss 개선이 Exact@5 개선과 반드시 일치하지 않을 수 있다는, DECISIONS
  [012]가 이미 우려했던 한계(EXP-012도 loss 기준으로 선택된 모델이 Exact@5 최적 epoch가 아닐 수
  있다는 한계)를 실제 데이터로 뒷받침하는 사례다. 즉 DECISIONS [012]에서 checkpoint를 valid_loss가
  아닌 **post-hoc valid Exact@5 기준으로 채택하기로 한 결정**이 없었다면, 018B는 (Exact@5 기준으로
  더 나은) epoch 2 대신 (valid_loss 기준) epoch 1이 채택되어 더 낮은 성능의 체크포인트가 공식
  후보로 잘못 선택될 뻔했다.
  다만 이 사실이 018B의 "채택"을 의미하지는 않는다 — epoch 2 기준으로도 Exact@5 +0.70pp는
  McNemar p=0.350642로 통계적으로 유의하지 않으므로 gate를 통과하지 못했다.
- 다음 실험:
  1) ~~018A/018B best-loss 체크포인트(epoch 1)에 대해 `select_checkpoint_by_exact.py`로 post-hoc
     valid Exact@5를 공식 평가한다~~ — **2026-09-17 완료.** 018A/018B 모두 gate 미통과로 확인됨.
  2) 018C(lr=2e-5) 및 추가 SillokBERT HP sweep은 **후순위**로 둔다.
  3) **다음 주요·최우선 작업은 EXP-019(ARI-8B) Top-K constrained decoder 구현이다.**
     EXP-015는 그 비교의 공식 incumbent validation baseline으로 계속 유지한다(DECISIONS [014]).
  4) `drive_loss_vs_docs.csv` 기준 018A/018B loss 표 정정을 위해 실제 수치를 확인하고
     본 로그를 갱신한다(TODO — 위 표의 "수치 정정 보류" 메모 참조).
- 산출물 경로:
  `src/data/finetune_sillokbert.py`, `src/eval/select_checkpoint_by_exact.py`,
  `src/eval/run_ft_eval_chain.py`, `notebooks/exp018_colab.ipynb`,
  `reports/colab_exp018_runbook.md`, `tests/test_select_checkpoint_by_exact.py`,
  `tests/test_finetune_run_mode.py` (path guards)
  GPU Phase B(018A/018B) 학습 로그·체크포인트 실제 저장 경로: **확인 완료(2026-09-17)** —
  `/content/drive/MyDrive/exp018/018a/checkpoint-epoch-1~4`,
  `/content/drive/MyDrive/exp018/018b/checkpoint-epoch-1~4`
  (각 checkpoint에 `model.safetensors`·tokenizer 확인됨).
  2026-09-17 post-hoc Exact@k 평가 정본 산출물:
  `/content/drive/MyDrive/exp018/eval_results/20260917T090850Z/`
  (`exp018_eval_summary.md`, `checkpoint_selection.csv`, `epoch_valid_exact_metrics.csv`,
  `paired_vs_exp015.csv`, `gate_results.csv`, `drive_loss_vs_docs.csv`)
- 보존 상태: Phase A 코드·문서 커밋. GPU Phase B 018A/018B 가중치 저장 위치는 위에서 확인 완료
  (Google Drive). 018C 가중치 없음(미실행). post-hoc 평가 결과 CSV/MD는 Drive에만 있으며
  로컬·Git 백업 여부는 별도 확인 필요.
- 한계:
  EXP-015 prediction CSV는 gitignore — McNemar는 Drive/로컬 사본 필요
  (`reports/evaluation/exp015_sillokbert_valid_summary.csv`는 커밋됨).
  CPU smoke 지표는 공식 비교에 쓰지 않음.
  **"학습 완료"와 "Exact@k 평가 완료"는 다르다** — 018A/018B는 2026-09-17부로 양쪽 다 완료됐으나,
  **gate를 통과하지 못했으므로 공식 후보가 아니다.** full valid(n=10,679) 수치는 존재하지 않는다
  (subsample n=2,000 결과만 있다) — full valid 성능이 있는 것처럼 서술하지 않는다.
  `gate_results.csv`의 baseline 수치(full-valid 53.24%)와 `paired_vs_exp015.csv`의 subsample
  수치(53.40%)는 표본이 다르므로 직접 빼서 개선량으로 쓰지 않는다(위 "수치 불일치 메모" 참조).
  018A/018B 학습 loss 표 일부는 `drive_loss_vs_docs.csv` 대조 결과 Drive 실제 값과 다를 수 있음이
  확인됐으나, 본 갱신에서는 구체 수치를 확보하지 못해 정정을 완료하지 못했다(TODO).

### EXP-017 (2026-08-02) — ZS/FT 확률 보간 (재학습 없음)

- 유형/상태: `EVAL` / `completed` (학습 없음; DECISIONS [012]; CONTEXT_WINDOW=±64)

- 목적/가설:
  EXP-013의 "FT=고빈도 강·ZS=hapax/rare 상대 우위"를
  `p_final=α·p_FT+(1−α)·p_ZS` 선형 보간으로 활용해 overall·구간 균형을 개선할 수 있는지 확인한다.

- 데이터 버전:
  `artificial_mask_dataset.csv` valid. 1단계 subsample n=2,000 seed=42
  (`_select_rows`, EXP-016과 동일). 2단계 full valid n=10,679.
  **test 미사용.**

- 모델:
  ZS=`ddokbaro/SillokBert`; FT=`notebooks/best/` (EXP-012 best). 가중치 변경 없음.
  후보: 각 모델 top-50 단일문자 합집합 + 양쪽 softmax 확률 조회.

- 코드/환경:
  `scripts/exp017_zs_ft_interpolate.py` (기존 src 평가/학습 코드 미수정)
  device: **cpu**. paired: `src.eval.paired_exact_compare` (McNemar continuity).

- 하이퍼파라미터:
  α ∈ {0.0,…,1.0} step 0.1. top_k=50. context_window=±64.

- 모델 선택 기준:
  subsample overall Exact@5 최대 α를 1차 후보. hapax 고원 끝 α를 트레이드오프 후보.
  마일스톤 확정: **α\*=0.8**.

- test 사용: **없음**

- Sanity:
  α=1.0 subsample Exact@1/5/10 = EXP-015 동일 sample **34.30 / 53.40 / 60.70** (Δ=0).
  α=1.0 full Exact@5 = **53.235%** = EXP-015 공식값.

- 평가 지표 결과 (subsample n=2,000):
  - α=0.8: Exact@5 **53.45%** (overall max); hapax@5 16.13% (n=31)
  - α=0.5: Exact@5 52.60%; hapax@5 **22.58%** (hapax 고원)
  - α=1.0: Exact@5 53.40%
  - 전체 α표·구간별 Exact@5: `reports/exp017/exp017_alpha_sweep.csv`

- paired vs EXP-015 (subsample):
  - α=0.8 overall Exact@5: Δ +0.05pp, **p=1.00** (비유의)
  - α=0.5 overall Exact@5: Δ −0.80pp, p=0.205; Exact@1 Δ −2.0pp, **p=0.002** (악화)
  - α=0.5 rare Exact@5: Δ +6.0pp, p=0.008 (n=150); hapax Δ +6.45pp, p=0.48 (n=31)

- full valid 마일스톤 (α\*=0.8, n=10,679):
  - Exact@1/5/10: **33.78% / 53.21% / 61.21%**
  - vs EXP-015 Exact@5: Δ **−0.028pp**, p=0.917 (비유의·사실상 동등 이하)
  - 참고 α=1.0: 33.78 / **53.24** / 60.90; α=0.5: 32.03 / 52.52 / 60.93

- 결과 요약:
  선형 보간은 FT overall을 유의하게 이기지 못한다. subsample α=0.8 미세 이득은 full에서 소멸.

- 해석:
  고빈도 질량이 overall을 지배해 ZS 혼합의 hapax/rare 이득이 overall Exact@5로 잘 전달되지 않는다.
  α→1에 가까울수록 FT와 동일해지고, α를 낮추면 hapax/rare는 일부 회복되나 overall·@1이 깎인다.

- 다음 실험:
  공식 후보는 **EXP-015(FT) 유지**. EXP-018 HP 재탐색 진행.
  구간 특화는 별도 라우팅 실험으로 분리. test 예산은 보간에 사용하지 않음.

- 산출물 경로:
  `reports/exp017/exp017_results.md`, `exp017_alpha_sweep.json|.csv`,
  `paired_alpha0.8_vs_exp015.json`, `paired_alpha0.5_vs_exp015.json`,
  `paired_alpha0.8_full_vs_exp015.json`, `exp017_milestone_alpha*.json`,
  `outputs/predictions/exp017_dist_*.jsonl`, `exp017_alpha*_*.csv`,
  `scripts/exp017_zs_ft_interpolate.py`

- 보존 상태: reports는 커밋 후보. 대용량 jsonl/예측은 outputs(로컬).

- 한계:
  CPU 추론. hapax subsample n=31. top-50 합집합 근사(전 vocab 보간 아님).

### EXP-016 (2026-08-02) — 학습·평가 입력 형식 정합 진단 + inference context window 스윕

- 유형/상태: `AUDIT`+`EVAL` / `completed` (재학습 없음; DECISIONS [012] iteration)

- 목적/가설:
  EXP-012의 `max_seq_len=512`와 `context window ±64` 병기, 그리고 표준 MLM collator(15% 다중 마스크)
  vs 평가(단일 마스크) 불일치 가능성을 코드로 판정한다. 이어서 재학습 없이 추론 context window만
  ±64→±128/±256으로 바꿔 valid subsample Exact@k가 개선되는지 확인한다.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` valid subsample **n=2,000**, `seed=42`
  (test 미사용)

- 모델:
  `notebooks/best/` — EXP-012 Colab GPU best (`minimize_valid_loss`, epoch 1), 가중치 변경 없음

- 코드/환경:
  - 학습 코드 조사: `src/data/finetune_sillokbert.py` (Colab은 `notebooks/exp012_colab.ipynb`→동일 CLI)
  - 평가: `src/eval/mask_window.py` / `run_fill_mask_eval.py` / `predictors.py`
  - 스윕 드라이버(기존 src 미수정): `outputs/exp016/run_context_window_sweep.py`
  - paired: `src/eval.paired_exact_compare` (McNemar continuity correction)
  - device: **cpu** (로컬 CUDA 미가용)
  - Git: `1e45776` 근처 작업 트리

- 하이퍼파라미터:
  해당 없음 (추론 조건만). 비교 창: **±64 / ±128 / ±256**. 후보 정규화·top-k는 기존과 동일.

- 모델 선택 기준:
  해당 없음 (체크포인트 고정)

- test 사용:
  **없음**

- 코드 조사 판정:
  **(a) 정합.** `DataCollatorForLanguageModeling` 미사용. 학습은 샘플당 단일 `[MASK]`+label `-100`,
  collate는 패딩만. 학습·평가 모두 `CONTEXT_WINDOW=64`. `max_seq_len=512`는 학습 초과 스킵·eval
  truncation 상한. **(b) 다중 vs 단일 마스크 불일치 없음. (c) 기본 문맥 길이 불일치 없음.**

- 평가 지표 결과 (valid subsample n=2,000):
  - ±64: Exact@1 **34.30%** / @5 **53.40%** / @10 **60.70%** (EXP-015 예측 필터)
  - ±128: Exact@1 33.30% / @5 52.65% / @10 60.75%
  - ±256: Exact@1 27.10% / @5 44.55% / @10 50.90%

- paired vs ±64 (Exact@5 주력):
  - ±128: Δ −0.75pp, only_64=64 / only_128=49, p=0.188 (비유의)
  - ±256: Δ −8.85pp, only_64=251 / only_256=74, p=1.6e−22 (**유의 악화**)
  - Exact@1/10도 동일 방향 (±128 비유의·±256 유의 악화; @10 ±128은 ≈0)

- 결과 요약:
  학습·평가 형식은 정합. 추론 window 확대는 개선이 없고 ±256은 유의하게 악화.

- 해석:
  표준 MLM 다중 마스크 불일치 가설은 본 코드베이스에 해당하지 않는다. ±256은 877/2000이
  tokenizer 512 truncate에 들어가나 마스크 소실은 0 — 하락은 OOD 장문맥·비대칭 truncate로 보는 것이 타당.

- 다음 실험:
  - EXP-017 ZS/FT 보간: context window **±64 고정**
  - EXP-018 HP 재탐색: 단일-마스크 유지; 문맥 길이는 학습·평가 동시 변경만 후보

- 산출물 경로:
  - `reports/exp016/exp016_results.md`
  - `reports/exp016/exp016_context_window_results.json`
  - `reports/exp016/paired_cw{128,256}_vs_cw64_k{1,5,10}.json`
  - `outputs/predictions/exp016_cw{64,128,256}_valid_sub2000_predictions.csv`
  - `outputs/exp016/run_context_window_sweep.py`

- 보존 상태:
  reports JSON/MD는 커밋 후보. 예측 CSV·드라이버는 outputs(로컬).

- 한계:
  ±64는 EXP-015 재사용(드라이버 재추론 미실시). CPU 추론. Colab 세션 임시 패치는 로그상 확인 불가.

### EXP-015 (2026-08-01) — EXP-012 best full-valid Exact@k 기준선

- 유형/상태: `EVAL` / `completed` (학습 없음; RISK_CHECKLIST §5 · DECISIONS [012])

- 목적/가설:
  EXP-012는 validation에 `valid_loss`만 기록하고 Exact@k는 test만 있다.
  이후 FT 개선 iteration의 "좋아졌는지" 판단을 위해 **동일 best 체크포인트**의
  full valid Exact@k 기준선을 확보한다.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` valid **10,679**행
  (기존 inscription split, `seed=42`)

- 모델:
  `notebooks/best/` — EXP-012 Colab GPU best (`minimize_valid_loss`, epoch 1)
  `model_version=exp012_sillokbert_gpu_best_valid_loss`

- 코드/환경:
  `python -m src.eval.run_fill_mask_eval --run-mode full --split valid
    --model-kind ddokbaro/SillokBert --model-path notebooks/best --require-model-path
    --batch-size 16 --overwrite --committed-summary`
  device: **cpu** (로컬 CUDA 미가용)

- 모델 선택 기준:
  해당 없음 (기존 체크포인트 평가만). checkpoint를 다시 고르지 않음.

- test 사용:
  **없음** (valid only)

- 평가 지표 결과 (valid n=10,679):
  - Exact@1: **33.78%** (0.3377657083996629)
  - Exact@5: **53.24%** (0.532353216593314)
  - Exact@10: **60.90%** (0.608952149077629)

- 결과 요약:
  동일 가중치의 test(EXP-012: 33.75 / 53.04 / 60.47)와 근접.
  이후 일상 개선 비교의 valid 기준선으로 채택(DECISIONS [012]).

- 해석:
  valid≈test 수준이라 큰 분포 붕괴는 보이지 않는다.
  다만 EXP-012는 loss로 고른 모델이므로 Exact@5 최적 epoch와 다를 수 있음(한계).

- 다음 실험:
  FT 후보는 valid subsample n=2,000 + `paired_exact_compare`로 EXP-015와 비교.
  마일스톤에서만 full valid. full test는 잔여 예산 2회.

- 산출물 경로:
  `outputs/predictions/exp015_sillok_ft_valid_predictions.csv` (10,679행)
  `reports/evaluation/exp015_sillokbert_valid_summary.csv`
  `reports/exp015_results.md` / `reports/exp015_results.json`

- 보존 상태:
  summary·JSON·MD는 Git 추적. 예측 CSV는 gitignore 로컬.

- 한계:
  CPU 추론. epoch 2 체크포인트 소실으로 Exact@k 기준 재선택 불가.
  EXP-012 loss-selection 한계는 DECISIONS [012] 참조.

### EXP-014 (2026-07-31 기록 · 산출물 반영 2026-08-08; full 재집계 보류) — hapax 하락 원인 탐색 (가설)

- 유형/상태: `ANALYSIS` / `hypothesis` (표본 소) + EXP-012 full 재집계는 `deferred` (DECISIONS [011])
  산출물(`reports/exp014_*`, `scripts/analyze_hapax_decline.py`)은 본 브랜치에 커밋.
  EXP-012 full predictions 기반 hapax 232 전수 재집계는 여전히 보류.

- 목적/가설:
  FT 이후 hapax/unseen Exact@k 약화 여부를 UNK 교차·ZS↔FT 쌍비교로 가늠.
  EXP-012 full 예측 부재로 EXP-011 subsample(hapax 겹침 n=48) 중심.

- 평가 지표 결과 (가설 표본):
  - EXP-007 zs hapax (full n=232): Exact@1 3.4% / @5 8.2%
  - EXP-011 FT hapax (subsample n=48): Exact@1 0.0% / @5 2.1%
  - 쌍대 48건: ZS @1/@5 = 2.1%/4.2%, FT @1/@5 = 0.0%/2.1%
  - ZS@5 ∧ ¬FT@5: **1건** (`mask_00013462`, gold=鱗)
  - SillokBert UNK/multi on hapax labels: **43/232 (18.5%)** — FT 전후 동일

- 결과 요약:
  겹침 표본에서 양쪽 top-5 실패가 46/48로 지배적. 명확한 ZS→FT 하락은 1건뿐.
  UNK는 hapax 실패의 일부만 설명하며 FT가 UNK를 늘렸다는 증거는 없다.

- 해석 한도:
  가설 수준. 강한 결론 금지. 재개 시 EXP-012 `exp012_sillok_ft_full_predictions.csv`로
  동일 분석을 다시 돌린 뒤 채택 여부를 재검토한다.

- 산출물 경로:
  `reports/exp014_hapax_decline.md` / `.json` / `exp014_hapax_paired_cases.csv`
  `scripts/analyze_hapax_decline.py`

- 관련: DECISIONS [011], `docs/CURSOR_HANDOFF.md` §6

### 발표 데모 shortlist (2026-07-31) — EXP-012 real_defect 후보 7건

- 유형/상태: `ANALYSIS` / `candidate` (정량 평가 아님 · 팀 검토 대기)
- 비고: 원격 커밋에서 "EXP-013"으로 잘못 번호됐으나, 공식 EXP-013은 층화 분석이므로 EXP 번호 없이 기록.

- 목적/가설:
  EXP-012 실제 결손 예측에서 발표용 5~7건을 압축한다.
  정답이 없으므로 "문맥상 그럴듯한 후보" 수준만 서술한다.

- 데이터 버전:
  1차 풀: `reports/expert_panel/expert_review_panel.csv` real_defect 25건
  (원천 전수 `outputs/predictions/exp012_real_defect_predictions.csv` 3,704건은 git 밖)

- 모델: `exp012_sillokbert_gpu_best_valid_loss`

- 결과 요약:
  HTML/각주 노이즈 배제·점수 고/저 혼합·비문당 1건 기준으로 **7건** 선정.
  패널 시대 편향(고구려 중심)으로 시대 분산은 제한적.

- 해석:
  발표 §7을 EXP-011 `real_defect_demo_30.csv`에서 EXP-012 출처로 교체해야
  본문 수치(EXP-012)와 데모 모델이 일치한다.

- 다음 실험:
  팀 검수 후 5건 확정 · (선택) 전수 3,704에서 조선/고려 사례 추가.

- 산출물 경로:
  `reports/exp012_demo_cases.md` / `reports/exp012_demo_cases.csv`
  `reports/presentation_outline.md` §7

### EXP-013 (2026-08-01) — Phase 4 fine-tuning 층화 분석 (EXP-012 후처리)

- 유형/상태: `AUDIT` / `completed` (예측 CSV 후처리 분석; 새 학습·추론 없음)

- 목적/가설:
  EXP-008(Phase 3 zero-shot 층화)과 동일한 구간 정의로 EXP-012 fine-tuned 예측을 층화해,
  fine-tuning의 +9.9pp(Exact@5) 이득이 어느 빈도 구간에서 나왔는지 확인한다.
  가설: 희귀자 구간에서도 개선되면 도메인 적응, 고빈도에만 몰리면 학습 분포 과적합.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` test 9,719행 + 아래 예측 CSV 3종

- split 방식:
  test split (동일, seed=42)

- 모델:
  후처리 분석 only — `outputs/predictions/exp004b_bert_zs_full_predictions.csv` (EXP-006),
  `outputs/predictions/exp005b_sillok_zs_full_predictions.csv` (EXP-007),
  `notebooks/best/exp012_sillok_ft_full_predictions.csv` (EXP-012)

- 코드/환경:
  `python -m src.eval.stratified_analysis` (EXP-005/008과 동일 스크립트·구간 정의), CPU

- 모델 선택 기준:
  해당 없음 (기존 예측 재집계)

- test 사용:
  기존 test 예측 재집계. 신규 test 평가 **없음**.

- 평가 지표 결과 (Exact@5, train label 빈도 구간별):

  | 구간 | n | EXP-006 zs | EXP-007 zs | EXP-012 FT | FT−zs |
  |---|--:|--:|--:|--:|--:|
  | very_common (100+) | 3,364 | 54.6% | 58.0% | **71.7%** | +13.7pp |
  | common (21-100) | 3,690 | 32.7% | 44.9% | **55.5%** | +10.6pp |
  | mid (6-20) | 1,588 | 16.5% | 27.6% | **34.1%** | +6.5pp |
  | rare (2-5) | 602 | 11.3% | 20.8% | **22.1%** | +1.3pp |
  | hapax (count=1) | 232 | 7.8% | 8.2% | 6.0% | **−2.2pp** |
  | unseen_in_train | 243 | 4.1% | 3.7% | 3.7% | 0.0pp |

- 결과 요약:
  fine-tuning 이득이 train 빈도에 **단조 비례**한다. hapax 구간은 zero-shot보다 **하락**(Exact@10도 10.8%→9.1%),
  unseen은 변화 없음(3.7%).

- 해석:
  가설 중 후자에 가깝다. fine-tuning은 학습 corpus 분포에 모델을 맞춰 빈출 한자 정확도를 크게 올렸고,
  그 대가로 극희귀자 일반화는 zero-shot 대비 약간 손해를 봤다.
  EXP-008에서 SillokBert zero-shot이 bert-base-chinese를 **전 구간** 상회했던 것과 달리,
  EXP-012는 전 구간 개선이 **아니다**. 발표에서 전체 평균만 제시하며 "모든 구간 개선"으로 서술하면 안 된다.
  다만 실제 결손 복원 수요가 정형구·빈출자에 몰린다면 실용적으로는 여전히 유의미한 이득이다.

- 다음 실험:
  - hapax 하락이 UNK 처리·토큰화 때문인지 별도 확인 (EXP-010 vocab coverage와 교차)
  - 발표 슬라이드에 구간별 표 반영, 한계 항목 갱신
  - (선택) 희귀자 보강을 위한 label smoothing·클래스 가중치는 이번 범위 밖

- 산출물 경로:
  `reports/phase4_stratified_analysis_2026-08-01.md`
  `reports/phase4_stratified_analysis_2026-08-01.json`
  `src/eval/stratified_analysis.py`

- 보존 상태:
  리포트 2종 Git 추적. 입력 예측 CSV는 `outputs/`·`notebooks/`(gitignore) 로컬 보관,
  해시는 `reports/exp012_artifacts/MANIFEST.md`.

- 한계:
  - hapax n=232, unseen n=243로 표본이 작아 구간별 차이의 통계적 유의성은 검정하지 않았다.
  - EXP-004(ngram)는 full test 예측 CSV가 로컬에 남아 있지 않아 이번 비교표에서 제외했다.
### EXP-012 (실행 2026-07-12, 채택 DECISIONS [010]) — SillokBERT GPU full-data fine-tuning + full test

- 유형/상태: `TRAIN` / `completed` (Colab GPU full-data; 공식 full-test 비교 수치)

- 목적/가설:
  full train(46,139)과 validation MLM loss 기반 checkpoint 선택 후,
  full test(9,719)에서 zero-shot SillokBERT(EXP-007) 대비 fine-tuning 이득을 측정한다.
  EXP-011(CPU subsample, test n=2,000)과 표본·설정을 혼동하지 않는다.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` (66,537)
  - train 46,139 / valid 10,679 / test 9,719 (기존 inscription split, `seed=42`)
  - `full_data_no_row_cap=true` (행 상한·subsample 없음)

- 모델:
  `ddokbaro/SillokBert`
  - fine-tuned best: Colab `/content/exp012/sillokbert_exp012_gpu/best`
  - 로컬 백업: `notebooks/best/` (모델·메타·예측; Git 커밋 금지)

- 코드/환경:
  - Google Colab GPU, `device=cuda`
  - 브랜치: `cursor/cloud-agent-1783706016574-58a09`
  - `python -m src.data.finetune_sillokbert --run-mode full --planned-experiment-id EXP-012`
  - `python -m src.eval.run_fill_mask_eval --run-mode full --model-path .../best --require-model-path`
  - runbook: `reports/colab_exp012_runbook.md`, `notebooks/exp012_colab.ipynb`

- 하이퍼파라미터:
  - epochs: `2`
  - batch size: `16`
  - learning rate: `5e-5`
  - max_seq_len: `512`
  - context window: `±64`
  - train_steps: `5620` (epoch당 2810)
  - resume: 없음

- 모델 선택 기준:
  epoch별 **validation MLM loss 최소** (`minimize_valid_loss`).
  - best_epoch: **1**
  - best_valid_loss: **3.66192810216807**
  - epoch 2 valid_loss: 3.7699451506961874 (채택하지 않음)
  - test Exact@k로 checkpoint를 다시 고르지 않음

- test 사용:
  설정·best 고정 후 full test **9,719건 1회**

- 평가 지표 결과 (test n=9,719):
  - Exact@1: **33.75%** (0.3374832801728573)
  - Exact@5: **53.04%** (0.5304043625887437)
  - Exact@10: **60.47%** (0.6046918407243543)
  - Group@1/5/10: 미산출 (매핑 미확정)

- 결과 요약:
  동일 full test n=9,719에서 EXP-007 zero-shot(24.89% / 43.17% / 51.22%) 대비
  Exact@1/5/10 모두 상승. EXP-011(n=2,000, Exact@5 50.45%)과는 표본 크기가 달라 직접 병기·동일시하지 않는다.

- 해석:
  full-data MLM fine-tuning과 valid_loss 기반 epoch 선택이 zero-shot 대비 이득을 냈다.
  best가 epoch 1인 점은 epoch 2에서 valid loss가 소폭 나빠진 것과 일치한다.

- 다음 실험:
  - 전문가 사용성 실험 (`reports/expert_panel/expert_review_panel.csv`)
  - EXP-012 real_defect 전수 예측 보관·정성 검토 (`reports/real_defect_exp012_runbook.md`)
  - 필요 시 층화·실패 사례 정리, 발표 수치 검수
  - (선택) EXP-006/007 local CUDA 재실행

- 산출물 경로:
  - 로컬: `notebooks/best/`
    - `best_checkpoint.json`, `finetune_metrics.json`, `epoch_metrics.json`
    - `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `training_state.pt`
    - `evaluation_summary.csv`, `exp012_sillok_ft_full_predictions.csv` (9,719행, `model_version=exp012_sillokbert_gpu_best_valid_loss`)
  - 재검증: `reports/exp012_metrics_reverify.json` (Exact@k·checkpoint 일치 확인, 2026-07-13)
  - real_defect: `outputs/predictions/exp012_real_defect_predictions.csv` (+ full runbook)
  - 전문가 패널: `reports/expert_panel/`, `handoff/fixtures/expert_review_panel.json`
  - Colab 원 경로: `/content/exp012/sillokbert_exp012_gpu/` (세션 종료 시 소멸 가능)
  - EXP-011 `outputs/finetune/sillokbert_gsm_baseline/` 와 분리·미덮어쓰기

- 보존 상태:
  모델·대용량 예측 CSV는 Git 추적 대상이 아님. 로컬 `notebooks/best/`에 보관.
  전 파일 SHA-256은 `reports/exp012_artifacts/MANIFEST.md`에 기록(2026-08-01).
  **외부 백업 미완료** — 현재 단일 로컬 사본이다.

- 한계:
  - 층화 결과(EXP-013) 기준 fine-tuning 이득은 고빈도 구간에 집중되고 hapax 구간은 하락
  - Colab smoke(`_exp012_smoke`) 수치·메타와 혼동 금지
  - 전수 수동 검수 미완 corpus 기준
  - 인공 마스크 Exact@k ≠ 실제 결손 복원 성능
  - `official_performance` 플래그는 파이프라인 기본값 false이나, 본 EXP-012 full-test는 공식 비교용으로 채택(DECISIONS [010])

### EXP-011 (실행 2026-07-03, 독립 검증·채택 2026-07-08) — SillokBERT CPU fine-tuning baseline

- 유형/상태: `TRAIN` / `completed` (CPU subsample 1차 baseline)

- 목적/가설:
  금석문 인공 마스크 데이터로 `ddokbaro/SillokBert`를 MLM fine-tuning하면,
  동일한 test subsample 조건의 zero-shot SillokBert보다 `Exact@k`가 개선되는지 확인한다.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv`
  - 전체 샘플 수: 66,537
  - 학습 후보: 7,999행
  - 실제 tokenizable 학습 샘플: 7,790개
  - 검증 후보: 2,000행
  - 실제 tokenizable 검증 샘플: 1,947개

- split 방식:
  inscription 단위 split, `seed=42`
  - train inscriptions: 2,355
  - valid inscriptions: 530
  - test inscriptions: 510
  - train/test 교집합: 0
  - train/valid 교집합: 0
  - valid/test 교집합: 0
  - 검증 파일: `work/step1_verification.json`

- 모델:
  `ddokbaro/SillokBert`
  - fine-tuned model 경로:
    `outputs/finetune/sillokbert_gsm_baseline/final/`

- 하이퍼파라미터:
  - device: CPU
  - learning rate: `5e-5`
  - batch size: `8`
  - epochs: `1`
  - train steps: `974`
  - max train rows: `8,000`
  - max valid rows: `2,000`
  - subsample seed: `42`
  - test evaluation: subsample `n=2,000`, batch size `16`, context window `±64`

- 평가 지표 결과:
    - Exact@1: **31.30%**
    - Exact@5: **50.45%**
    - Exact@10: **57.75%**
    - train loss: **4.33**
    - valid loss: **3.74**

- 검증:
  예측 CSV 2,000행을 직접 재집계한 결과,
  Exact@1은 626건(31.30%), Exact@5는 1,009건(50.45%),
  Exact@10은 1,155건(57.75%)으로 기록된 지표와 일치했다

- 비교 결과:
  동일한 `seed=42`, test subsample `n=2,000` 조건의 SillokBert zero-shot 결과
  (`Exact@1=23.65%`, `Exact@5=42.60%`, `Exact@10=50.90%`)와 비교하면:
    - Exact@1: **+7.65%p**
    - Exact@5: **+7.85%p**
    - Exact@10: **+6.85%p**

- test 사용 주의:
  이 비교는 fine-tuning 효과의 1차 확인을 위해 test subsample 2,000개를 사용했다.
  이후 모델·epoch·learning rate 선택에는 이 결과를 사용하지 않고 validation만 사용한다.
  full test 9,719개는 설정을 고정한 최종 모델 평가에 제한한다.

- 결과 요약:
  CPU 1 epoch, train 약 8,000개 subsample이라는 제한된 조건에서도
  SillokBert fine-tuning은 동일 표본 기준 zero-shot보다 모든 `Exact@k`에서 개선됐다.
  특히 전문가 검토용 후보 성능에 해당하는 `Exact@5`가 42.60%에서 50.45%로 상승했다.

- 해석:
  도메인 금석문 인공 마스크 데이터로 추가 학습하는 것이
  사전학습 모델을 그대로 사용하는 zero-shot보다 문맥 복원 성능을 높이는 데 효과가 있었다.
  다만 전체 train 데이터와 전체 test 9,719개를 사용한 결과는 아니므로,
  본 실험은 최종 모델이 아니라 CPU 기반 subsample fine-tuning baseline으로 해석한다.
  test 정답 중 UNK 비율은 2.4%이므로 vocabulary 한계도 함께 고려해야 한다.

- 다음 실험:
  - GPU 환경에서 full train 실행
  - 전체 test split `n=9,719` 평가
  - epoch 수 및 learning rate 조정
  - 후보별 score 저장 기능 추가
  - real defect 정성 평가 연계

- 실험 ID 정리:
  - 공식 실험 ID: `EXP-011`
  - 실행 당시 legacy ID: `EXP-006`
  - 예측 CSV 내부 `experiment_id`와 파일명은 실행 당시 번호를 유지한다.
  - 원본 산출물은 재현성 보존을 위해 이름을 변경하지 않는다.

- 산출물 경로:
  - `outputs/finetune/sillokbert_gsm_baseline/final/`
  - `outputs/finetune/sillokbert_gsm_baseline/finetune_metrics.json`
  - `outputs/predictions/exp006_sillok_ft_predictions.csv`
  - `work/step1_verification.json`
  - `outputs/predictions/real_defect_demo.csv`

- 보존 상태(2026-07-11 감사 기준):
  - 체크포인트, 상세 예측 CSV, 검증 JSON은 `outputs/`·`work/` 아래 로컬 파일이며 Git 미추적이다.
  - PC 외부 백업 위치와 SHA-256은 아직 문서에 기록되지 않았다.
  - 백업을 완료하기 전에는 유일본 소실 위험이 있다.

- 한계:
  - 전체 train이 아닌 약 8,000개 subsample 학습
  - 전체 test가 아닌 2,000개 subsample 평가
  - CPU, 1 epoch 실험
  - 예측 후보의 개별 score가 CSV에 저장되지 않음
  - 산출물의 legacy ID와 공식 실험 ID가 다름

### EXP-010 (2026-07-06) — tokenizer vocabulary coverage (파인튜닝 전 게이트)

- 목적/가설:
  학습 corpus(`is_training_text=True`) 한자가 bert-base-chinese / SillokBert tokenizer에서
  UNK로 떨어지는 비율을 측정해 fine-tuning 전 모델 적합성을 점검한다.

- 데이터 버전:
  `data/processed_layer/readings.csv` (training text only)

- split 방식:
  해당 없음 (corpus 전수)

- 모델:
  `bert-base-chinese`, `ddokbaro/SillokBert`

- 평가 지표 결과:
    - bert-base-chinese: UNK type 45.2%, UNK occurrence **3.42%**
    - SillokBert: UNK type 46.5%, UNK occurrence **2.51%**
    - unique CJK types: 9,023 (동일 corpus)

- 결과 요약:
  SillokBert가 약간 더 많은 글자 종류를 UNK로 보지만, **실제 등장 빈도 기준 UNK는 더 낮음**.
  금석문 특유 한자(尙, 淸, 旣 등)는 양쪽 모두 UNK 후보.

- 해석:
  vocab coverage만으로 모델 선택을 뒤집을 근거는 없음. EXP-008 층화 우위와 결합해 SillokBert 채택.

- 다음 실험:
  Phase 4 SillokBert fine-tuning

- 산출물 경로:
  `reports/tokenizer_vocab_coverage_2026-07-06.md`
  `reports/tokenizer_vocab_coverage_2026-07-06.json`

### EXP-009 (2026-07-06) — cross-split n-gram context overlap audit

- 목적/가설:
  inscription 단위 split 이후에도 test 마스크 위치의 좌/우 n-gram이 train에 등장하는지 측정한다.
  ngram baseline(EXP-004) 성능 중 memorization 비율을 가늠하고, BERT 해석에 참고한다.

- 데이터 버전:
  `artificial_mask_dataset.csv` train 46,139 / test 9,719

- split 방식:
  기존 inscription split (seed=42)

- 모델:
  없음 (데이터 감사)

- 평가 지표 결과:
    - any n-gram key match (n≥1): **99.6%** (unigram 포함, 거의 전부)
    - n-gram key match (n≥2): **59.5%**
    - 8-gram match: 1.8%
    - 4-gram match: 8.1%
    - 2-gram match: 59.5%

- 결과 요약:
  1-gram은 한자 재사용으로 거의 항상 매칭. **2-gram 이상 공유가 59.5%** — 정형구·반복 문맥 영향 큼.

- 해석:
  inscription split은 동일 비문 누수는 막지만 ngram lookup memorization 경로는 남아 있음.
  ngram 28% Exact@5는 이 영향을 상당 부분 받을 수 있음. BERT/SillokBert는 lookup이 아니나
  쉬운 케이스 비율은 높을 수 있음 → EXP-008 층화로 보완 해석.

- 다음 실험:
  Phase 4 fine-tuning; 필요 시 n≥4 매칭 subset만 별도 Exact@k 산출

- 산출물 경로:
  `reports/ngram_cross_split_overlap_2026-07-06.md`
  `reports/ngram_cross_split_overlap_2026-07-06.json`

### EXP-008 (2026-07-06) — Phase 3 BERT zero-shot 층화 분석

- 목적/가설:
  EXP-005(Phase 2)가 ngram/frequency에만 적용된 층화를 BERT zero-shot(EXP-006/007)에 확장한다.
  SillokBert +8.2pp 우위가 고빈도 글자에만 국한되는지 확인한다.

- 데이터 버전:
  EXP-004/006/007 test 예측 + `artificial_mask_dataset.csv` test 9,719행

- split 방식:
  test split (동일)

- 모델:
  후처리 분석 (ngram, bert-base-chinese, SillokBert 예측 csv)

- 평가 지표 결과 (Exact@5 by train label frequency):
    - | 구간 | ngram | bert | SillokBert |
    - | very_common | 47.2% | 54.6% | **58.0%** |
    - | common | 26.7% | 32.7% | **44.9%** |
    - | mid | 9.0% | 16.5% | **27.6%** |
    - | rare | 1.7% | 11.3% | **20.8%** |
    - | hapax | 1.7% | 7.8% | **8.2%** |
    - | unseen | 0.0% | 4.1% | 3.7% |

- 결과 요약:
  SillokBert는 train에서 한 번 이상 등장한 빈도 구간(very_common~hapax)에서 bert-base-chinese를 상회했다.
  rare 구간 격차(+9.5%p)가 특히 컸다. 단, unseen 구간은 SillokBert 3.7%로 bert-base-chinese 4.1%보다 0.4%p 낮았다.

- 해석:
  SillokBert의 우위가 고빈도에만 국한되지는 않았으므로 Phase 4 primary fine-tuning 대상으로 채택할 근거가 된다.
  다만 train 미출현 글자까지 우위라고 해석해서는 안 된다(DECISIONS [008]).

- 다음 실험:
  EXP-009 n-gram overlap audit, EXP-010 vocab coverage, Phase 4 fine-tuning

- 산출물 경로:
  `reports/phase3_stratified_analysis_2026-07-06.md`
  `reports/phase3_stratified_analysis_2026-07-06.json`
  `src/eval/stratified_analysis.py`

### EXP-007 (2026-07-06) — SillokBert zero-shot fill-mask (Phase 3)

- 목적/가설:
  Phase 3 zero-shot으로 `ddokbaro/SillokBert` fill-mask 성능을 artificial_mask test split에서 측정한다. ngram(28.07% Exact@5) 및 bert-base-chinese(EXP-006) 대비 성능을 수치로 기록한다.

- 데이터 버전:
  `gsm_preprocess_v4`
  `data/model_inputs/artificial_mask_dataset.csv` (artificial_mask test only; real_defect 미사용)

- split 방식:
  test split, seed=42. smoke n=500 → full n=9,719.

- 모델:
  `ddokbaro/SillokBert` (HuggingFace, zero-shot, 미세조정 없음)

- 하이퍼파라미터:
  `batch_size=8`, `device=cpu` (클라우드 VM에 GPU 미탑재; 로컬 RTX 3060 Ti에서 `device=cuda` 재실행 권장)
  `--model-kind ddokbaro/SillokBert`, `--split test`

- 평가 지표 결과:

  | run | n | device | Exact@1 | Exact@5 | Exact@10 |
  |---|---:|---|---:|---:|---:|
  | smoke | 500 | cpu | 20.40% | 38.60% | 45.80% |
  | full | 9,719 | cpu | 24.89% | 43.17% | 51.22% |

  비교 기준 (Phase 1 test):
    - frequency Exact@5: 6.60%
    - ngram Exact@5: 28.07%
    - bert-base-chinese (EXP-006) Exact@5: 34.98%

- 결과 요약:
  full test 9,719건 완료. SillokBert zero-shot Exact@5 43.17% — ngram(28.07%) 대비 +15.1pp, bert-base-chinese EXP-006(34.98%) 대비 +8.2pp, frequency(6.60%) 대비 대폭 우위.

- 해석:
  한국 고전 도메인 사전학습 모델이 일반 중국어 BERT·ngram 규칙보다 우수. fine-tuning 여부는 Phase 4에서 판단.

- 다음 실험:
  로컬 RTX 3060 Ti에서 `device=cuda`로 동일 명령 재실행해 하드웨어 수용 기준 충족. Phase 4 SillokBert fine-tuning 검토.

- 산출물 경로:
  `reports/evaluation/exp007_sillokbert_test_summary.csv` (커밋됨)
  `outputs/predictions/exp007_sillokbert_test_predictions.csv` (gitignore)
  `outputs/logs/exp007_smoke.log`, `outputs/logs/exp007_full.log` (gitignore)

### EXP-006 (2026-07-06) — bert-base-chinese zero-shot fill-mask (Phase 3)

- 목적/가설:
  Phase 3 zero-shot 베이스라인으로 `bert-base-chinese` fill-mask 성능을 artificial_mask test split에서 측정한다. ngram(28.07% Exact@5) 대비 개선 여부를 확인한다.

- 데이터 버전:
  `gsm_preprocess_v4`
  `data/model_inputs/artificial_mask_dataset.csv` (artificial_mask test only; real_defect 미사용)

- split 방식:
  test split, seed=42. smoke n=500 → full n=9,719.

- 모델:
  `bert-base-chinese` (HuggingFace, zero-shot, 미세조정 없음)

- 하이퍼파라미터:
  `batch_size=8`, `device=cpu` (클라우드 VM에 GPU 미탑재; 로컬 RTX 3060 Ti에서 `device=cuda` 재실행 권장)
  `--model-kind bert-base-chinese`, `--split test`

- 평가 지표 결과:

  | run | n | device | Exact@1 | Exact@5 | Exact@10 |
  |---|---:|---|---:|---:|---:|
  | smoke | 500 | cpu | 17.80% | 33.00% | 39.40% |
  | full | 9,719 | cpu | 19.69% | 34.98% | 42.38% |

  비교 기준 (Phase 1 test):
    - frequency Exact@5: 6.60%
    - ngram Exact@5: 28.07%

- 결과 요약:
  full test 9,719건 완료. bert-base-chinese zero-shot Exact@5 34.98%로 ngram(28.07%) 대비 +6.9pp, frequency(6.60%) 대비 대폭 우위.

- 해석:
  일반 중국어 BERT가 문맥 기반 ngram 규칙보다 우수하나, Phase 2 고빈도 label(47.1%)에는 미달. 한국 고전/금석문 도메인 특화 없이는 상한이 제한적임을 시사.

- 다음 실험:
  EXP-007 SillokBert zero-shot과 비교. 로컬 RTX 3060 Ti에서 `device=cuda` 재실행 권장.

- 산출물 경로:
  `reports/evaluation/exp006_bert_test_summary.csv` (커밋됨)
  `outputs/predictions/exp006_bert_test_predictions.csv` (gitignore)

### EXP-005 (2026-07-06) — Phase 2 baseline 층화 분석 (EXP-003/004 후처리)

- 목적/가설:
  baseline 점수가 어떤 조건(정답 글자 빈도, 텍스트 길이, 시대)에서 강하고 약한지 파악. 마스킹 v2 변경 전 데이터 진단.

- 데이터 버전:
  EXP-003/004 test 예측 + `artificial_mask_dataset.csv` test 9,719행

- split 방식:
  test split (동일)

- 모델:
  후처리 분석 only (frequency, ngram 예측 csv 재사용)

- 평가 지표 결과 (ngram Exact@5 하이라이트):
    - very_common label (train≥100): **47.1%** (n=3,334)
    - common (21-100): 26.9%
    - mid (6-20): 9.0%
    - rare/hapax (≤5): **~1.7%**
    - unseen_in_train: **0.0%** (n=243)
    - ultra_long (≥5000자): short 대비 낮음 (리포트 참조)

- 결과 요약:
  ngram 성능은 고빈도 정답 글자·train에 본 문맥에 집중. 희귀자·미출현 글자는 거의 실패.

- 해석:
  Exact@5 28%는 고빈도 글자가 끌어올린 값. 희귀 한자 복원은 BERT도 어려울 수 있음 → Phase 3에서 확인.
  마스킹 규칙 v2보다 층화 리포트·BERT가 우선.

- 다음 실험:
  bert-base-chinese / SillokBert zero-shot (EXP-006/007)

- 산출물 경로:
  `reports/phase2_stratified_analysis_2026-07-06.md`
  `reports/phase2_stratified_analysis_2026-07-06.json`

### EXP-004 (2026-07-06) — ngram context baseline (test split)

- 목적/가설:
  train split 문맥(좌우 n-gram)으로 마스크 위치 한자를 맞출 수 있는지 확인. frequency baseline 대비 개선 여부.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` (gsm_preprocess_v4, 66,537 samples)
  train 46,139 / test 9,719

- split 방식:
  `splits.csv` inscription 단위, seed=42. 평가 split=test.

- 모델:
  `NgramContextBaseline` — n-gram sizes (8,4,2,1), context_window=64, train label 기반

- 하이퍼파라미터:
  `python -m src.eval.run_fill_mask_eval --experiment-id EXP004 --model-kind ngram --split test`
  smoke: `--max-samples 500` → full test 9,719

- 평가 지표 결과 (test, n=9,719):
    - Exact@1: **14.48%**
    - Exact@5: **28.07%**
    - Exact@10: **34.67%**
    - smoke (n=500): Exact@1 13.8%, Exact@5 24.2%, Exact@10 30.0%
    - era (주요): 조선 Exact@5 29.5%, 고려 24.0%, 일제강점기 24.3%

- 결과 요약:
  frequency 대비 Exact@1 +12.2%p, Exact@5 +21.5%p. 문맥 기반 비딥러닝 baseline이 유의미한 기준선.

- 해석:
  동일 좌우 문맥이 train에 있으면 맞출 수 있으나, 여전히 70% 이상은 top-5에도 정답 없음 → BERT zero-shot 여지 있음.

- 다음 실험:
  bert-base-chinese / SillokBert zero-shot (EXP-006/007).

- 산출물 경로:
  `reports/evaluation/exp004_ngram_test_summary.csv` (커밋됨)
  `outputs/predictions/exp004_ngram_test_predictions.csv` (gitignored, 로컬)

### EXP-003 (2026-07-06) — frequency unigram baseline (test split)

- 목적/가설:
  eval 파이프라인 end-to-end 동작 확인 + 가장 단순한 기준선(train label 빈도 top-k) 수치 확보.

- 데이터 버전:
  `data/model_inputs/artificial_mask_dataset.csv` (gsm_preprocess_v4, 66,537 samples)
  train 46,139 / test 9,719

- split 방식:
  `splits.csv` inscription 단위, seed=42. 평가 split=test.

- 모델:
  `FrequencyBaseline` — train `label_char` 빈도 순 top-10 (문맥 무시)

- 하이퍼파라미터:
  `python -m src.eval.run_fill_mask_eval --experiment-id EXP003 --model-kind frequency --split test`
  smoke: `--max-samples 500` → full test 9,719

- 평가 지표 결과 (test, n=9,719):
    - Exact@1: **2.25%**
    - Exact@5: **6.60%**
    - Exact@10: **10.33%**
    - smoke (n=500): Exact@1 1.2%, Exact@5 5.2%, Exact@10 9.2%

- 결과 요약:
  문맥 없이 고빈도 한자만 맞추는 하한선. 이후 모든 모델은 이 수치보다 높아야 의미 있음.

- 해석:
  파이프라인(로드→예측→Exact@k) 정상. 금석문 도메인에서는 빈도만으로는 거의 부족.

- 다음 실험:
  ngram baseline (EXP-004).

- 산출물 경로:
  `reports/evaluation/exp003_freq_test_summary.csv` (커밋됨)
  `outputs/predictions/exp003_freq_test_predictions.csv` (gitignored, 로컬)

### EXP-002 (2026-07-06) — gsm_preprocess_v4 재정규화 및 파생 데이터셋 재생성

- 목적/가설:
  v3에서 결손(`□`)과 한글 치환이 동일 기호로 합쳐진 문제를 해소하기 위해 v4(결손=`□`, 한글=`◇`)로 `readings.csv`를 재정규화하고 파생 데이터셋을 재생성한다.

- 데이터 버전:
  `data/processed_layer/readings.csv` (`gsm_preprocess_v4`)
  `data/model_inputs/artificial_mask_dataset.csv`
  `data/model_inputs/real_defect_inference.csv`

- split 방식:
  기존 `splits.csv` 유지 (inscription 단위, 변경 없음).

- 모델:
  없음. 데이터 재생성·검증 단계.

- 하이퍼파라미터:
  `python -m src.data.reprocess_readings`
  `python -m src.data.build_artificial_mask`
  `python -m src.data.build_real_defect_inference`

- 평가 지표 결과:
  성능 평가 아님. 정합성 검증:
    - `is_training_text=True` row: 7,158 (v3와 동일)
    - `has_real_defect=True` row: 5,286 (v3 7,544 → 한글-only 2,258건 제외)
    - 실제 결손 `□` 총량: 152,277 (v3 475,423에서 한글 혼입분 제거)
    - `artificial_mask_dataset` sample: 66,537 (동일)
    - `real_defect_inference` 한글-only reading 포함: 0건 (v3 807건 → 0)
    - 단위 테스트: 7 passed

- 결과 요약:
  결손과 한글 치환을 텍스트·플래그·데모 데이터에서 분리했다. baseline 학습용 artificial mask 규모는 변하지 않았다.

- 해석:
  `has_real_defect`와 결손 통계가 실제 금석문 결손에 더 가깝게 정의되었다. `real_defect_inference`는 이제 `□` 위치만 포함한다.

- 다음 실험:
  pretrained fill-mask baseline 평가 (EXP-001 후속).

- 산출물 경로:
  `data/processed_layer/readings.csv`
  `data/processed_layer/defect_markers_inventory.json`
  `data/model_inputs/artificial_mask_dataset.csv`
  `data/model_inputs/real_defect_inference.csv`

### EXP-001 (2026-07-03) — artificial mask dataset 재생성 및 정합성 검증

- 목적/가설:
  기존 `artificial_mask_dataset.csv`가 현재 `readings.csv`와 불일치하여, 현재 `is_training_text=True` 기준으로 mask dataset을 재생성한다.

- 데이터 버전:
  `data/processed_layer/readings.csv`
  `data/model_inputs/splits.csv`

- split 방식:
  기존 `splits.csv` 사용. inscription 단위 split.
  train / valid / test split mismatch 0건.

- 모델:
  없음. 데이터 생성/검증 단계.

- 하이퍼파라미터:
  `python -m src.data.build_artificial_mask`
  reading당 mask 생성 규칙은 현재 코드 기준.

- 평가 지표 결과:
  성능 평가 아님.
  데이터 정합성 검증:
    - 총 sample: 66,537
    - train sample: 46,139
    - valid sample: 10,679
    - test sample: 9,719
    - non-training reading 포함: 0건
    - split null/공란: 0건
    - splits.csv 미존재 inscription: 0건
    - mask.split과 splits.csv mismatch: 0건

- 결과 요약:
  기존 stale sample 531개가 제거되었고, 현재 eligible training reading 전체 기준으로 66,537개 artificial mask sample이 생성되었다.

- 해석:
  baseline 학습/평가에 사용할 artificial mask dataset의 현재 `readings.csv` 기준 정합성이 확보되었다.
  단, 장문 training row 9건의 truncation spot-check는 별도 확인 필요하다.

- 다음 실험:
  pretrained fill-mask baseline 평가:
    1. `bert-base-chinese`
    2. `ddokbaro/SillokBert`
  지표: Exact@1/5/10, 가능하면 era별 Exact@k.

- 산출물 경로:
  `data/model_inputs/artificial_mask_dataset.csv`
---

*관련 문서: `DECISIONS.md`(결정), `DATA_PREPROCESSING_RULES.md`(데이터·split), `PROJECT_OVERVIEW.md`(현재 상태), `RISK_CHECKLIST.md`(평가 위험).*

---

## 2026-09-17 업데이트 요약 — ARI-8B / EXP-019

### 0. 병합 메모 (문서 정합성)

- 이 업데이트는 원래 EXP-018 GPU Phase B(018A/018B 학습 완료, epoch별 loss 표) 실행 확인 내용을 담은
  2026-08-30 버전과 별도로 작성되어, 두 버전을 그대로 교체하면 018A/018B 학습 결과가 유실될 위험이 있었다.
  이번 병합에서는 **EXP-018의 GPU Phase B 학습 기록(018A/018B loss 표, 해석, 한계)을 삭제하지 않고 그대로 유지**했고,
  아래 EXP-019 관련 신규 내용만 추가했다.

### 1. 기존 진행상황 문서에서 변경된 부분

- 최상단 현재 상태를 2026-09-17 기준으로 갱신했다. EXP-018은 여전히 GPU Phase B와 동일 Exact@k가 미실행이며, EXP-019는 설계·감사 완료, 구현·실행 미완료로 분리했다.
- 실험 기록 최신순 첫 항목에 `EXP-019`를 추가했다. 기존 EXP-001~018 기록과 수치는 삭제·수정하지 않았다.
- EXP-015를 현 시점 ARI 직접 비교 기준선으로 명시하고, EXP-018은 동일 Exact@k 부재로 비교 baseline이 아님을 기록했다.
- inscription-level split의 안전성뿐 아니라 cross-split exact duplicate 26건과 retrieval leakage audit 필요성을 추가했다.
- ChromaDB 담당자 이탈을 반영해 019C를 `deferred / optional`로 변경하고, 019A와 019B를 critical path로 고정했다.

### 2. 현재 EXP-019 상태표

| 항목 | 상태 | 확정 내용 | 아직 없는 것 |
|---|---|---|---|
| 통합 가능성 감사 | 완료 | 별도 ARI runner로 기존 valid 및 `src/eval/metrics.py` 재사용 가능 | 실제 runner 코드 |
| Top-K 설계 | 완료 | Appendix C 기반 constrained decoding, PAPER/PUBLIC/OURS/UNKNOWN 구분 | decoder 구현·검증 |
| EXP-019A no-retrieval | 구현 전 | EXP-015와 동일 Exact@k 비교 예정 | 모델 다운로드·GPU smoke·전체 valid 결과 |
| EXP-019B BM25 | 대기 | 019A 규칙 freeze와 leakage audit 후 진행 | BM25 index·audit·성능 수치 |
| EXP-019C ChromaDB/dense | deferred / optional | 담당자 이탈로 critical path 제외 | 담당자·audit·구현·성능 수치 |

### 3. 다음 액션 5개

1. `docs/ARI_TOPK_DESIGN.md`에 따라 constrained Top-K decoder를 구현한다.
2. tokenizer byte 경로, 한자 완성 판정, 중복 제거, 후보 정렬을 GPU 없이 unit test한다.
3. 소량 sample에서 ARI-8B 모델 로딩, prompt, JSON parser만 확인하는 `generate()` smoke를 수행한다.
4. constrained Top-1과 `generate()` parsing Top-1 일치율, B1/B2/Lmax sensitivity를 측정하고 Top-K 규칙을 freeze한다.
5. EXP-019A full validation을 실행한 뒤 BM25 retrieval corpus의 exact/near-duplicate leakage audit을 통과시켜 EXP-019B로 진행한다.

### 4. blocker / risk

- 공개 ARI 코드에 필요한 Top-5 constrained decoder가 없어 세부 구현 일부가 OURS다.
- B1/B2/Lmax가 후보 탐색 폭에 영향을 주므로 sensitivity 확인 전 Exact@5를 모델 성능으로 해석할 수 없다.
- Qwen tokenizer의 multi-token/byte 경로, invalid Hanja output, JSON parsing 실패를 실제 모델에서 검증하지 않았다.
- T4 16GB 4-bit inference의 로딩·VRAM·속도는 아직 실측하지 않았다.
- cross-split duplicate 26건 때문에 retrieval은 train-only index만으로 충분하다고 단정할 수 없다. 019B/C 전에 exact/near-duplicate 및 retrieved context leakage audit이 필요하다.
- 019C가 보류된 상태에서는 ChromaDB/dense retrieval의 효과나 BM25 대비 우열을 결론낼 수 없다.

### 5. 현재 발표에서 주장 가능한 것과 아직 주장하면 안 되는 것

**주장 가능**

- ARI-8B를 기존 SillokBERT single-mask validation 및 Exact@k evaluator에 맞춰 평가하는 최소 통합 설계를 확보했다.
- EXP-015가 현 시점 직접 비교 기준선이며, EXP-018은 동등 Exact@k가 없어 비교 기준선이 아니다.
- BM25 retrieval 유무를 통제한 ablation이 현재 핵심 연구 질문이며, retrieval leakage 통제가 필요하다.
- ARI Top-K는 BERT softmax와 다른 constrained decoding 문제이므로 후보 생성 규칙을 명시적으로 고정해야 한다.

**아직 주장하면 안 됨**

- ARI-8B가 EXP-015보다 성능이 좋다.
- ARI-8B가 GPU에서 정상적으로 동작한다.
- BM25 retrieval이 성능을 개선한다.
- ChromaDB/dense retrieval의 성능 향상 또는 BM25 대비 우열이 입증됐다.
- ARI Appendix C를 세부까지 그대로 재현했다.
- 최적 retriever를 찾았거나 ARI를 최종 채택했다.

---

## 2026-09-17(후속) 업데이트 요약 — EXP-018 post-hoc Exact@k 평가 결과

이 절은 같은 날짜(2026-09-17) 안에서 위 "ARI-8B / EXP-019" 요약보다 **나중에** 추가된 내용이다.
EXP-018 섹션 본문(위쪽 실험 기록)이 정본이며, 이 절은 빠르게 훑어볼 수 있는 요약이다.

- 018A/018B 모두 post-hoc valid Exact@k(subsample n=2,000, seed=42, checkpoint-epoch-1~4 전체 평가)를
  공식 실행했다. repo commit `c6df2c2d4c6e215c850df66e1fa0d36c9200895e`, GPU NVIDIA L4, test 미사용.
- **018A**(best epoch=1): Exact@1 34.05% / @5 52.20% / @10 59.70%. vs EXP-015(subsample) 53.40%: **−1.20pp**, McNemar p=0.093455 → **gate 미통과**.
- **018B**(best epoch=2, Exact@5 기준 — valid_loss 기준 best epoch 1과 다름): Exact@1 35.65% / @5 54.10% / @10 61.55%. vs EXP-015(subsample) 53.40%: **+0.70pp**, McNemar p=0.350642 → **통계적으로 유의하지 않아 gate 미통과**.
- **018A/018B 어느 쪽도 공식 incumbent로 승격하지 않는다. EXP-015(공식 full-valid Exact@5 53.24%, n=10,679)를 계속 공식 valid baseline으로 유지한다.**
- full valid(n=10,679) 승격 평가는 수행하지 않았다 — 위 수치는 전부 subsample n=2,000 기준이다.
- `gate_results.csv`가 자동 계산한 +0.86pp(full-valid 53.24% vs subsample 54.10%, 표본 상이)는 paired 개선량으로 사용하지 않는다. 공식 paired 개선량은 동일 subsample 기준 **+0.70pp**다.
- 018A/018B 체크포인트 실제 저장 경로가 확인됐다(`/content/drive/MyDrive/exp018/018a/`, `018b/`, 각 `checkpoint-epoch-1`~`4`, `model.safetensors`+tokenizer 확인). 기존 "저장 위치 TODO"는 해소됐다.
- `drive_loss_vs_docs.csv` 대조 결과 018A/018B 학습 loss 표 일부가 Drive 실제 값과 다를 수 있음이 확인됐으나, 본 갱신에서는 구체 수치를 확보하지 못해 표를 정정하지 못했다 — **TODO로 남긴다.**
- 018C(lr=2e-5)는 여전히 미실행/결과 미확보 상태이며 후순위다.
- **다음 최우선 작업은 EXP-019(ARI-8B) Top-K constrained decoder 구현이다.**

---

## 2026-10-01 업데이트 요약 — 발표 수치 고정 · EXP-021–028

실험 기록 본문이 정본이다. 이 절은 그날 합의만 짧게 모은 것이다(DECISIONS [017]).

- 발표 S5 Exact@k는 EXP-025 동일 n=200 **근사 Exact(beam5)** 로 고정. Exact@5 Sillok 0.515 > ARI-FT 0.410 > ZS 0.225 (McNemar 양쪽 유의). Exact@1 0.350 / 0.220 / 0.105. Exact@10은 Sillok 0.575. 슬라이드의 ARI Exact@10은 **n.a. (beam5)**. beam10 재평가 없음. SOTA·공식 Appendix C Top-K 아님.
- 금지: EXP-015 n≈500 Exact@5 0.504와 EXP-024 0.410의 직접 비교. 공식 ARI Top-K / Appendix C / SOTA 주장.
- EXP-021: n=200 beam5 근사 Exact@1 0.105 / Exact@5 0.225.
- EXP-024: A100 full, train_loss 0.4577, adapter `MyDrive/exp024/full/adapter_final`, n=200 beam5 Exact@1 0.220 / @5 0.410 (vs ZS +11.5 / +18.5 pp).
- EXP-025: 위 잠금의 paired·McNemar. PR #24는 `main`에 병합(EXP-025만). Drive `MyDrive/exp025/paired_exact_compare.json`.
- EXP-023: 이미 완료. **재실행하지 않음.** 지표는 `reports/exp023/`이 정본이며 여기 다시 적지 않는다.
- EXP-026: 건너뜀.
- EXP-028: 정성 n=25 완료. Exact@k 없음. side-by-side Top-1 완전 일치 0. 프레이밍은 전문가 검토용 불확실 후보(복원 성공 아님). 데모 고정: real_defect `real_000275`(Sillok 戊 / ARI-FT 甲), `real_000445`(Sillok 山 / ARI-FT 此, 山은 Top-5). artificial_mask 성공 3·실패 2는 DECISIONS [017]. Drive `MyDrive/exp028/exp028_sillok_vs_ari_ft_side_by_side.csv` (+ raw/summary). PR #26은 `exp024/ari-qlora`.
- EXP-027: 야간 A100 `--mode full --train-pool -1` **진행 중**. 출력 `MyDrive/exp027`. 비공식. 2026-10-02 오전 미완이면 중단. **완료 지표 없음.** PR #25(`--train-pool -1`)는 `exp024/ari-qlora`에 병합.
- 027·028 숫자는 S5/S10에 넣지 않는다. Cursor가 commit/push하고 GitHub 팀원이 review/merge한다. Colab 산출물은 Drive에 둔다.

---

## 2026-10-01(후속) 업데이트 요약 — EXP-029

실험 기록 EXP-029가 정본이다. 위 2026-10-01 절의 S5 잠금 숫자는 그대로다(DECISIONS [017]·[018]).

- 방법: **mean token logprob** Top-K. one-pass KV cache + `expand_character_candidates`. Appendix C–style. **공식 ARI Top-K 아님.** 순위 함수 이름은 mean token logprob.
- adapter: `MyDrive/exp024/full/adapter_final` (EXP-024 QLoRA). 스크립트 `src/eval/ari/score_exp029.py` (PR #30, `exp024/ari-qlora`에 병합).
- n=200 pilot: Exact@1/5/10 = 0.205 / 0.385 / 0.445. beam5 top1_agreement 0.715. mismatch review는 mean 대 beam(sum)의 예상 divergence(버그 아님). GO for n=2000.
- n=2000 FT: Exact@1 **0.239** / Exact@5 **0.4115** / Exact@10 **0.4875**. multi-token gold n=144 Exact@5 ≈0.361.
- paired vs EXP-015 Sillok, 동일 n=2000: Exact@1 0.343 vs 0.239 (Δ−10.4pp, only_S 345 / only_ARI 137, p≈4e-21). Exact@5 0.534 vs 0.4115 (Δ−12.3pp, only_S 417 / only_ARI 172, p≈9e-24). Exact@10 0.607 vs 0.4875 (Δ−12.0pp, only_S 405 / only_ARI 166, p≈2e-23).
- 산출물: Drive `MyDrive/exp029/` (summary, n=2000 predictions, raw jsonl, mismatch_review).
- **S5는 바꾸지 않는다.** 잠금은 Sillok **0.515** / ARI-FT beam5 **0.410** / ZS **0.225** (n=200). **0.4115를 0.410과 같게 읽지 않는다. Sillok 0.534를 0.515와 같게 읽지 않는다.**
- EXP-027 야간 학습은 별개이며 아직 돌고 있을 수 있다. **027 결과를 적지 않는다.** 다음 축은 Claude 이후 EXP-030+.
