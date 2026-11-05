# EXP-014 — hapax 하락 원인 탐색 (가설 수준)

> **결론을 강하게 쓰지 말 것.** hapax n=232(full test) / EXP-011 겹침 n은 더 작다.

## 0. 데이터 가용성

- artificial_mask: `data/model_inputs/artificial_mask_dataset.csv`
- EXP-011 FT predictions: `reports/exp011_baseline/exp006_sillok_ft_predictions.csv`
- EXP-012 full predictions: **없음** (로컬 `notebooks/best/` 미탑재)
- EXP-007 per-sample predictions: **없음** (요약 CSV만)

## 1. Zero-shot 층화 기준치 (EXP-008 / phase3)

| 모델 | hapax Exact@1 | hapax Exact@5 | hapax n | unseen Exact@5 |
|------|-------------:|-------------:|--------:|---------------:|
| EXP-006 bert-base-chinese zs | 5.2% | 7.8% | 232 | 4.1% |
| EXP-007 SillokBert zs | 3.4% | 8.2% | 232 | 3.7% |

## 2. EXP-011 FT subsample에서 같은 빈도 버킷

| 버킷 | n (subsample) | Exact@1 | Exact@5 | Exact@10 |
|------|--------------:|--------:|--------:|---------:|
| hapax | 48 | 0.0% | 2.1% | 2.1% |
| rare | 134 | 12.7% | 25.4% | 31.3% |
| unseen | 49 | 0.0% | 2.0% | 2.0% |

동일 full-test hapax 232와 표본이 다르므로 **직접 동일시하지 않는다.**
다만 EXP-011 겹침 hapax에서 Exact@5가 zero-shot 8.2%보다 낮게 관측되면
“FT가 희귀 정답 복원을 약화했을 가능성”을 **가설**로 둔다.

## 3. Hapax 정답 × tokenizer UNK (EXP-010 교차)

### unk_hapax_test
- model `ddokbaro/SillokBert`: UNK/multi-piece **43/232** (18.5%)
- examples: 攷, 鎰, 悏, 韙, 槪, 徉, 玹, 黽, 紈, 竁, 狎, 徉, 㑹, 侫, 衲

### unk_hapax_exp011_overlap
- model `ddokbaro/SillokBert`: UNK/multi-piece **6/48** (12.5%)
- examples: 狎, 劒, 瑬, 黽, 攷, 㑹

Fine-tuning은 tokenizer를 바꾸지 않으므로 UNK 비율 자체는 FT 전후 동일하다.
UNK hapax는 ZS·FT 모두 Exact@1이 원리적으로 어렵다(단일 토큰 복원).

## 4. 쌍대 비교 (EXP-011 hapax 겹침 × SillokBert zero-shot 재평가)

동일 48건에 대해 pretrained `ddokbaro/SillokBert` zero-shot을 재실행해 FT와 짝비교.

- overlap n: **48**
- ZS Exact@1/5: 2.1% / 4.2%
- FT Exact@1/5: 0.0% / 2.1%
- ZS@5∧¬FT@5 (하락 후보): **1**
- 상세 CSV: `reports/exp014_hapax_paired_cases.csv`

| failure_type | n | 의미 |
|-------------|--:|------|
| both_miss5 | 46 | ZS·FT 모두 top-5 미스 (지배적) |
| zs_hit5_ft_miss5 | 1 | ZS는 top-5 적중, FT는 이탈 |
| zs_hit1_ft_rank_drop | 1 | ZS top-1 적중 → FT는 top-1 이탈·top-5 잔류 |

### 하락·순위변동 사례 (전수 2건)

| sample_id | era | gold | ZS top-5 | FT top-5 | type |
|-----------|-----|------|----------|----------|------|
| `mask_00013462` | 대한제국 | 鱗 | 玉;巨;金;風;**鱗** | 石;巨;玉;龍;沙 | zs_hit5_ft_miss5 (`碧[MASK]鱗` 시구) |
| `mask_00036356` | 조선 | 剃 | **剃**;削;長;斷;毫 | 削;**剃**;毛;斷;白 | zs_hit1_ft_rank_drop (`私自[MASK]髮`) |

**표본이 너무 작아** “FT가 hapax를 체계적으로 망친다”고 말할 수 없다.  
관측은 “겹침 48건에서 양쪽 모두 거의 실패하며, ZS→FT 명확 하락은 1건” 정도다.

## 5. 해석 (가설만)

1. FT는 전체 Exact@k를 올리지만, hapax/unseen subsample에서는 이득이 없거나 소폭 감소처럼 보인다.
2. Tokenizer UNK는 full-test hapax의 **18.5%**(43/232)만 설명. FT 전후 동일하므로 “FT가 UNK를 늘렸다”는 가설은 기각.
3. 나머지 실패는 희귀자·문맥 부족·고빈도 편향 쪽으로 보는 편이 안전하다. 실패 유형 분류의 주류는 `both_miss5`.
4. EXP-012 full 9,719 예측이 오면 `scripts/analyze_hapax_decline.py`로 hapax 232 전수를 재집계할 것.

## 산출물

- JSON: `reports/exp014_hapax_decline.json`
- 본 리포트: `reports/exp014_hapax_decline.md`

