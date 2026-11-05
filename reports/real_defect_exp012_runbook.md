# EXP-012 real_defect 예측 실행 가이드

`prediction_type=real_defect` CSV를 EXP-012 체크포인트(`notebooks/best`)로 생성한다.  
정답이 없으므로 Exact@k를 계산하지 않는다. 점수는 **상대 선호 점수**이며 신뢰 확률이 아니다.

## 사전 조건

- `notebooks/best/`에 `model.safetensors`, tokenizer, `config.json`
- `data/model_inputs/real_defect_inference.csv` (5,000건)
- GPU 권장. CPU는 수 시간 소요 가능.

현재 산출: `outputs/predictions/exp012_real_defect_predictions.csv`는
`--max-cases 0` + 기본 `clean_only`(한글 문맥 제외) 기준 **전수 3,704건**이다.
(`real_defect_inference.csv` 원본 5,000건 중 hangul-context 행 제외.)

전수 재실행:

```bash
python -m src.eval.run_real_defect_demo --max-cases 0 --batch-size 8 --notes "full real_defect EXP-012" --output-csv outputs/predictions/exp012_real_defect_predictions.csv
```

한글 문맥 포함 전수(5,000)가 필요하면 `--include-hangul-context`를 추가한다.

## 명령

프로젝트 루트에서:

```bash
# 스모크 (예: 32건)
python -m src.eval.run_real_defect_demo --max-cases 32 --batch-size 8 --notes "smoke"

# 전수 (max-cases 0)
python -m src.eval.run_real_defect_demo --max-cases 0 --batch-size 8 --notes "full real_defect EXP-012"

# 기본 출력
# outputs/predictions/exp012_real_defect_predictions.csv
# model_version=exp012_sillokbert_gpu_best_valid_loss
```

## 전문가 패널

```bash
python -m src.eval.select_expert_panel \
  --n-artificial 25 --n-real 25 \
  --out-csv reports/expert_panel/expert_review_panel.csv \
  --out-json handoff/fixtures/expert_review_panel.json
```

## 산출물

| 파일 | 용도 |
|------|------|
| `outputs/predictions/exp012_real_defect_predictions.csv` | 전수/부분 예측 (대용량, git 비권장) |
| `reports/expert_panel/expert_review_panel.csv` | 전문가용 혼합 패널 |
| `handoff/fixtures/expert_review_panel.json` | 프론트 모킹/연동 |
