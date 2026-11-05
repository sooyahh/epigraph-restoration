# EXP-012 SillokBERT — 프론트/백엔드 전달 패키지

실록 한자 원문에서 `[MASK]` 1글자를 예측하는 모델의 **파이프라인 연동용** 전달물입니다.
**서빙·발표용 공식 후보는 여전히 EXP-012**(`exp012_sillokbert_gpu_best_valid_loss`)입니다.
FT 개선 루프의 valid 기준선은 EXP-015(신규 모델 비교의 공식 incumbent, DECISIONS [014]). EXP-018(HP 재탐색)은 GPU Phase B 018A·018B 학습(4epoch)은 완료됐으나 Exact@k 공식 평가는 아직 실행 전이라 새 공식 수치는 없습니다(018C는 결과 미확보).
이 패키지의 목적은 **프론트·백엔드가 계약(스키마)에 맞춰 작업을 진행**하도록 하는 것입니다.

> ⚠️ 아래 Exact@k는 **Google Colab GPU full 실행(EXP-012) test** 결과입니다. CPU/부분실행·valid-only 값이 아닙니다.

## 폴더 구성

```
handoff/
├─ README.md                     # 이 문서
├─ fixtures/
│  ├─ predict_samples.json       # 프론트 모킹용 샘플 12건 (연속 비문 — 모킹 전용)
│  └─ expert_review_panel.json   # 전문가 사용성 평가용 혼합 패널 (분산 샘플)
├─ backend/
│  ├─ app.py                     # FastAPI 추론 스텁 (/predict, /health)
│  └─ requirements.txt
└─ exp018/                       # FT iteration용 (FE/BE 계약 아님)
   ├─ exp015_sillok_ft_valid_predictions.csv
   └─ exp015_sillokbert_valid_summary.csv
```

- `fixtures/`, `backend/`: 프론트·백엔드 연동용.
- `exp018/`: EXP-015 full-valid 예측·요약 사본. EXP-018 등 FT 개선 비교용이며 **API·UI 모킹에 쓰지 말 것**.

모델 가중치(`model.safetensors` 등)는 용량 문제로 Git·이 폴더에 포함하지 않습니다 → `notebooks/best/`(로컬) 또는 별도 공유 스토리지로 전달.

## API 계약

### `POST /predict`

요청:
```json
{ "text": "...貧道[MASK]居長逰庵...", "top_k": 10 }
```
- `text`: `[MASK]` 토큰을 **정확히 1개** 포함한 원문 (한자). 없거나 2개 이상이면 `422`.
- `top_k`: 1~50, 기본 10.

응답 (`200`):
```json
{
  "input_text": "...貧道[MASK]居長逰庵...",
  "mask_index": 40,
  "candidates": [
    { "rank": 1, "char": "所", "score": 0.043539 },
    { "rank": 2, "char": "而", "score": 0.042441 }
  ],
  "model_version": "exp012_sillokbert_gpu_best_valid_loss",
  "meta": { "max_seq_len": 512, "context_window": 64 }
}
```
- `mask_index`: 원문에서 `[MASK]`가 시작하는 **문자 오프셋**(0-based). UI 하이라이트용.
- `candidates`: **후보 간 상대적 선호 점수** 내림차순. vocab softmax에서 온 값이지만 **보정된 “정답일 확률/신뢰 확률”이 아님**. `score` 합은 1이 아님(top_k만 반환). 전문가 UI에서 “신뢰 확률”이라 부르지 말 것.
- `char`가 `##`/`[UNK]` 등 서브워드·특수토큰으로 나올 수 있음 → UI에서 필터링/표시 정책 필요.

### `GET /health`
`{ "status": "ok", "model_dir": "...", "model_version": "exp012_sillokbert_gpu_best_valid_loss" }`

## 백엔드 실행

```bash
cd handoff/backend
pip install -r requirements.txt
export MODEL_DIR=../../notebooks/best      # Windows PowerShell: $env:MODEL_DIR="..\..\notebooks\best"
uvicorn app:app --host 0.0.0.0 --port 8000
```

모델 번들에 필요한 파일(공유 스토리지에서 `MODEL_DIR`로 전달):
`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
(`training_state.pt` 1.4GB는 옵티마이저 상태 — 추론 불필요, 전달 제외)

전처리는 학습(EXP-012)과 동일하게 `[MASK]` 좌우 **±64자**만 사용합니다(EXP-016: 추론 window ±128/±256 확대는 개선 없음·악화). 512 토큰 truncation에서 마스크가 잘리지 않도록 합니다.

## 프론트엔드 사용

`fixtures/predict_samples.json`으로 백엔드 없이 UI를 먼저 붙일 수 있습니다.
각 항목의 `response` 객체가 **실제 API 응답과 동일한 형태**입니다.
`gold`(정답·순위·정오답 여부)는 참고/정오답 뱃지 UI 판단용이며 **실제 응답에는 포함되지 않습니다**. 점수 필드를 “신뢰도”로 표시하지 말 것.

전문가 사용성 평가에는 동일 비문이 연속되지 않도록 고른 `fixtures/expert_review_panel.json`(및 `reports/expert_panel/expert_review_panel.csv`)을 사용한다. `predict_samples.json`의 342~351대 연속 장문은 **프론트 모킹 전용**이다.

- `artificial_mask`: 정답이 있어 Exact@k 참고 가능
- `real_defect`: 실제 `□` 결손 — `label_char` 없음, 전문가 후보 타당성 평가용

## 성능 & 한계 (전달 시 함께 고지)

| 지표 | 값 |
|------|-----|
| Exact@1 | 33.75% |
| Exact@5 | 53.04% |
| Exact@10 | 60.47% |
| test n | 9,719 (Colab GPU full, 1회) |

- **서빙 공식 후보는 EXP-012**. 파이프라인 연동·발표 비교용이며, EXP-013~017 이후에도 체크포인트는 교체되지 않았습니다.
- FT iteration 기준선은 EXP-015(full valid Exact@k; DECISIONS [014]로 신규 모델 비교의 공식 incumbent 고정). EXP-017(ZS/FT 보간)은 overall 개선 없음 → FT 유지. EXP-018은 GPU Phase B 018A·018B 학습(4epoch)까지 완료됐으나 Exact@k 공식 평가 전이라 EXP-015 대비 우열은 아직 미확정(018C는 결과 미확보; 학습 완료 ≠ Exact@k 평가 완료).
- `model_version`: `exp012_sillokbert_gpu_best_valid_loss` (valid_loss로 고른 EXP-012 best).
- era 층화상 조선(n=7,425)이 표본 대부분이고 Exact@1 35.87%. 가야·백제·신라 등은 표본 2~10건으로 **수치 신뢰 불가** → UI에서 시대별 성능을 노출한다면 "표본 부족" 표기 권장.
- 근거 데이터: `notebooks/best/evaluation_summary.csv`, `reports/exp012_results.md`, `reports/exp012_metrics_reverify.json`, `docs/EXPERIMENT_LOG.md` EXP-012·015~018, `docs/DECISIONS.md` [010]·[012]·[014].
