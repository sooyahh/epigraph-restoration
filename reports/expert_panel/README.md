# Expert review panel

EXP-012 전문가 사용성 평가용 혼합 패널.

- `expert_review_panel.csv` — artificial_mask 25 + real_defect 25
- 비문당 최대 2건, mask/defect 위치 간격, era·점수 고/저·성공/실패 혼합
- JSON 미러: `handoff/fixtures/expert_review_panel.json`
- `model_version`: `exp012_sillokbert_gpu_best_valid_loss`
- 점수는 상대 선호 점수 (신뢰 확률 아님)

재생성: `python -m src.eval.select_expert_panel`
