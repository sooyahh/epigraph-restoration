# 금석문 결손 한자 top-k 복원 보조 (Google × 아주대 AI 캡스톤)

> **공개용 사본 안내**
> 이 저장소는 캡스톤 조직 저장소(`google-ajou-Epigraph/sooyah`, 비공개)의 `main`(2026-10-02, `98eef68`)을 포트폴리오용으로 옮긴 것입니다. 이 저장소의 코드·문서·실험은 sooyahh가 Cursor 등 AI 코딩 도구와 함께 작성했습니다. 팀의 프론트엔드·백엔드 저장소는 포함하지 않습니다.
>
> - 판독문 원천은 국립문화유산연구원 한국금석문 종합영상정보시스템(NRICH)입니다. 데이터 재배포를 피하려고 `data/` 전체(원본 XLS·가공 corpus·마스크 데이터셋)와 판독문이 들어간 대량 예측 CSV는 넣지 않았습니다. 데모·전문가 패널용 소량 예시(수십 건)만 남겼습니다.
> - 모델 가중치와 커밋 이력은 옮기지 않았고, 문서 안의 로컬 절대 경로는 상대 경로로 바꿨습니다.
> - 아래 본문은 작업 당시 안내서를 그대로 둔 것입니다. 실행하려면 원천 데이터를 직접 받아 `data/`를 만들어야 합니다.

한국 금석문(비석·탁본) 판독문에서 **결손 한자(`□`)** 위치에 **top-k 후보**를 제시하는 프로젝트입니다.  
완전 자동 복원이 아니라, 전문가·검수자의 판독을 **보조**하는 것이 목표입니다.

모델은 결손 자리를 `"[MASK]"`로 두고 **text-only MLM(마스킹 언어모델)** 으로 한자를 예측합니다.

## 무엇을 하나요

- 입력: 결손(`□`) 또는 마스크가 표시된 금석문 판독문
- 출력: 각 마스크 위치의 한자 후보 top-k (+ 상대 점수)
- 정량 평가: 정답이 있는 **인공 single-mask** 데이터에서 `Exact@1 / Exact@5 / Exact@10`
- 실제 결손: gold가 없어 **정성 데모**로만 다룸 (인공 Exact@k와 동일시하지 않음)

## 현재 결과 (공식)

동일 **test n=9,719** 기준. 서빙·발표용 공식 비교 수치는 **EXP-012**입니다.

| 모델 | Exact@1 | Exact@5 | Exact@10 |
| --- | ---: | ---: | ---: |
| SillokBERT zero-shot (EXP-007) | 24.89% | 43.17% | 51.22% |
| **SillokBERT fine-tune (EXP-012)** | **33.75%** | **53.04%** | **60.47%** |

- 체크포인트: `exp012_sillokbert_gpu_best_valid_loss` (Git에 미포함)
- CPU subsample baseline(EXP-011, test n=2,000)과 혼동하지 말 것
- 이득은 고빈도 한자에 집중되며, hapax 구간은 zero-shot 대비 하락할 수 있음 (EXP-013/014)

## 레포 구조

| 경로 | 내용 |
| --- | --- |
| `src/` | 전처리 · 학습 · 평가 · 추론 |
| `data/` | 원본/가공 데이터 (대용량은 로컬·LFS) |
| `docs/` | 스키마 · 규칙 · 의사결정 · 실험 로그 |
| `tests/` | 단위 테스트 |
| `handoff/` | FE/BE 계약, FastAPI 스텁, 모킹 fixture |
| `reports/` | 실험 리포트 · 발표 골격 |
| `notebooks/` | Colab/실험 노트북 (가중치는 보통 로컬) |

## 빠른 시작

### 의존성 · 테스트

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
```

### FastAPI 추론 스텁 (`handoff/backend`)

```bash
cd handoff/backend
pip install -r requirements.txt

# 체크포인트 경로 (필수) — 예: 로컬 notebooks/best
export MODEL_DIR=../../notebooks/best   # Windows: set MODEL_DIR=..\..\notebooks\best

uvicorn app:app --host 0.0.0.0 --port 8000
```

- `POST /predict` — `text`에 `"[MASK]"` **정확히 1개**, `top_k` 기본 10  
  응답: `candidates[{ rank, char, score }]` (`score`는 상대 선호도, 보정된 신뢰확률이 아님)
- `GET /health`
- UI만 먼저 붙이려면 `handoff/fixtures/predict_samples.json`으로 모킹

모델 가중치는 용량 때문에 **이 저장소에 없습니다.** 별도 경로로 `MODEL_DIR`을 준비하세요.  
계약·응답 예시는 [`handoff/README.md`](handoff/README.md).

## 데이터 · 평가 원칙 (요약)

- 실제 결손 `□`와 한글 치환 `◇`를 구분 (`gsm_preprocess_v4`)
- split은 **inscription 단위** (`seed=42`) — 동일 비문이 train/test에 섞이지 않음
- 체크포인트·HP 선택은 validation; test는 최종 비교에 제한적으로 사용
- 인공 마스크 성능 ≠ 실제 훼손 복원 능력

상세: [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md), [`docs/DATA_PREPROCESSING_RULES.md`](docs/DATA_PREPROCESSING_RULES.md)

## 한계

- 인공 single-char 마스크만 정량 평가 (연속 결손 미포함)
- test–train n-gram overlap이 남아 정형구 효과가 성능에 섞일 수 있음
- tokenizer UNK 정답은 해당 vocab 모델이 직접 출력하기 어려움
- 전수 수동 검수 corpus는 아직 공식 채택 전
- 체크포인트·상세 예측 CSV는 Git 밖 로컬/Drive에 있어 백업이 필요함

## 더 읽기

- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — 목표 · 현재 상태 · 다음 작업
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — 실험 기록 (수치 근거)
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — 채택/보류 결정
- [`docs/RISK_CHECKLIST.md`](docs/RISK_CHECKLIST.md) — 위험·대응
- [`handoff/README.md`](handoff/README.md) — 서빙 계약 · API 예시
