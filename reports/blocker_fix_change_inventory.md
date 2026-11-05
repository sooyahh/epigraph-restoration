# Blocker Fix — Pre-existing Change Inventory

작성: 2026-07-11  
기준 브랜치: `review/exp011-integration`  
blocker 수정 작업 시작 HEAD: `30676471c57bbd28bcd279a1b3978aa158459be2`  
(직전 커밋: `3067647 docs: add pre-GPU readiness audit`)

## 조사 결과

감사 시작 시점(`5d4f782`)에는 아래 5파일이 **modified(unstaged)** 로 보고됐다.
blocker 수정 작업 시작 시점에는 working tree가 **clean**이며, 해당 5파일에 대한 `git diff`는 **비어 있다**.

이 작업에서는 사용자 지시에 따라 restore/stash/삭제를 시도하지 않았다.
현재 상태에서는 되살릴 unstaged diff가 없다.

| 파일 | 현재 git 상태 | 감사 시점 보고 | 감사 시점 diff 성격 (당시 기록) |
|---|---|---|---|
| `reports/exp007_gpu_status.md` | clean (tracked) | modified | 번호 체계 안내 배너 주변 공백/줄바꿈만 변경 (내용 의미 변화 없음) |
| `reports/gpu_eve_checklist.md` | clean (tracked) | modified | 동일 (whitespace-only) |
| `reports/local_gpu_runbook.md` | clean (tracked) | modified | 동일 (whitespace-only) |
| `reports/presentation_outline.md` | clean (tracked) | modified | 동일 (whitespace-only) |
| `reports/presentation_qa.md` | clean (tracked) | modified | 동일 (whitespace-only) |

`5d4f782..3067647` 사이 커밋은 `reports/pre_gpu_readiness_audit_2026-07-11.md` 추가만 포함하며, 위 5파일은 그 커밋에서 변경되지 않았다. 따라서 감사 시점 unstaged whitespace diff는 **커밋되지 않은 채 working tree에서 사라진 상태**로 보인다(외부에서 discard됐거나 동일 내용으로 정리됨).

## 현재 파일 내용 요약 (blocker 수정 전 HEAD 기준)

공통: 모두 노트북 legacy 번호 체계 안내 배너를 포함하며, **GPU full FT를 EXP-007**로 기술한다. 공식 로그에서 EXP-007은 SillokBERT zero-shot이다.

1. **`reports/local_gpu_runbook.md`** — `run_exp007_gpu.py` 일괄 실행, `--max-train 0`, 평가 ID `EXP-007`, 산출물 `sillokbert_gsm_baseline_gpu` / `exp007_*`. validation 선택 절차 없음.
2. **`reports/gpu_eve_checklist.md`** — 당일 작업을 EXP-007 FT+eval로 안내. EXP-011(CPU)과 혼동 가능한 legacy EXP-006 표기.
3. **`reports/exp007_gpu_status.md`** — EXP-007 pending PC bang, 로그에 EXP-007 metrics 추가 지시.
4. **`reports/presentation_outline.md`** — 발표용; subsample EXP-006(=공식 EXP-011) 수치와 GPU EXP-007 갱신 예정 서술.
5. **`reports/presentation_qa.md`** — Q&A에서 EXP-006=CPU FT, EXP-007=GPU full train으로 설명 (공식 ID와 불일치).

## 처리 방침

- 위 5파일의 **과거 whitespace diff는 복원하지 않음** (현재 없음).
- blocker 수정 과정에서 `local_gpu_runbook.md`, `gpu_eve_checklist.md`, `exp007_gpu_status.md`는 **예정 EXP-012** 절차로 **내용 갱신**한다 (기존 변경 삭제가 아니라 신규 수정).
- `presentation_outline.md`, `presentation_qa.md`는 이번 blocker 범위 밖이면 최소 고지만 하거나 건드리지 않는다. GPU 실행 안전과 직접 관련된 runbook/체크리스트/상태 문서를 우선한다.
