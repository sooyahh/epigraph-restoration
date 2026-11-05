# Fine-tune checkpoint MANIFEST template

후속 FT 실험마다 `reports/<exp_id>_artifacts/MANIFEST.md`를 이 템플릿으로 만든다.
목적은 로컬/Drive 사본의 **위치·크기·SHA-256**을 git에 남겨 유실·혼동을 막는 것이다.

> **필수:** `best/`만 기록하지 않는다. `checkpoint-epoch-*` **전부**를 백업·해시한다.
> 학습 중 alias `best/`는 valid_loss 기준일 수 있다. 공식 epoch 채택은 post-hoc **valid Exact@5** (동점이면 valid_loss 최소·낮은 epoch) — DECISIONS 평가 프로토콜.

## 1. 메타

| 항목 | 값 |
|---|---|
| experiment_id | EXP-0XX |
| generated_at | YYYY-MM-DD |
| output_dir (원 경로) | |
| local_backup_dir | |
| external_backup | TODO / 위치·업로드일 |

## 2. 공통 메타 JSON

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `best_checkpoint.json` | | |
| `epoch_metrics.json` | | |
| `finetune_metrics.json` | | |

## 3. Epoch별 체크포인트 (전부)

각 `checkpoint-epoch-N/`에 대해:

| 경로 | 파일 | 크기 | SHA-256 |
|---|---|---:|---|
| `checkpoint-epoch-1/` | `model.safetensors` (또는 `pytorch_model.bin`) | | |
| `checkpoint-epoch-1/` | `training_state.pt` | | |
| `checkpoint-epoch-1/` | `config.json` | | |
| `checkpoint-epoch-2/` | … | | |

## 4. Loss alias `best/` (선택적으로 epoch와 동일)

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `best/model.safetensors` | | |
| `best/training_state.pt` | | |

## 5. 채택 기록

| 항목 | 값 |
|---|---|
| in_training_best_epoch (valid_loss) | |
| post_hoc_adopted_epoch (valid Exact@5) | |
| valid Exact@1 / @5 / @10 (adopted) | |
| tie-break rule | valid_loss min, then lower epoch |

## 6. 해시 재계산 예

```bash
python -c "import hashlib,pathlib; root=pathlib.Path('PATH_TO_BACKUP');
[print(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(root))
 for p in sorted(root.rglob('*')) if p.is_file()]"
```
