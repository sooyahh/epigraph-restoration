# EXP-012 산출물 무결성 기록 (SHA-256)

생성: 2026-08-01 · 목적: EXP-012(SillokBERT GPU full-data fine-tuning) 산출물의 **위치·크기·해시**를 git에 남겨,
로컬 사본이 손상·유실됐는지 판별하고 외부 백업본의 동일성을 검증할 수 있게 한다.

> 실제 파일은 `notebooks/best/`(`.gitignore` 대상)에만 있으며 **현재 이 PC 단일 사본**이다.
> 이 문서는 파일을 보존하지 않는다. 해시만 보존한다. 실제 보존은 아래 §3 외부 백업으로 수행해야 한다.
> 수치·설정은 `reports/exp012_results.md`, 기록은 `docs/EXPERIMENT_LOG.md` EXP-012, 채택은 `docs/DECISIONS.md` [010].
>
> **교훈 (epoch 보존):** Colab의 `checkpoint-epoch-2`는 세션 종료로 소실됐고 로컬/zip/원격에 없다.
> `epoch_metrics.json`에는 loss만 남아 Exact@k로 epoch를 재평가할 수 없다.
> 후속 FT는 `reports/ft_checkpoint_manifest_TEMPLATE.md`대로 **모든 `checkpoint-epoch-*`를 백업**한다.

## 1. 파일 목록 (원본 경로: `notebooks/best/`)

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `model.safetensors` | 711,921,812 | `f8f4be01c77c3b60cd2880f5f9698f0654820df62973e76ab4d54113950673c0` |
| `training_state.pt` | 1,423,968,831 | `13f250b37f7a984410232afcb59c9f2da0f3b0ab635a8a3ef64dc58877f275ca` |
| `exp012_sillok_ft_full_predictions.csv` | 8,685,693 | `0a206c2df39fd120378aa984f69e500d27f88852bf5ffcc77a18d1fe2a63c361` |
| `tokenizer.json` | 2,919,362 | `bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088` |
| `evaluation_summary.csv` | 4,246 | `243ee11ea88c273bc27b3a5f7dd011e352cb31b997ed33df549ff4916a30c124` |
| `finetune_metrics.json` | 1,788 | `c5da53fbe984e8a9238f9a6287888f885a6ebe67c2e9b92e7721674aa73807b1` |
| `best_checkpoint.json` | 1,165 | `8d395ff0f2d2491215298f63b8e824cee009d9cc07924f506042cdaeac3b09b8` |
| `config.json` | 909 | `25c01b53c8bfc03aafe9175977b4ddf8f7c74e0c068006d12642e6f2a136161b` |
| `epoch_metrics.json` | 564 | `7488f1dde0a7c793c8434ef09637ada6b15aaeec4be8e8f566e5cceeaa3d4922` |
| `tokenizer_config.json` | 393 | `d889e6872190d93d92c75c5bdc1ecba143e657a6d78bd617ce988a7428c7e0ad` |

해시 재계산:

```bash
python -c "import hashlib,pathlib;[print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name) for p in sorted(pathlib.Path('notebooks/best').iterdir()) if p.is_file()]"
```

`notebooks/best/`에서 tokenizer 로드가 정상 동작함을 확인했다 (`BertTokenizerFast`, vocab 119,547, mask `[MASK]`).
`special_tokens_map.json`·`vocab.txt`는 없으나 `tokenizer.json`만으로 로드된다.

## 2. 이미 git에 보존된 것 (유실돼도 무관)

- 공식 수치: `reports/exp012_results.md`, `reports/exp012_metrics_reverify.json`
- 시대별 층화: `reports/exp012_results.md` (원본 `evaluation_summary.csv` 요약)
- 빈도 층화: `reports/phase4_stratified_analysis_2026-08-01.md` / `.json`
- 재현 코드: `notebooks/exp012_colab.ipynb`, `work/run_exp012_gpu.py`, `src/data/finetune_sillokbert.py`
- 학습 데이터: `data/model_inputs/artificial_mask_dataset.csv`, `splits.csv`

## 3. 외부 백업 (미완료 — 담당자 조치 필요)

아래 2개만 올리면 유실 리스크가 해소된다. 나머지 경량 파일은 git에 있거나 재생성이 즉시 가능하다.

- [ ] `model.safetensors` (712MB) — 추론·데모·발표에 필요
- [ ] `training_state.pt` (1.42GB) — 학습 재개(`--resume-from`)에만 필요. 재개 계획이 없으면 생략 가능
- [ ] 업로드 후 이 문서에 **저장 위치·업로드일**을 기록하고, 내려받은 사본의 SHA-256이 §1과 일치하는지 확인

업로드 위치: `TODO:` (Google Drive 팀 폴더 등)
업로드일: `TODO:`

## 4. 재생성 가능성과 한계

모델을 잃어도 `notebooks/exp012_colab.ipynb`로 재학습이 가능하다 (Colab GPU, epochs=2, batch=16, lr=5e-5, seed=42,
train 46,139 / valid 10,679, `minimize_valid_loss`로 epoch 1 선택). 다만:

- GPU 연산 비결정성 때문에 **가중치가 bit-identical하게 재현되지는 않으며**, Exact@k가 소폭 달라질 수 있다.
- 따라서 재학습본은 EXP-012와 **동일 결과가 아니라 별도 run**으로 취급하고, 발표 수치를 바꿔야 한다.
- 즉 발표 전까지는 백업이 실질적으로 유일한 안전장치다.
