# Truncation spot-check (long training readings)

- Generated: 2026-07-02T21:10:55.210778+00:00
- Targets: `is_training_text=True`, `text_length>=5000`, no `detail_augment` note
- Samples: 9
- Truncated in XLS (detail longer): 0

## Split / eval policy note

- Quantitative eval uses `artificial_mask_dataset.csv` split column only.
- `splits.csv` may include defect-only inscriptions; era breakdowns should filter
  to inscriptions present in the mask dataset.

| reading_id | ksm_idx | title | xls_len | detail_len | delta | truncated |
| --- | --- | --- | ---: | ---: | ---: | --- |
| nrich_ksm_011051_x8384 | 11051 | 우복정경세신도비 | 7023 | 7023 | 0 | no |
| nrich_ksm_008299_x9354 | 8299 | 이경석신도비 | 7438 | 7438 | 0 | no |
| nrich_ksm_008550_x9755 | 8550 | 이산해신도비 | 5302 | 5302 | 0 | no |
| nrich_ksm_014741_x11679 | 14741 | 정온 신도비 | 5945 | 5945 | 0 | no |
| nrich_ksm_008793_x12030 | 8793 | 조경신도비 | 5496 | 5496 | 0 | no |
| nrich_ksm_013380_x12585 | 13380 | 진천 정철 신도비 | 6031 | 6031 | 0 | no |
| nrich_ksm_002201_x12985 | 2201 | 청주 최명길 신도비 | 6543 | 6543 | 0 | no |
| nrich_ksm_001302_x13754 | 1302 | 학봉신도비및묘방석 | 7227 | 7227 | 0 | no |
| nrich_ksm_008092_x14619 | 8092 | 홍처후신도비 | 6024 | 6024 | 0 | no |
