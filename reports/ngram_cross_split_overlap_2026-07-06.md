# Cross-split n-gram context overlap audit

- **생성일:** 2026-07-06
- **평가 split:** test
- **train samples:** 46,139
- **eval samples:** 9,719
- **n-gram sizes:** [8, 4, 2, 1]
- **context window:** 64

## 요약

- test 샘플 중 train과 **동일 좌/우 n-gram 키(길이≥2)** 를 가진 비율: **5,787 / 9,719 (59.5%)**
- test 샘플 중 train과 **어떤 n-gram 키든(길이≥1)** 매칭: 9,682 (99.6%) — unigram(1-gram) 포함 시 거의 전부
- left context match: 9,379
- right context match: 9,410
- 샘플당 평균 매칭 키 수: 2.82

## n-gram 크기별 (test 샘플 기준)

| n | matched samples | fraction |
|--:|----------------:|---------:|
| 8 | 176 | 1.8% |
| 4 | 789 | 8.1% |
| 2 | 5,787 | 59.5% |
| 1 | 9,682 | 99.6% |

## 해석

- inscription 단위 split은 **동일 비문** 누수를 막지만, **정형구·반복 문맥**은 train/test에 걸쳐 n-gram 키가 겹칠 수 있다.
- ngram baseline(EXP-004)은 train에서 본 context key를 lookup하므로, 위 비율이 높으면 test 성능 일부는 **memorization**일 수 있다.
- BERT/SillokBert zero-shot은 lookup을 쓰지 않지만, 동일 문맥 반복은 여전히 쉬운 케이스를 늘릴 수 있다 → 층화 분석과 함께 해석한다.
