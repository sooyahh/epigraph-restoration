"""GPU-free tests for ARI Appendix C constrained Top-K search."""

from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence

from src.eval.ari.topk import (
    COMPLETE_HANJA,
    INCOMPLETE,
    INVALID,
    HanjaCandidate,
    _Hypothesis,
    _as_logprob_map,
    _keep_better,
    _top_tokens,
    classify_decoded_span,
    expand_character_candidates,
    json_value_prefix_d1,
    order_candidates,
    score_mean_logprob,
)
from src.eval.metrics import compute_exact_metrics, rank_of_gold

TOK_LANG_U90CE = 1
TOK_LANG_U6717 = 2
TOK_TANG = 3
TOK_BYTE = 4
TOK_LANG_U90DE = 5
TOK_YAN = 6
TOK_LV = 7
TOK_QUOTE = 8

DECODE: dict[tuple[int, ...], str] = {
    (TOK_LANG_U90CE,): "郎",
    (TOK_LANG_U6717,): "朗",
    (TOK_TANG,): "堂",
    (TOK_BYTE,): "",
    (TOK_BYTE, TOK_LANG_U90DE): "郞",
    (TOK_BYTE, TOK_YAN): "郔",
    (TOK_BYTE, TOK_LV): "郘",
    (TOK_QUOTE,): '"',
}

STEP1_LOGPROBS = {
    TOK_BYTE: -0.002,
    TOK_LANG_U90CE: -6.877,
    TOK_LANG_U6717: -7.752,
    TOK_TANG: -9.065,
    TOK_QUOTE: -20.0,
}

STEP2_LOGPROBS = {
    TOK_LANG_U90DE: -1.0,
    TOK_YAN: -3.0,
    TOK_LV: -4.0,
}


def _decode(token_ids: Sequence[int]) -> str:
    return DECODE.get(tuple(token_ids), "�")


def _next_logprobs(prefix_ids: Sequence[int]) -> dict[int, float]:
    if len(prefix_ids) == 1 and prefix_ids[0] == 0:
        return dict(STEP1_LOGPROBS)
    if tuple(prefix_ids) == (0, TOK_BYTE):
        return dict(STEP2_LOGPROBS)
    return {}


def _expand_resetting_completed(
    prefix_ids: Sequence[int],
    *,
    k: int = 5,
    b1: int = 32,
    b2: int = 8,
    lmax: int = 4,
) -> list[HanjaCandidate]:
    """Buggy Appendix C loop: `completed = []` every step, no running merge."""
    from src.eval.ari.topk import classify_decoded_span, score_mean_logprob

    classify = classify_decoded_span
    mean_lp = score_mean_logprob
    frontier = [_Hypothesis(prefix_ids=tuple(prefix_ids), token_ids=(), logprobs=())]
    completed: dict[str, HanjaCandidate] = {}
    for step in range(1, lmax + 1):
        take = b1 if step == 1 else b2
        new_frontier: list[_Hypothesis] = []
        completed = {}
        for hyp in frontier:
            ranked = _top_tokens(_as_logprob_map(_next_logprobs(hyp.prefix_ids)), take)
            for token_id, token_lp in ranked:
                token_ids = hyp.token_ids + (token_id,)
                logprobs = hyp.logprobs + (token_lp,)
                decoded = _decode(token_ids)
                status = classify(decoded)
                if status == COMPLETE_HANJA:
                    char = decoded
                    cand = HanjaCandidate(
                        char=char,
                        score=mean_lp(logprobs),
                        token_ids=token_ids,
                        logprobs=logprobs,
                    )
                    completed[char] = _keep_better(completed.get(char), cand)
                elif status == INCOMPLETE:
                    new_frontier.append(
                        _Hypothesis(
                            prefix_ids=hyp.prefix_ids + (token_id,),
                            token_ids=token_ids,
                            logprobs=logprobs,
                        )
                    )
        frontier = new_frontier
        if len(completed) >= k and not frontier:
            break
    return order_candidates(list(completed.values()), k)


class ClassifyDecodedSpanTests(unittest.TestCase):
    def test_complete_hanja(self) -> None:
        self.assertEqual(classify_decoded_span("郎"), COMPLETE_HANJA)
        self.assertEqual(classify_decoded_span("郞"), COMPLETE_HANJA)

    def test_incomplete_empty_and_replacement(self) -> None:
        self.assertEqual(classify_decoded_span(""), INCOMPLETE)
        self.assertEqual(classify_decoded_span("�"), INCOMPLETE)
        self.assertEqual(classify_decoded_span("��"), INCOMPLETE)

    def test_invalid_non_hanja_and_multichar(self) -> None:
        self.assertEqual(classify_decoded_span('"'), INVALID)
        self.assertEqual(classify_decoded_span("{"), INVALID)
        self.assertEqual(classify_decoded_span("가"), INVALID)
        self.assertEqual(classify_decoded_span("郎朗"), INVALID)


class ScoringAndOrderTests(unittest.TestCase):
    def test_mean_logprob(self) -> None:
        self.assertAlmostEqual(score_mean_logprob([-6.877]), -6.877)
        self.assertAlmostEqual(score_mean_logprob([-0.002, -1.0]), -0.501)

    def test_json_prefix_matches_dumps(self) -> None:
        self.assertEqual(json_value_prefix_d1(), '{"[D1]": "')

    def test_order_and_rank_of_gold_contract(self) -> None:
        cands = [
            HanjaCandidate("字5", -2.4, (5,), (-2.4,)),
            HanjaCandidate("字1", -0.4, (1,), (-0.4,)),
            HanjaCandidate("字3", -1.5, (3,), (-1.5,)),
            HanjaCandidate("字2", -1.1, (2,), (-1.1,)),
            HanjaCandidate("字4", -2.0, (4,), (-2.0,)),
        ]
        ordered = [c.char for c in order_candidates(cands, k=5)]
        self.assertEqual(ordered, ["字1", "字2", "字3", "字4", "字5"])
        self.assertEqual(rank_of_gold(ordered, "字3"), 3)
        metrics = compute_exact_metrics(["字3"], [ordered])
        self.assertAlmostEqual(metrics.exact_at_1, 0.0)
        self.assertAlmostEqual(metrics.exact_at_5, 1.0)


class ExpandCharacterCandidatesTests(unittest.TestCase):
    def test_resetting_completed_drops_step1_single_token_hanja(self) -> None:
        last_step_only = _expand_resetting_completed([0], k=5, lmax=2)
        last_step_chars = [c.char for c in last_step_only]
        self.assertCountEqual(last_step_chars, ["郞", "郔", "郘"])
        self.assertNotIn("郎", last_step_chars)
        self.assertAlmostEqual(
            compute_exact_metrics(["郎"], [last_step_chars]).exact_at_5, 0.0
        )

        wiped_by_later_empty_steps = _expand_resetting_completed([0], k=5, lmax=4)
        self.assertEqual([c.char for c in wiped_by_later_empty_steps], [])

    def test_running_completed_keeps_fig12_one_token_candidates(self) -> None:
        result = expand_character_candidates(
            [0],
            _next_logprobs,
            _decode,
            k=5,
            b1=8,
            b2=8,
            lmax=4,
        )
        chars = [c.char for c in result]
        scores = {c.char: c.score for c in result}
        self.assertEqual(chars, ["郞", "郔", "郘", "郎", "朗"])
        self.assertAlmostEqual(scores["郞"], -0.501)
        self.assertAlmostEqual(scores["郎"], -6.877)
        self.assertEqual(len(set(chars)), 5)
        for ch in chars:
            self.assertEqual(len(ch), 1)
            self.assertEqual(classify_decoded_span(ch), COMPLETE_HANJA)
        self.assertEqual(rank_of_gold(chars, "郎"), 4)
        metrics = compute_exact_metrics(["郎"], [chars])
        self.assertAlmostEqual(metrics.exact_at_1, 0.0)
        self.assertAlmostEqual(metrics.exact_at_5, 1.0)

        with_sixth = expand_character_candidates(
            [0], _next_logprobs, _decode, k=6, b1=8, b2=8, lmax=4
        )
        self.assertEqual([c.char for c in with_sixth][-1], "堂")

        next_token_top5 = [
            tok
            for tok, _ in sorted(STEP1_LOGPROBS.items(), key=lambda kv: -kv[1])[:5]
        ]
        decoded_next = [_decode((tok,)) for tok in next_token_top5]
        self.assertNotIn("郞", decoded_next)

    def test_duplicate_paths_keep_max_score(self) -> None:
        def decode(ids: Sequence[int]) -> str:
            if tuple(ids) in {(1,), (2, 3)}:
                return "郎"
            if tuple(ids) == (2,):
                return ""
            return '"'

        def logprobs(prefix: Sequence[int]) -> Mapping[int, float]:
            if list(prefix) == [0]:
                return {1: -2.0, 2: -0.5}
            if list(prefix) == [0, 2]:
                return {3: -0.5}
            return {}

        result = expand_character_candidates(
            [0], logprobs, decode, k=5, b1=4, b2=4, lmax=4
        )
        self.assertEqual([c.char for c in result], ["郎"])
        self.assertAlmostEqual(result[0].score, -0.5)
        self.assertEqual(result[0].token_ids, (2, 3))

    def test_empty_when_only_invalid(self) -> None:
        def decode(_ids: Sequence[int]) -> str:
            return '"'

        def logprobs(_prefix: Sequence[int]) -> Mapping[int, float]:
            return {9: -0.1}

        result = expand_character_candidates(
            [0], logprobs, decode, k=5, b1=4, b2=4, lmax=2
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
