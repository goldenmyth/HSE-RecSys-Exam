"""Ручные проверки repeat scores и объединения списков без score mixing."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.fusion import native_agreement, ranked_lists, weighted_rrf
from src.personal_repeat import repeat_scores


class PersonalRepeatTests(unittest.TestCase):
    def matrices(self):
        counts = csr_matrix([[3, 3, 0], [0, 2, 1]], dtype=np.float64)
        last = csr_matrix([[864000, 691200, 0], [0, 864000, 777600]], dtype=np.float64)
        return counts, last

    def test_frequency_scores_keep_counts_and_numeric_ties(self):
        counts, last = self.matrices()
        scores = repeat_scores(counts, last, [0, 1], [0, 1, 2], 864000)
        self.assertEqual(scores.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(scores, [[3, 3, np.nan], [np.nan, 2, 1]], equal_nan=True)
        self.assertEqual(ranked_lists(scores, [20, 2, 99]), [[2, 20], [2, 99]])

    def test_recency_halves_score_after_one_half_life(self):
        counts, last = self.matrices()
        scores = repeat_scores(counts, last, [0], [0, 1, 2], 864000, half_life_days=2)
        np.testing.assert_allclose(scores, [[3, 1.5, np.nan]], equal_nan=True)

    def test_unknown_users_items_and_unseen_pairs_are_nan(self):
        counts, last = self.matrices()
        scores = repeat_scores(counts, last, [-1, 0], [2, -1, 0], 864000)
        np.testing.assert_allclose(
            scores, [[np.nan, np.nan, np.nan], [np.nan, np.nan, 3]], equal_nan=True
        )

    def test_future_last_watch_is_rejected_even_without_recency(self):
        counts, last = self.matrices()
        with self.assertRaises(ValueError):
            repeat_scores(counts, last, [0], [0], 863999)

    def test_invalid_counts_patterns_and_half_life_are_rejected(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(count=value):
                counts = csr_matrix(([value], ([0], [0])), shape=(1, 1))
                last = csr_matrix([[100]], dtype=np.float64)
                with self.assertRaises(ValueError):
                    repeat_scores(counts, last, [0], [0], 100)
        counts, last = self.matrices()
        for half_life in (0, -1, np.nan, np.inf):
            with self.subTest(half_life=half_life):
                with self.assertRaises(ValueError):
                    repeat_scores(counts, last, [0], [0], 864000, half_life_days=half_life)
        with self.assertRaises(ValueError):
            repeat_scores(counts, csr_matrix(counts.shape), [0], [0], 864000)


class FusionTests(unittest.TestCase):
    def test_native_agreement_never_pads_unknown_or_cold_scores(self):
        scores = np.array(
            [[np.nan, 4, np.nan, -2, np.inf], [np.nan, np.nan, np.nan, np.nan, np.nan]]
        )
        feature = native_agreement(scores, [50, 10, 20, 30, 40], top_n=20)
        np.testing.assert_array_equal(feature, [[0, 1, 0, 1, 0], [0, 0, 0, 0, 0]])
        self.assertEqual(feature.dtype, np.dtype(np.float64))

    def test_native_agreement_ranks_negative_als_scores_and_breaks_numeric_ties(self):
        scores = np.array([[-3, -1, -1, -5]])
        ids = [9, 30, 2, 1]
        np.testing.assert_array_equal(
            native_agreement(scores, ids, top_n=1), [[0, 0, 1, 0]]
        )
        np.testing.assert_array_equal(
            native_agreement(scores, ids, top_n=2), [[0, 1, 1, 0]]
        )

    def test_native_agreement_is_bounded_and_preserves_row_and_column_shapes(self):
        scores = np.array([[1, 2, 3, np.nan], [-4, 0, -np.inf, 9]])
        feature = native_agreement(scores, [10, 20, 30, 40], top_n=2)
        self.assertEqual(feature.shape, scores.shape)
        self.assertTrue(np.isfinite(feature).all())
        self.assertTrue(set(np.unique(feature)) <= {0, 1})
        np.testing.assert_array_equal(feature.sum(axis=1), [2, 2])
        np.testing.assert_array_equal(feature, [[0, 1, 1, 0], [0, 1, 0, 1]])

    def test_native_agreement_rejects_invalid_shapes_candidates_and_limits(self):
        for scores, ids, top_n in (
            ([1, 2], [1, 2], 20),
            ([[1, 2]], [1], 20),
            ([[1, 2]], [1, 1], 20),
            ([[1, 2]], [0, 1], 20),
            ([[1, 2]], [1, 2], 0),
            ([[1, 2]], [1, 2], 1.5),
        ):
            with self.subTest(scores=scores, ids=ids, top_n=top_n):
                with self.assertRaises(ValueError):
                    native_agreement(scores, ids, top_n=top_n)

    def test_ranked_lists_filters_unknowns_without_fallback_padding(self):
        scores = np.array([[3, np.nan, 3, np.inf], [np.nan, np.nan, np.nan, np.nan]])
        self.assertEqual(ranked_lists(scores, [10, 20, 2, 30], limit=100), [[2, 10], []])
        self.assertEqual(ranked_lists(scores, [10, 20, 2, 30], limit=1), [[2], []])

    def test_item_below_fifth_rank_can_win_rrf(self):
        components = {"first": [[1, 2, 3, 4, 5, 99]], "second": [[6, 7, 8, 9, 10, 99]]}
        predictions, used = weighted_rrf(
            components, {"first": 1, "second": 1}, list(range(1, 11)) + [99], [1, 2, 3, 4, 5]
        )
        self.assertEqual(predictions, [[99, 1, 6, 2, 7]])
        self.assertEqual(used, [False])

    def test_raw_score_scaling_cannot_change_rank_fusion(self):
        ids = [1, 2, 3, 4, 5, 6]
        first = np.array([[4, 3, np.nan, 2, 1, 0]])
        second = np.array([[1, 3, 4, np.nan, 2, 0]])
        weights = {"first": 0.2, "second": 0.8}
        original = weighted_rrf(
            {"first": ranked_lists(first, ids), "second": ranked_lists(second, ids)},
            weights, ids, [1, 2, 3, 4, 5],
        )
        rescaled = weighted_rrf(
            {"first": ranked_lists(first * 100 + 17, ids), "second": ranked_lists(second * 0.01 - 100, ids)},
            weights, ids, [1, 2, 3, 4, 5],
        )
        self.assertEqual(original, rescaled)

    def test_empty_personal_history_uses_same_fallback_once(self):
        fallback = [50, 40, 30, 20, 10]
        predictions, used = weighted_rrf(
            {"repeat": [[], [60]], "ppr": [[], []]},
            {"repeat": 1, "ppr": 1}, [10, 20, 30, 40, 50, 60], fallback,
        )
        self.assertEqual(predictions, [fallback, [60, 50, 40, 30, 20]])
        self.assertEqual(used, [True, True])

    def test_popularity_component_and_zero_weight_are_not_double_counted(self):
        predictions, used = weighted_rrf(
            {"pop": [[5, 4, 3, 2, 1, 6]], "repeat": [[6]]},
            {"pop": 1, "repeat": 0}, [1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1],
        )
        self.assertEqual(predictions, [[5, 4, 3, 2, 1]])
        self.assertEqual(used, [False])

    def test_invalid_component_inputs_are_rejected(self):
        ids, fallback = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]
        cases = [
            ({"one": [[]]}, {"missing": 1}),
            ({}, {}),
            ({"one": [[]], "two": [[], []]}, {"one": 1, "two": 1}),
            ({"one": [[1, 1]]}, {"one": 1}),
            ({"one": [[99]]}, {"one": 1}),
            ({"one": [[]]}, {"one": -1}),
            ({"one": [[]]}, {"one": np.nan}),
            ({"one": [[]]}, {"one": np.inf}),
        ]
        for components, weights in cases:
            with self.subTest(components=components, weights=weights):
                with self.assertRaises(ValueError):
                    weighted_rrf(components, weights, ids, fallback)
        with self.assertRaises(ValueError):
            weighted_rrf({"one": [[]]}, {"one": 1}, [], [])
        with self.assertRaises(ValueError):
            weighted_rrf({"one": [[]]}, {"one": 1}, ids, [1, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
