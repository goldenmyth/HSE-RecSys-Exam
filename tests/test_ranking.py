import unittest

import numpy as np

from src.ranking import rank_with_fallback, reciprocal_rank_fusion


class RankingTests(unittest.TestCase):
    def test_numeric_tie_and_fallback_without_repeats(self):
        ids = [10, 2, 3, 4, 5, 6]
        rows = np.array([[1, 1, np.nan, np.nan, np.nan, np.nan], [np.nan] * 6])
        predictions, fallback = rank_with_fallback(rows, ids, [2, 3, 4, 5, 6])
        self.assertEqual(predictions, [[2, 10, 3, 4, 5], [2, 3, 4, 5, 6]])
        self.assertEqual(fallback, [True, True])

    def test_seen_items_are_not_filtered(self):
        predictions, fallback = rank_with_fallback([[5, 4, 3, 2, 1]], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        self.assertEqual(predictions, [[1, 2, 3, 4, 5]])
        self.assertEqual(fallback, [False])

    def test_unknown_user_fusion_is_same_popularity(self):
        ids = [1, 2, 3, 4, 5, 6]
        pop = {6: 10, 5: 9, 4: 8, 3: 7, 2: 6}
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(weight=weight):
                fused = reciprocal_rank_fusion(np.full((1, 6), np.nan), ids, pop, weight)
                predictions, _ = rank_with_fallback(fused, ids, [6, 5, 4, 3, 2])
                self.assertEqual(predictions, [[6, 5, 4, 3, 2]])

    def test_fusion_uses_ranks_not_raw_score_scale(self):
        scores = np.array([[3, 2, 1, 0, -1]], dtype=float)
        a = reciprocal_rank_fusion(scores, [1, 2, 3, 4, 5], {5: 10}, 0.5)
        b = reciprocal_rank_fusion(scores * 100 + 7, [1, 2, 3, 4, 5], {5: 10}, 0.5)
        np.testing.assert_array_equal(a, b)

    def test_invalid_fallback_rejected(self):
        with self.assertRaises(ValueError):
            rank_with_fallback([[1] * 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 6])


if __name__ == "__main__":
    unittest.main()
