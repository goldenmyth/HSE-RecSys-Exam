"""Небольшие проверки confidence, ориентации и воспроизводимости ALS."""

import unittest
from types import SimpleNamespace

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares as CPUAlternatingLeastSquares
from scipy.sparse import csr_matrix, isspmatrix_csr
from threadpoolctl import threadpool_limits

from src.matrix_factorization import confidence_matrix, fit_als, score_als


class MatrixFactorizationTests(unittest.TestCase):
    def counts(self):
        return csr_matrix([[1, 0, 4], [0, 2, 0]], dtype=np.float32)

    def tiny_counts(self):
        return csr_matrix(
            [[1, 2, 0, 0, 1], [0, 1, 3, 0, 0], [1, 0, 0, 2, 1]],
            dtype=np.float32,
        )

    def config(self):
        return {
            "factors": 3,
            "regularization": 0.1,
            "weighting": "log_count",
            "confidence_alpha": 10,
            "iterations": 20,
        }

    def test_binary_confidence_preserves_sparse_unseen_and_input(self):
        counts = self.counts()
        original = counts.toarray().copy()
        confidence = confidence_matrix(counts, weighting="binary", alpha=10)
        self.assertTrue(isspmatrix_csr(confidence))
        self.assertEqual(confidence.dtype, np.dtype(np.float32))
        self.assertEqual(confidence.shape, (2, 3))
        self.assertEqual(confidence.nnz, 3)
        np.testing.assert_array_equal(confidence.toarray(), [[11, 0, 11], [0, 11, 0]])
        np.testing.assert_array_equal(counts.toarray(), original)

    def test_log_count_confidence_uses_log1p_only_for_observed_entries(self):
        confidence = confidence_matrix(self.counts(), weighting="log_count", alpha=2)
        expected = [[1 + 2 * np.log(2), 0, 1 + 2 * np.log(5)], [0, 1 + 2 * np.log(3), 0]]
        self.assertEqual(confidence.dtype, np.dtype(np.float32))
        self.assertEqual(confidence.nnz, 3)
        np.testing.assert_allclose(confidence.toarray(), expected, rtol=1e-6)

    def test_invalid_stored_counts_are_rejected(self):
        for invalid in (0, -1, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                explicit = csr_matrix(([invalid], ([0], [0])), shape=(1, 2))
                with self.assertRaises(ValueError):
                    confidence_matrix(explicit)

    def test_real_tiny_fit_has_user_item_orientation_and_cpu_settings(self):
        with threadpool_limits(limits=1):
            model = fit_als(self.tiny_counts(), self.config())
        self.assertIsInstance(model, CPUAlternatingLeastSquares)
        self.assertEqual(model.user_factors.shape, (3, 3))
        self.assertEqual(model.item_factors.shape, (5, 3))
        self.assertEqual(model.user_factors.dtype, np.dtype(np.float32))
        self.assertEqual(model.item_factors.dtype, np.dtype(np.float32))
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.num_threads, 1)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.iterations, 20)
        scores = score_als(model, [0, 1, 2], [0, 1, 2, 3, 4])
        self.assertEqual(scores.shape, (3, 5))
        self.assertTrue(np.isfinite(scores).all())

    def test_manual_factor_products_and_unknown_nan(self):
        model = SimpleNamespace(
            user_factors=np.array([[1, 2], [3, 4]], dtype=np.float32),
            item_factors=np.array([[1, 0], [0, 1], [-1, 2]], dtype=np.float32),
        )
        scores = score_als(model, [1, -1, 0], [2, -1, 0])
        self.assertEqual(scores.dtype, np.dtype(np.float32))
        expected = [[5, np.nan, 3], [np.nan, np.nan, np.nan], [3, np.nan, 1]]
        np.testing.assert_allclose(scores, expected, equal_nan=True)

    def test_two_seeded_fits_have_equal_scores_and_ranking(self):
        with threadpool_limits(limits=1):
            first = fit_als(self.tiny_counts(), self.config(), seed=42)
            second = fit_als(self.tiny_counts(), self.config(), seed=42)
            first_scores = score_als(first, [0, 1, 2], [0, 1, 2, 3, 4])
            second_scores = score_als(second, [0, 1, 2], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(first.user_factors, second.user_factors)
        np.testing.assert_allclose(first.item_factors, second.item_factors)
        np.testing.assert_allclose(first_scores, second_scores)
        np.testing.assert_array_equal(
            np.argsort(-first_scores, axis=1, kind="stable"),
            np.argsort(-second_scores, axis=1, kind="stable"),
        )


if __name__ == "__main__":
    unittest.main()
