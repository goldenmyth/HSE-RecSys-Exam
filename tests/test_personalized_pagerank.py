"""Ручные вероятности PPR и свойства разреженного co-view графа."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.personalized_pagerank import build_graph, personalized_scores


class PersonalizedPageRankTests(unittest.TestCase):
    def test_two_node_closed_form_and_distinct_personalisation(self):
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        counts = sparse.csr_matrix([[1, 0], [0, 1]])
        restart = 0.2
        scores, audit = personalized_scores(transition, counts, [0, 1], [0, 1], restart=restart)
        expected = np.array([[1, 1 - restart], [1 - restart, 1]]) / (2 - restart)
        np.testing.assert_allclose(scores, expected, atol=1e-8)
        np.testing.assert_allclose(scores.sum(axis=1), [1, 1], atol=1e-12)
        self.assertEqual(scores.dtype, np.float64)
        self.assertLessEqual(audit["max_l1_residual"], 1e-8)
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertGreater(scores[1, 1], scores[1, 0])

    def test_isolates_return_mass_to_own_seed_and_unknowns_stay_nan(self):
        counts = sparse.csr_matrix([[1, 0], [0, 3], [0, 0]])
        transition, graph_audit = build_graph(counts, [11, 22])
        scores, audit = personalized_scores(transition, counts, [0, 1, 2, -1], [0, 1, -1], weighting="log_count")
        np.testing.assert_allclose(scores[:2, :2], np.eye(2))
        self.assertTrue(np.all(np.isnan(scores[:, 2])))
        self.assertTrue(np.all(np.isnan(scores[2:])))
        self.assertEqual(graph_audit["dangling_items"], 2)
        self.assertEqual(audit["unseeded_known_user_rows"], 1)
        self.assertEqual(audit["unknown_user_rows"], 1)
        self.assertEqual(audit["max_probability_mass_error"], 0)

    def test_topk_has_no_loops_and_breaks_ties_by_numeric_item_id(self):
        counts = sparse.csr_matrix([[2, 5, 9]])
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense graph prohibited")):
            transition, audit = build_graph(counts, [30, 10, 20], neighbors=1, block_size=2)
        self.assertEqual(transition.indices.tolist(), [1, 2, 1])
        np.testing.assert_array_equal(transition.diagonal(), [0, 0, 0])
        np.testing.assert_array_equal(np.diff(transition.indptr), [1, 1, 1])
        np.testing.assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), [1, 1, 1])
        self.assertEqual(audit["max_similarity_block_shape"], [2, 3])
        self.assertFalse(audit["full_dense_item_matrix_created"])

    def test_user_normalisation_downweights_long_histories(self):
        counts = sparse.csr_matrix([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        cosine, _ = build_graph(counts, [10, 20, 30], normalisation="cosine", neighbors=2)
        normalised, _ = build_graph(counts, [10, 20, 30], normalisation="user_cosine", neighbors=2)
        self.assertAlmostEqual(float(cosine[0, 2]), 1 / 3)
        similarity = (1 / 3) / np.sqrt((1 / 3 + 1 / 2) * (1 / 3 + 1))
        self.assertAlmostEqual(float(normalised[0, 2]), similarity / (1 + similarity))
        self.assertLess(float(normalised[0, 2]), float(cosine[0, 2]))

    def test_log_seed_and_batching_match_manual_solution(self):
        counts = sparse.csr_matrix([[1, 3], [0, 2], [2, 0]])
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        options = dict(restart=0.3, weighting="log_count", tol=1e-10)
        by_one, _ = personalized_scores(transition, counts, [0, 1, 2], [0, 1], batch_size=1, **options)
        by_two, _ = personalized_scores(transition, counts, [0, 1, 2], [0, 1], batch_size=2, **options)
        s0, s1 = 1 / 3, 2 / 3  # log(2) и log(4), нормированные на сумму.
        expected0 = (s0 + 0.7 * s1) / 1.7
        self.assertAlmostEqual(float(by_one[0, 0]), expected0, places=9)
        np.testing.assert_allclose(by_one, by_two, atol=1e-10)

    def test_nonconvergence_and_invalid_parameters_raise(self):
        counts = sparse.csr_matrix([[1, 0]])
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        with self.assertRaises(RuntimeError):
            personalized_scores(transition, counts, [0], [0, 1], restart=0.01, tol=1e-20, max_iter=1)
        for options in ({"restart": 0}, {"weighting": "duration"}, {"max_iter": 0}, {"tol": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                personalized_scores(transition, counts, [0], [0, 1], **options)
        for options in ({"neighbors": 0}, {"block_size": 65}, {"normalisation": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_graph(counts, [10, 20], **options)
        with self.assertRaises(ValueError):
            personalized_scores(sparse.csr_matrix([[0, 2], [1, 0]]), counts, [0], [0, 1])

    def test_external_seed_matches_closed_form_without_reapplying_log_or_binary(self):
        counts = sparse.csr_matrix([[2, 8]])
        weights = sparse.csr_matrix([[1, 9]], dtype=np.float64)
        original = weights.toarray().copy()
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        options = dict(restart=0.3, weighting="log_count", tol=1e-10)
        scores, audit = personalized_scores(
            transition, counts, [0], [0, 1], restart_weights=weights, **options
        )
        expected0 = (0.1 + 0.7 * 0.9) / 1.7
        np.testing.assert_allclose(scores, [[expected0, 1 - expected0]], atol=1e-10)
        scaled, _ = personalized_scores(
            transition, counts, [0], [0, 1], restart_weights=weights * 10, **options
        )
        np.testing.assert_allclose(scores, scaled, atol=1e-12)
        np.testing.assert_array_equal(weights.toarray(), original)
        self.assertEqual(audit["seed_source"], "external_restart_weights")
        self.assertIn("no binary/log transform", audit["seed_normalisation"])

    def test_default_seed_scores_and_audit_are_unchanged_with_explicit_none(self):
        counts = sparse.csr_matrix([[1, 3], [0, 2]])
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        options = dict(restart=0.3, weighting="log_count", tol=1e-10)
        default, audit = personalized_scores(transition, counts, [0, 1], [0, 1], **options)
        explicit, explicit_audit = personalized_scores(
            transition, counts, [0, 1], [0, 1], restart_weights=None, **options
        )
        np.testing.assert_array_equal(default, explicit)
        expected0 = (1 / 3 + 0.7 * 2 / 3) / 1.7
        self.assertAlmostEqual(float(default[0, 0]), expected0, places=9)
        audit.pop("elapsed_seconds")
        explicit_audit.pop("elapsed_seconds")
        self.assertEqual(audit, explicit_audit)
        self.assertNotIn("seed_source", audit)
        self.assertEqual(
            audit["seed_normalisation"], "own train binary/log1p counts divided by row sum"
        )

    def test_external_zero_seeds_and_unknown_users_remain_nan_for_fallback(self):
        counts = sparse.csr_matrix([[1, 1], [1, 0]])
        weights = sparse.csr_matrix([[0, 0], [2, 0]])
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        scores, audit = personalized_scores(
            transition, counts, [0, 1, -1], [0, 1, -1], restart_weights=weights
        )
        self.assertTrue(np.isnan(scores[[0, 2]]).all())
        self.assertTrue(np.isnan(scores[:, 2]).all())
        np.testing.assert_allclose(scores[1, :2], [1 / 1.85, 0.85 / 1.85], atol=1e-8)
        self.assertEqual(audit["unseeded_known_user_rows"], 1)
        self.assertEqual(audit["unknown_user_rows"], 1)
        zeros, zero_audit = personalized_scores(
            transition, counts, [0, 1], [0, 1], restart_weights=sparse.csr_matrix(counts.shape)
        )
        self.assertTrue(np.isnan(zeros).all())
        self.assertEqual(zero_audit["seeded_user_rows"], 0)
        self.assertEqual(zero_audit["unseeded_known_user_rows"], 2)

    def test_external_seed_rejects_invalid_weights_shape_and_nontrain_support(self):
        counts = sparse.csr_matrix([[1, 0], [0, 1]])
        transition = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        invalid = (
            sparse.csr_matrix([[-1, 0], [0, 1]]),
            sparse.csr_matrix([[np.nan, 0], [0, 1]]),
            sparse.csr_matrix([[np.inf, 0], [0, 1]]),
            sparse.csr_matrix([[1, 0]]),
            sparse.csr_matrix([[0, 1], [0, 1]]),
        )
        for weights in invalid:
            with self.subTest(weights=weights.toarray()), self.assertRaises(ValueError):
                personalized_scores(
                    transition, counts, [0, 1], [0, 1], restart_weights=weights
                )
        with self.assertRaises(TypeError):
            personalized_scores(transition, counts, [0], [0, 1], restart_weights=[[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
