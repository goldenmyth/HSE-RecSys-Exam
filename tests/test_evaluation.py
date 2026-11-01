"""Ручные проверки метрик и popularity без исходных данных соревнования."""

import csv
import importlib
import tempfile
import unittest
from pathlib import Path

from src.metrics import average_precision_at_k, mean_average_precision_at_k


class MetricTests(unittest.TestCase):
    def test_empty_truth_or_predictions(self):
        self.assertEqual(average_precision_at_k([], [1, 2, 3]), 0.0)
        self.assertEqual(average_precision_at_k({1, 2}, []), 0.0)
        self.assertEqual(average_precision_at_k([], []), 0.0)

    def test_perfect_lists_use_minimum_of_k_and_truth_size(self):
        for size in (1, 2, 5, 6):
            with self.subTest(size=size):
                relevant = set(range(1, size + 1))
                self.assertEqual(
                    average_precision_at_k(relevant, list(range(1, 6))), 1.0
                )

    def test_one_relevant_item_at_third_rank(self):
        self.assertAlmostEqual(average_precision_at_k({7}, [1, 2, 7]), 1 / 3)

    def test_two_hits_at_first_and_third_rank(self):
        self.assertAlmostEqual(
            average_precision_at_k({1, 2}, [1, 3, 2]), (1 + 2 / 3) / 2
        )

    def test_duplicate_prediction_occupies_rank_without_extra_hit(self):
        self.assertAlmostEqual(
            average_precision_at_k({1, 2}, [1, 1, 2]), 5 / 6
        )
        self.assertEqual(average_precision_at_k({1, 2}, [1, 1, 1]), 0.5)

    def test_duplicate_truth_does_not_inflate_denominator(self):
        self.assertEqual(average_precision_at_k([1, 1, 2], [1]), 0.5)

    def test_only_first_k_ranks_are_evaluated(self):
        self.assertEqual(average_precision_at_k({1}, [2, 3, 4, 5, 6, 1]), 0.0)
        self.assertEqual(average_precision_at_k({1, 2}, [1, 2], k=1), 1.0)

    def test_unavailable_relevant_item_remains_in_denominator(self):
        self.assertEqual(
            average_precision_at_k({1, 2}, [1, 3, 4, 5, 6]), 0.5
        )
        self.assertEqual(
            average_precision_at_k({7, 8}, [1, 3, 4, 5, 6]), 0.0
        )

    def test_iterable_inputs(self):
        self.assertAlmostEqual(
            average_precision_at_k(iter([1, 2]), iter([1, 3, 2])), 5 / 6
        )

    def test_nonpositive_k_raises_even_for_empty_inputs(self):
        for k in (0, -1):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    average_precision_at_k([], [], k=k)
                with self.assertRaises(ValueError):
                    mean_average_precision_at_k({}, {}, [], k=k)

    def test_empty_population_is_zero(self):
        self.assertEqual(mean_average_precision_at_k({}, {}, []), 0.0)

    def test_explicit_population_includes_missing_truth_and_predictions(self):
        relevant = {"9007199254740993": {1}, "no_predictions": {2}, "outside": {9}}
        predictions = {"9007199254740993": [1], "no_truth": [3], "outside": [9]}
        population = ["9007199254740993", "no_truth", "no_predictions"]
        self.assertAlmostEqual(
            mean_average_precision_at_k(relevant, predictions, population), 1 / 3
        )

    def test_duplicate_population_user_is_rejected(self):
        with self.assertRaises(ValueError):
            mean_average_precision_at_k({"user": {1}}, {"user": [1]}, ["user", "user"])


class PopularityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.popularity = importlib.import_module("src.popularity")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sweet-tv-evaluation-")
        self.addCleanup(self.temporary.cleanup)
        self.train_path = Path(self.temporary.name) / "train.csv"

    def write_train(self, rows):
        with self.train_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"]
            )
            for row in rows:
                writer.writerow(row)
        return self.train_path

    def test_fit_all_variants_and_preserve_long_user_ids(self):
        first, second = "3398461054087191302", "3398461054087191303"
        path = self.write_train(
            [
                (first, 10, 2, 120, "2020-05-01 12:00:00", "train"),
                (first, 20, 1, 180, "2020-05-01 12:00:00", "train"),
                (second, 10, 4, 300, "2020-05-01 12:00:00", "train"),
                (second, 30, 3, 60, "2020-05-01 12:00:00", "train"),
            ]
        )
        scores, activity = self.popularity.fit_popularity(path)
        self.assertEqual(
            self.popularity.VARIANTS, ("unique_users", "watch_count", "watch_seconds")
        )
        self.assertEqual(dict(scores["unique_users"]), {10: 2, 20: 1, 30: 1})
        self.assertEqual(dict(scores["watch_count"]), {10: 6, 20: 1, 30: 3})
        self.assertEqual(dict(scores["watch_seconds"]), {10: 420, 20: 180, 30: 60})
        self.assertEqual(set(activity), {first, second})

    def test_fit_rejects_validation_rows(self):
        path = self.write_train(
            [("u", 10, 1, 60, "2020-05-18 12:00:00", "validation")]
        )
        with self.assertRaises(ValueError):
            self.popularity.fit_popularity(path)

    def test_fit_rejects_repeated_pair_within_user_group(self):
        path = self.write_train(
            [
                ("u", 10, 1, 60, "2020-05-01 12:00:00", "train"),
                ("u", 20, 1, 60, "2020-05-01 12:00:00", "train"),
                ("u", 10, 2, 120, "2020-05-01 12:00:00", "train"),
            ]
        )
        with self.assertRaises(ValueError):
            self.popularity.fit_popularity(path)

    def test_fit_rejects_repeated_pair_across_user_groups(self):
        path = self.write_train(
            [
                ("u", 10, 1, 60, "2020-05-01 12:00:00", "train"),
                ("v", 20, 1, 60, "2020-05-01 12:00:00", "train"),
                ("u", 10, 2, 120, "2020-05-01 12:00:00", "train"),
            ]
        )
        with self.assertRaises(ValueError):
            self.popularity.fit_popularity(path)

    def test_top_five_uses_allowed_candidates_and_numeric_ties(self):
        scores = {10: 4, 2: 4, 99: 100}
        candidates = {1, 2, 10, 20, 30, 40}
        self.assertEqual(
            self.popularity.top_popular(scores, candidates), [2, 10, 1, 20, 30]
        )

    def test_zero_score_fallback_still_has_five_distinct_candidates(self):
        result = self.popularity.top_popular({}, [10, 2, 5, 1, 20, 10])
        self.assertEqual(result, [1, 2, 5, 10, 20])
        self.assertEqual(len(set(result)), 5)

    def test_too_few_distinct_candidates_is_rejected(self):
        for candidates in ([1, 2, 3, 4], [1, 1, 2, 3, 4]):
            with self.subTest(candidates=candidates):
                with self.assertRaises(ValueError):
                    self.popularity.top_popular({}, candidates)

    def test_nonpositive_candidate_id_is_rejected(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.popularity.top_popular({}, [1, 2, 3, 4, 5, invalid])


if __name__ == "__main__":
    unittest.main()
