"""Train-only context и ограниченные календарные признаки на ручных примерах."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import SCHEDULE_COLUMNS, wall_seconds
from src.history_reranking import bounded_rerank, load_train_context, restart_weights, schedule_affinity, schedule_context


class HistoryRerankingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sweet-tv-context-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "airings.csv"
        cutoff = wall_seconds("2020-05-18 00:00:00")
        self.history = {"users": {"9007199254740993": 0}, "items": np.array([11, 22]),
            "counts": sparse.csr_matrix([[4, 1]], dtype=np.float64),
            "last": sparse.csr_matrix([[cutoff - 30 * 86400, cutoff]], dtype=np.float64),
            "cutoff": cutoff, "lower": "2020-03-09 00:00:00", "period": "train"}

    def events(self):
        return [
            ["9007199254740993", "11", "5", "2020-03-30 09:00:00", "2020-03-30 10:00:00", "2020-03-30 09:05:00", "2020-03-30 10:00:00", "train"],
            ["9007199254740993", "11", "5", "2020-03-30 12:00:00", "2020-03-30 13:00:00", "2020-03-30 12:05:00", "2020-03-30 13:00:00", "train"],
            ["9007199254740993", "11", "5", "2020-03-31 09:00:00", "2020-03-31 10:00:00", "2020-03-31 09:05:00", "2020-03-31 10:00:00", "train"],
            ["9007199254740993", "22", "6", "2020-03-31 15:00:00", "2020-03-31 16:00:00", "2020-03-31 15:05:00", "2020-03-31 16:00:00", "train"],
        ]

    def write_events(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["user_id", "tv_show_id", "channel_id", "airing_start", "airing_end", "watch_start", "watch_end", "period"])
            writer.writerows(rows)

    def test_context_counts_distinct_calendar_days_and_unmodified_string_ids(self):
        self.write_events(self.events())
        context = load_train_context(self.path, self.history)
        np.testing.assert_array_equal(context["days"].toarray(), [[2, 1]])
        self.assertEqual(context["channel"][0], {5: 3, 6: 1})
        self.assertEqual(context["weekday"][0], {0: 2, 1: 2})
        self.assertEqual(context["hour"][0], {3: 2, 4: 1, 5: 1})
        self.assertEqual(context["joint"][0][(5, 0, 3)], 1)
        self.assertEqual(context["users"], {"9007199254740993": 0})
        expected = [2 ** (-(self.history["cutoff"] - wall_seconds(row[6])) / (30 * 86400)) for row in self.events()]
        np.testing.assert_allclose(context["recency_30"].toarray(), [[sum(expected[:3]) / 3, expected[3]]])

    def test_future_or_validation_event_and_duplicate_airing_are_rejected(self):
        rows = self.events()
        rows[0][-1] = "validation"
        self.write_events(rows)
        with self.assertRaises(ValueError):
            load_train_context(self.path, self.history)
        self.write_events(self.events() + [self.events()[0]])
        with self.assertRaises(ValueError):
            load_train_context(self.path, self.history)
        rows = self.events()
        rows[0][5:7] = ["2020-05-18 00:00:00", "2020-05-18 00:40:00"]
        self.write_events(rows)
        with self.assertRaises(ValueError):
            load_train_context(self.path, self.history)

    def test_ready_seed_weights_match_log_recency_and_regularity_formulas(self):
        context = {"days": sparse.csr_matrix([[2, 1]], dtype=np.float64),
                   "recency_30": sparse.csr_matrix([[0.25, 0.75]], dtype=np.float64)}
        expected = np.log1p([4, 1])
        np.testing.assert_allclose(restart_weights(self.history, context).toarray(), [expected])
        np.testing.assert_allclose(restart_weights(self.history, context, "recency").toarray(), [expected * [0.25, 0.75]])
        np.testing.assert_allclose(restart_weights(self.history, context, "regularity").toarray(), [np.sqrt(expected * np.log1p([2, 1]))])
        np.testing.assert_allclose(self.history["counts"].toarray(), [[4, 1]])
        with self.assertRaises(ValueError):
            restart_weights(self.history, context, "recency", half_life_days=7)

    def test_forecast_canonicalisation_excludes_conflict_and_uses_calendar_hours(self):
        rows = [
            {"channel_id": "5", "tv_show_id": "11", "start_time": "18.05.2020 09:00:00", "duration": "3600"},
            {"channel_id": "5", "tv_show_id": "11", "start_time": "18.05.2020 09:00:00", "duration": "3600"},
            {"channel_id": "5", "tv_show_id": "22", "start_time": "18.05.2020 12:00:00", "duration": "3600"},
            {"channel_id": "5", "tv_show_id": "0", "start_time": "18.05.2020 12:20:00", "duration": "600"},
            {"channel_id": "6", "tv_show_id": "33", "start_time": "17.05.2020 23:50:00", "duration": "3600"},
        ]
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEDULE_COLUMNS)
            writer.writeheader()
            for values in rows:
                writer.writerow({**dict.fromkeys(SCHEDULE_COLUMNS, ""), **values})
        schedule = schedule_context([self.path], "2020-05-18 00:00:00", "2020-05-19 00:00:00", [11, 22, 33])
        self.assertEqual(schedule["items"][11]["airings"], 1)
        self.assertEqual(schedule["items"][11]["joint"], {(5, 0, 3): 1})
        self.assertNotIn(22, schedule["items"])
        self.assertEqual(schedule["items"][33]["joint"], {(6, 0, 0): 1})
        self.assertEqual(schedule["audit"]["candidate_count"], 3)
        self.assertEqual(schedule["audit"]["excluded_partly_ambiguous_positive_airings"], 1)

    def test_affinities_are_bounded_forecast_exposure_means_and_neutral_unknowns(self):
        self.write_events(self.events())
        context = load_train_context(self.path, self.history)
        schedule = {"items": {
            11: {"airings": 2, "channel": {5: 1, 6: 1}, "joint": {(5, 0, 3): 2}},
            22: {"airings": 1, "channel": {6: 1}, "joint": {(6, 1, 5): 1}},
        }}
        users, candidates = ["9007199254740993", "unknown"], [11, 22, 33]
        channel = schedule_affinity(context, schedule, users, candidates, "channel")
        np.testing.assert_allclose(channel[0], [2 / 3, 1 / 3, 0])
        np.testing.assert_array_equal(channel[1], [0, 0, 0])
        joint = schedule_affinity(context, schedule, users, candidates, "joint")
        np.testing.assert_allclose(joint[0], [1, 1, 0])
        availability = schedule_affinity(context, schedule, users, candidates, "availability")
        np.testing.assert_allclose(availability[0], [1, np.log(2) / np.log(3), 0])
        self.assertTrue(np.all((availability >= 0) & (availability <= 1)))

    def test_bounded_rerank_is_scale_invariant_and_never_adds_outside_native_topk(self):
        base = np.array([[1.0, 0.95, 0.9], [np.nan, np.nan, np.nan]])
        feature = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        result = bounded_rerank(base, feature, [30, 10, 20], limit=2)
        np.testing.assert_allclose(result[0, :2], [1, 1.1875])
        self.assertTrue(np.isnan(result[0, 2]))
        self.assertTrue(np.all(np.isnan(result[1])))
        np.testing.assert_allclose(bounded_rerank(base * 8, feature, [30, 10, 20], limit=2), result * 8, equal_nan=True)
        with self.assertRaises(ValueError):
            bounded_rerank(base, feature * 2, [30, 10, 20])


if __name__ == "__main__":
    unittest.main()
