"""Строгие проценты за один непрерывный кусок одной исходной сессии."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_interactions import build_channel_index, wall_seconds
from src.session_interactions import match_sessions


class SessionInteractionTests(unittest.TestCase):
    def match(self, views, schedules, periods=(("train", 0, 1000),), exclude_cross_channel=True):
        indices = {channel: build_channel_index(records)[0] for channel, records in schedules.items()}
        return match_sessions(views, indices, periods, exclude_cross_channel=exclude_cross_channel)

    def test_80_percent_is_strict_and_81_percent_qualifies(self):
        at, _ = self.match({1: [(0, 80)]}, {1: [(0, 100, 11)]})
        over, _ = self.match({1: [(0, 81)]}, {1: [(0, 100, 11)]})
        self.assertEqual(at["train"][80], {})
        self.assertEqual(over["train"][80], {11: (1, 81, 81)})

    def test_50_percent_is_strict(self):
        at, _ = self.match({1: [(0, 50)]}, {1: [(0, 100, 11)]})
        over, _ = self.match({1: [(0, 51)]}, {1: [(0, 100, 11)]})
        self.assertEqual(at["train"][50], {})
        self.assertEqual(over["train"][50], {11: (1, 51, 51)})
        self.assertEqual(over["train"][80], {})

    def test_separate_and_touching_original_rows_are_not_accumulated(self):
        for intervals in ([(0, 45), (55, 100)], [(0, 50), (50, 100)]):
            with self.subTest(intervals=intervals):
                result, _ = self.match({1: intervals}, {1: [(0, 100, 11)]})
                self.assertEqual(result, {"train": {50: {}, 80: {}}})

    def test_duplicate_and_overlapping_qualifying_rows_count_airing_once(self):
        result, audit = self.match({1: [(0, 90), (5, 95), (0, 90)]}, {1: [(0, 100, 11)]})
        self.assertEqual(result["train"][80], {11: (1, 95, 95)})
        self.assertEqual(audit["train"]["exact_duplicate_session_rows"], 1)
        self.assertEqual(audit["train"]["deduplicated_session_rows"], 2)
        self.assertEqual(audit["train"]["qualifying_pieces_80"], 2)
        self.assertEqual(audit["train"]["qualified_user_airings_80"], 1)

    def test_adjacent_airings_of_same_item_remain_two_watches(self):
        result, _ = self.match({1: [(0, 200)]}, {1: [(0, 100, 11), (100, 200, 11)]})
        self.assertEqual(result["train"][80], {11: (2, 200, 200)})

    def test_unknown_other_channel_splits_original_row_without_summing(self):
        result, audit = self.match({1: [(0, 100)], 2: [(40, 60)]}, {1: [(0, 100, 11)]})
        self.assertEqual(result, {"train": {50: {}, 80: {}}})
        self.assertEqual(audit["train"]["cross_channel_ambiguous_user_seconds"], 20)
        self.assertEqual(audit["train"]["row_weighted_cross_channel_ambiguous_seconds"], 40)
        self.assertEqual(audit["train"]["sessions_split_by_cross_channel_overlap"], 1)
        self.assertEqual(audit["train"]["cross_channel_mask_evaluated_users"], 1)
        sensitive, sensitivity_audit = self.match({1: [(0, 100)], 2: [(40, 60)]}, {1: [(0, 100, 11)]}, exclude_cross_channel=False)
        self.assertEqual(sensitive["train"][80], {11: (1, 100, 100)})
        self.assertEqual(sensitivity_audit["train"]["cross_channel_mask_evaluated_users"], 0)
        touching, touching_audit = self.match({1: [(0, 100)], 2: [(100, 200)]}, {1: [(0, 100, 11)]})
        self.assertEqual(touching["train"][80], {11: (1, 100, 100)})
        self.assertEqual(touching_audit["train"]["cross_channel_ambiguous_user_seconds"], 0)

    def test_trimmed_continuous_piece_can_still_qualify(self):
        result, _ = self.match({1: [(0, 100)], 2: [(0, 10)]}, {1: [(0, 100, 11)]})
        self.assertEqual(result["train"][80], {11: (1, 90, 100)})

    def test_schedule_ambiguity_is_not_bridged_and_zero_is_not_an_item(self):
        result, audit = self.match({1: [(0, 100)]}, {1: [(0, 100, 11), (40, 60, 0)]})
        self.assertEqual(result, {"train": {50: {}, 80: {}}})
        self.assertEqual(audit["train"]["ambiguous_schedule_overlap_seconds"], 20)
        zero, _ = self.match({1: [(0, 100)]}, {1: [(0, 100, 0)]})
        self.assertEqual(zero, {"train": {50: {}, 80: {}}})

    def test_period_clipping_keeps_full_airing_denominator(self):
        result, _ = self.match({1: [(0, 200)]}, {1: [(0, 200, 11)]}, periods=(("train", 0, 100), ("validation", 100, 200)))
        self.assertEqual(result, {"train": {50: {}, 80: {}}, "validation": {50: {}, 80: {}}})
        final, _ = self.match({1: [(0, 200)]}, {1: [(0, 200, 11)]}, periods=(("final", 0, 200),))
        self.assertEqual(final["final"][80], {11: (1, 200, 200)})

    def test_short_programmes_can_qualify_below_60_seconds(self):
        result, _ = self.match({1: [(0, 41), (100, 149)]}, {1: [(0, 50, 11), (100, 160, 22)]})
        self.assertEqual(result["train"][80], {11: (1, 41, 41), 22: (1, 49, 149)})

    def test_touching_positive_segments_same_airing_join_only_inside_one_row(self):
        index = {
            "starts": [0, 50], "ends": [50, 100], "items": [11, 11], "airings": [0, 0],
            "airing_starts": [0], "airing_ends": [100],
        }
        joined, _ = match_sessions({1: [(0, 100)]}, {1: index}, [("train", 0, 100)])
        separate, _ = match_sessions({1: [(0, 50), (50, 100)]}, {1: index}, [("train", 0, 100)])
        self.assertEqual(joined["train"][80], {11: (1, 100, 100)})
        self.assertEqual(separate["train"][80], {})

    def test_cross_channel_same_item_union_and_clock_gap_use_existing_coordinates(self):
        result, _ = self.match({1: [(0, 100)], 2: [(20, 120)]}, {1: [(0, 100, 11)], 2: [(20, 120, 11)]}, exclude_cross_channel=False)
        self.assertEqual(result["train"][80], {11: (1, 120, 120)})
        start = wall_seconds("2020-03-29 02:50:00")
        stop = wall_seconds("2020-03-29 04:10:00")
        dst, _ = self.match({1: [(start, stop)]}, {1: [(start, start + 1200, 11)]}, periods=(("train", start, stop),))
        self.assertEqual(dst["train"][80], {11: (1, 1200, stop)})

    def test_optional_sink_preserves_result_and_deduplicates_qualified_airings(self):
        index = build_channel_index([(0, 100, 11)])[0]
        views = {1: [(0, 90), (5, 95), (0, 90)]}
        expected = match_sessions(views, {1: index}, [("train", 0, 100)])
        sink = []
        actual = match_sessions(views, {1: index}, [("train", 0, 100)], qualified_events=sink)
        self.assertEqual(actual, expected)
        self.assertEqual(sink, [
            ("train", 50, 11, 1, 0, 100, 0, 95),
            ("train", 80, 11, 1, 0, 100, 0, 95),
        ])

    def test_sink_keeps_full_airing_bounds_and_clipped_watch_bounds(self):
        index = build_channel_index([(0, 100, 11)])[0]
        sink = []
        result, _ = match_sessions(
            {1: [(0, 100)]}, {1: index}, [("train", 0, 90), ("validation", 90, 150)], qualified_events=sink,
        )
        self.assertEqual(result["validation"], {50: {}, 80: {}})
        self.assertEqual(sink, [
            ("train", 50, 11, 1, 0, 100, 0, 90),
            ("train", 80, 11, 1, 0, 100, 0, 90),
        ])

    def test_sink_does_not_merge_adjacent_airings_of_same_programme(self):
        index = build_channel_index([(0, 100, 11), (100, 200, 11)])[0]
        sink = []
        result, _ = match_sessions({1: [(0, 200)]}, {1: index}, [("train", 0, 200)], thresholds=(50,), qualified_events=sink)
        self.assertEqual(result["train"][50], {11: (2, 200, 200)})
        self.assertEqual(sink, [
            ("train", 50, 11, 1, 0, 100, 0, 100),
            ("train", 50, 11, 1, 100, 200, 100, 200),
        ])

    def test_development_collector_single_pass_cache_and_preserved_standard_files(self):
        from scripts import prepare_history_context as collector

        with tempfile.TemporaryDirectory(prefix="history-context-test-") as directory:
            root = Path(directory)
            data = root / "data/processed"
            data.mkdir(parents=True)
            (root / "results").mkdir()
            (root / "submission_v2.csv").write_text("protected v2\n", encoding="utf-8")
            standard = data / "session_train_50.csv"
            standard.write_text("protected standard\n", encoding="utf-8")
            for name in (
                "scripts/prepare_history_context.py", "scripts/prepare_interactions.py",
                "scripts/prepare_sessions.py", "src/session_interactions.py",
                "data/dataset11-30.csv", "data/export_arh_11-20-final.csv", "data/export_arh_21-30-final.csv",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            first = wall_seconds("2020-03-15 00:00:00")
            second = wall_seconds("2020-04-20 00:00:00")
            index = build_channel_index([(first, first + 100, 11), (second, second + 100, 22)])[0]

            def spool(work):
                (work / "views_00.bin").write_bytes(
                    collector.RECORD.pack(0, 1, first, first + 100, 100)
                    + collector.RECORD.pack(1, 1, second, second + 100, 100)
                )
                return {"rows": 2}, ["00012345678901234567", "99912345678901234567"]

            with patch.multiple(collector, ROOT=root, DATA=data, BUCKETS=1), \
                    patch.object(collector, "_references", return_value={}), \
                    patch.object(collector, "history_index", return_value=({1: index}, {})), \
                    patch.object(collector, "read_views", side_effect=spool) as read:
                manifests = collector.prepare("development")
                self.assertEqual(read.call_count, 1)
                cached = collector.prepare("development")
                self.assertEqual(read.call_count, 1)
                self.assertEqual(cached, manifests)
            self.assertEqual(standard.read_text(encoding="utf-8"), "protected standard\n")
            self.assertEqual((root / "submission_v2.csv").read_text(encoding="utf-8"), "protected v2\n")
            expected = (("train_50", "train", 11), ("train_60", "train", 11), ("validation_80", "validation", 22))
            for key, period, item in expected:
                with (root / manifests["early"]["outputs"][key]["path"]).open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([(row["period"], int(row["tv_show_id"])) for row in rows], [(period, item)])
            self.assertEqual(manifests["main"]["outputs"]["train_airings"]["rows"], 2)
            self.assertEqual(manifests["early"]["outputs"]["train_airings"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
