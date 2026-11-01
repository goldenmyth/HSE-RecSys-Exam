"""Ручные проверки единого final train-окна без сложения offline-агрегатов."""

import csv
import importlib
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_interactions import build_channel_index, match_user


class FinalTrainingWindowTests(unittest.TestCase):
    def match(self, views, airings, periods):
        index, _ = build_channel_index(airings)
        return match_user({1: views}, {1: index}, periods, threshold=60)

    def test_short_parts_of_one_airing_qualify_only_in_full_window(self):
        views = [(60, 140)]
        airings = [(0, 200, 11)]
        split, _ = self.match(
            views, airings, [("train", 0, 100), ("validation", 100, 200)]
        )
        final, _ = self.match(views, airings, [("final", 0, 200)])
        self.assertEqual(split, {"train": {}, "validation": {}})
        self.assertEqual(tuple(final["final"][11]), (1, 80, 140))

    def test_accepted_split_parts_are_one_airing_in_final_window(self):
        views = [(30, 170)]
        airings = [(0, 200, 11)]
        split, _ = self.match(
            views, airings, [("train", 0, 100), ("validation", 100, 200)]
        )
        final, _ = self.match(views, airings, [("final", 0, 200)])
        self.assertEqual(tuple(split["train"][11]), (1, 70, 100))
        self.assertEqual(tuple(split["validation"][11]), (1, 70, 170))
        self.assertEqual(tuple(final["final"][11]), (1, 140, 170))

    def test_fragmented_duplicate_views_crossing_old_cutoff_count_unique_seconds(self):
        final, diagnostics = self.match(
            [(50, 100), (50, 100), (75, 125), (125, 160)],
            [(0, 200, 11), (0, 200, 11)],
            [("final", 0, 200)],
        )
        self.assertEqual(tuple(final["final"][11]), (1, 110, 160))
        self.assertEqual(diagnostics["final"]["unique_user_channel_view_seconds"], 110)
        self.assertEqual(diagnostics["final"]["accepted_user_airings"], 1)

    def test_final_end_clips_crossing_views_and_excludes_test_only_item(self):
        final, diagnostics = self.match(
            [(130, 250), (200, 300)],
            [(0, 200, 11), (200, 300, 22)],
            [("final", 0, 200)],
        )
        self.assertEqual(final, {"final": {11: (1, 70, 200)}})
        self.assertNotIn(22, final["final"])
        self.assertEqual(diagnostics["final"]["unique_user_channel_view_seconds"], 70)

    def test_distinct_airings_cannot_combine_seconds_to_reach_threshold(self):
        final, _ = self.match(
            [(60, 140)],
            [(0, 100, 11), (100, 200, 11)],
            [("final", 0, 200)],
        )
        self.assertEqual(final, {"final": {}})


class SubmissionFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sweet-tv-final-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def helpers(self):
        return importlib.import_module("scripts.build_submission")

    def write_csv(self, path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def template(self):
        users = ["9007199254740993123", "3398461054087191302"]
        path = self.write_csv(
            self.folder / "template.csv",
            ["user_id", "tv_show_id"],
            [(user, "0 0 0 0 0") for user in users],
        )
        return path, users

    def test_calendar_candidates_include_overnight_airing_from_previous_file(self):
        columns = ["channel_id", "start_time", "duration", "tv_show_id", "tv_show_title"]
        history = self.write_csv(
            self.folder / "history.csv",
            columns,
            [
                (1, "26.07.2020 23:30:00", 3600, 101, "Overnight programme"),
                (2, "26.07.2020 23:00:00", 3600, 102, "Ends exactly at test start"),
            ],
        )
        test = self.write_csv(
            self.folder / "test.csv",
            columns,
            [
                (3, "27.07.2020 00:00:00", 600, 201, "Beta"),
                (3, "27.07.2020 00:10:00", 600, 201, "Alpha"),
                (4, "19.10.2020 00:00:00", 600, 202, "Starts at exclusive test end"),
                (5, "27.07.2020 00:00:00", 600, 0, "Unlabelled"),
            ],
        )
        indices, candidates, titles, _ = self.helpers().load_schedules(
            [history], test, "2020-07-27 00:00:00", "2020-10-19 00:00:00"
        )
        self.assertEqual(candidates, {101, 201})
        self.assertEqual(set(indices), {1, 2})
        self.assertEqual(titles[101], "Overnight programme")
        self.assertEqual(titles[201], "Alpha")

    def test_submission_preserves_long_user_ids_order_schema_and_rank(self):
        template, users = self.template()
        original = template.read_bytes()
        output = self.folder / "submission.csv"
        top5 = [50, 2, 30, 4, 5]
        checks = self.helpers().create_submission(
            template, output, top5, {2, 4, 5, 30, 50, 99}, expected_rows=2
        )
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["user_id", "tv_show_id"])
        self.assertEqual(rows[1:], [[user, "50 2 30 4 5"] for user in users])
        self.assertEqual(template.read_bytes(), original)
        self.assertIsInstance(checks, dict)

    def test_invalid_top_five_and_template_row_count_are_rejected(self):
        template, _ = self.template()
        output = self.folder / "submission.csv"
        for invalid in (
            [1, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 99],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6],
        ):
            with self.subTest(top5=invalid):
                with self.assertRaises(ValueError):
                    self.helpers().create_submission(
                        template, output, invalid, {1, 2, 3, 4, 5, 6}, expected_rows=2
                    )
        with self.assertRaises(ValueError):
            self.helpers().create_submission(
                template, output, [1, 2, 3, 4, 5], {1, 2, 3, 4, 5}, expected_rows=3
            )

    def test_submission_cannot_overwrite_original_template(self):
        template, _ = self.template()
        original = template.read_bytes()
        with self.assertRaises(ValueError):
            self.helpers().create_submission(
                template, template, [1, 2, 3, 4, 5], {1, 2, 3, 4, 5}, expected_rows=2
            )
        self.assertEqual(template.read_bytes(), original)

    def interactions(self, period):
        return self.write_csv(
            self.folder / "interactions.csv",
            ["user_id", "tv_show_id", "watch_count", "total_watch_seconds", "last_watch_time", "period"],
            [
                ("9007199254740993123", 11, 3, 120, "2020-07-26 12:00:00", period),
                ("3398461054087191302", 11, 5, 300, "2020-07-26 12:00:00", period),
            ],
        )

    def test_popularity_explicit_final_period_counts_unique_users(self):
        fit_popularity = importlib.import_module("src.popularity").fit_popularity
        scores, activity = fit_popularity(
            self.interactions("final_train"), expected_period="final_train"
        )
        self.assertEqual(dict(scores["unique_users"]), {11: 2})
        self.assertEqual(dict(scores["watch_count"]), {11: 8})
        self.assertEqual(dict(scores["watch_seconds"]), {11: 420})
        self.assertEqual(
            dict(activity), {"9007199254740993123": 1, "3398461054087191302": 1}
        )

    def test_popularity_default_rejects_final_and_validation_cannot_be_fit(self):
        fit_popularity = importlib.import_module("src.popularity").fit_popularity
        with self.assertRaises(ValueError):
            fit_popularity(self.interactions("final_train"))
        with self.assertRaises(ValueError):
            fit_popularity(self.interactions("validation"), expected_period="validation")


if __name__ == "__main__":
    unittest.main()
