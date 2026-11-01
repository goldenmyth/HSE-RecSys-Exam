"""Небольшие примеры с вручную рассчитанными интервалами и split."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_interactions.py"
SPEC = importlib.util.spec_from_file_location("prepare_interactions", MODULE_PATH)
preparation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preparation
SPEC.loader.exec_module(preparation)


class IntervalPreparationTests(unittest.TestCase):
    def make_index(self, records):
        index, _ = preparation.build_channel_index(records)
        return index

    def hits(self, index, start, end):
        return [
            (item, overlap_start, overlap_end)
            for item, _, overlap_start, overlap_end in preparation.iter_intersections(
                index, start, end
            )
        ]

    def match(self, views, schedules, periods=None, threshold=60):
        indexes = {
            channel: self.make_index(records)
            for channel, records in schedules.items()
        }
        interactions, _ = preparation.match_user(
            views,
            indexes,
            periods if periods is not None else [("train", 0, 1000)],
            threshold=threshold,
        )
        return interactions

    def test_merge_unsorted_overlapping_and_touching_views(self):
        self.assertEqual(
            preparation.merge_intervals(
                [(5, 10), (3, 8), (0, 5), (10, 10), (20, 19)]
            ),
            [(0, 10)],
        )

    def test_strict_merge_preserves_adjacent_airings(self):
        self.assertEqual(
            preparation.merge_intervals([(5, 10), (0, 5), (1, 4)], touching=False),
            [(0, 5), (5, 10)],
        )

    def test_period_slices_clip_and_exclude_right_boundary(self):
        periods = [("train", 0, 10), ("validation", 10, 20)]
        self.assertEqual(
            list(preparation.period_slices(-5, 15, periods)),
            [("train", 0, 10), ("validation", 10, 15)],
        )
        self.assertEqual(
            list(preparation.period_slices(10, 20, periods)),
            [("validation", 10, 20)],
        )

    def test_touching_intervals_have_no_overlap(self):
        index = self.make_index([(0, 10, 11)])
        self.assertEqual(self.hits(index, 10, 20), [])
        self.assertEqual(self.hits(index, -10, 0), [])

    def test_one_view_can_intersect_multiple_airings(self):
        index = self.make_index([(0, 10, 11), (10, 20, 22)])
        self.assertEqual(self.hits(index, 5, 15), [(11, 5, 10), (22, 10, 15)])

    def test_nested_conflicting_airings_are_ambiguous_only_in_overlap(self):
        index = self.make_index([(0, 100, 11), (40, 60, 22)])
        self.assertEqual(
            self.hits(index, 0, 100),
            [(11, 0, 40), (-1, 40, 60), (11, 60, 100)],
        )
        hits = list(preparation.iter_intersections(index, 0, 100))
        self.assertEqual(hits[0][1], hits[-1][1])

    def test_duplicate_and_nested_same_item_airings_do_not_repeat_seconds(self):
        index = self.make_index([(0, 100, 11), (20, 80, 11), (0, 100, 11)])
        self.assertEqual(self.hits(index, 0, 100), [(11, 0, 100)])

    def test_zero_and_labelled_overlap_is_not_silently_assigned(self):
        index = self.make_index([(0, 100, 0), (20, 40, 11)])
        self.assertEqual(
            self.hits(index, 0, 100),
            [(0, 0, 20), (-1, 20, 40), (0, 40, 100)],
        )
        interactions = self.match({1: [(0, 100)]}, {1: [(0, 100, 0)]})
        self.assertEqual(interactions.get("train", {}), {})

    def test_long_view_finds_every_intersection_across_gaps(self):
        records = [(20 * i, 20 * i + 10, i + 1) for i in range(2000)]
        index = self.make_index(records)
        expected = [(i + 1, 20 * i, 20 * i + 10) for i in range(2000)]
        self.assertEqual(self.hits(index, -1, 40000), expected)

    def test_duplicate_and_overlapping_views_are_unioned_before_counting(self):
        interactions = self.match(
            {1: [(0, 80), (40, 100), (0, 80)]},
            {1: [(0, 200, 11)]},
        )
        self.assertEqual(tuple(interactions["train"][11]), (1, 100, 100))

    def test_fragmented_view_reaches_threshold_within_one_airing(self):
        interactions = self.match(
            {1: [(0, 35), (60, 90)]},
            {1: [(0, 100, 11)]},
        )
        self.assertEqual(tuple(interactions["train"][11]), (1, 65, 90))

    def test_threshold_is_inclusive(self):
        below = self.match({1: [(0, 59)]}, {1: [(0, 100, 11)]})
        at_threshold = self.match({1: [(0, 60)]}, {1: [(0, 100, 11)]})
        self.assertEqual(below.get("train", {}), {})
        self.assertEqual(tuple(at_threshold["train"][11]), (1, 60, 60))

    def test_cross_period_seconds_cannot_combine_to_pass_threshold(self):
        periods = [("train", 0, 100), ("validation", 100, 200)]
        views = {1: [(50, 150)]}
        schedules = {1: [(0, 200, 11)]}
        rejected = self.match(views, schedules, periods)
        self.assertEqual(rejected.get("train", {}), {})
        self.assertEqual(rejected.get("validation", {}), {})
        accepted = self.match(views, schedules, periods, threshold=40)
        self.assertEqual(tuple(accepted["train"][11]), (1, 50, 100))
        self.assertEqual(tuple(accepted["validation"][11]), (1, 50, 150))

    def test_same_item_on_two_channels_does_not_double_count_time_or_airings(self):
        interactions = self.match(
            {1: [(0, 100)], 2: [(50, 150)]},
            {1: [(0, 200, 11)], 2: [(0, 200, 11)]},
        )
        self.assertEqual(tuple(interactions["train"][11]), (1, 150, 150))

    def test_back_to_back_airings_of_same_item_count_separately(self):
        interactions = self.match(
            {1: [(0, 200)]},
            {1: [(0, 100, 11), (100, 200, 11)]},
        )
        self.assertEqual(tuple(interactions["train"][11]), (2, 200, 200))

    def test_ambiguous_seconds_do_not_help_reach_threshold(self):
        interactions = self.match(
            {1: [(0, 100)]},
            {1: [(0, 100, 11), (20, 80, 22)]},
        )
        # У item 11 остаётся только 40 однозначных секунд, у item 22 — ноль.
        self.assertEqual(interactions.get("train", {}), {})

    def test_clock_gap_uses_elapsed_seconds_for_view_and_airing_overlap(self):
        start = preparation.wall_seconds("2020-03-29 02:50:00")
        end = preparation.wall_seconds("2020-03-29 04:10:00")
        self.assertEqual(end - start, 1200)

        airing_start = preparation.wall_seconds("2020-03-29 01:25:00")
        airing_end = airing_start + 6000
        self.assertEqual(preparation.iso(airing_end), "2020-03-29 04:05:00")
        view_start = preparation.wall_seconds("2020-03-29 04:00:00")
        index = self.make_index([(airing_start, airing_end, 11)])
        hits = list(preparation.iter_intersections(index, view_start, end))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 11)
        self.assertEqual(hits[0][3] - hits[0][2], 300)

    def test_clock_gap_timestamp_roundtrip_preserves_calendar_labels(self):
        for value in (
            "2020-03-29 02:59:59",
            "2020-03-29 04:00:00",
            "2020-05-18 00:00:00",
        ):
            with self.subTest(value=value):
                self.assertEqual(preparation.iso(preparation.wall_seconds(value)), value)

    def test_missing_calendar_hour_is_rejected(self):
        for value in ("2020-03-29 03:00:00", "2020-03-29 03:59:59"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    preparation.wall_seconds(value)


if __name__ == "__main__":
    unittest.main()
