"""Ручные проверки двухстадийных кандидатов, oracle и final Top-5."""

import copy
import csv
import importlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import wall_seconds
from src.history_reranking import load_train_context, schedule_context


class HGBRerankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reranker = importlib.import_module("src.hgb_reranker")

    def test_candidate_union_has_200_unique_ids_and_preserves_mandatory_v3(self):
        lists = {
            "ppr": [list(range(1, 151)), list(range(151, 1, -1))],
            "repeat": [list(range(51, 251)), list(range(100, 300))],
            "als": [list(range(101, 301)), list(range(201, 401))],
            "content": [list(range(151, 351)), list(range(250, 450))],
        }
        mandatory = [[999, 998, 997, 996, 995], [1, 2, 3, 4, 5]]
        popularity = list(range(1, 501))
        original = copy.deepcopy((lists, mandatory, popularity))
        pairs = self.reranker.candidate_union(lists, mandatory, popularity)
        repeated = self.reranker.candidate_union(lists, mandatory, popularity)
        self.assertEqual(pairs.shape, (2, 200))
        self.assertEqual(pairs.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(pairs, repeated)
        np.testing.assert_array_equal(pairs[:, :5], mandatory)
        for row, required in zip(pairs, mandatory):
            self.assertEqual(len(set(row)), 200)
            self.assertTrue(set(required) <= set(row))
            self.assertTrue((row > 0).all())
        self.assertTrue(set(range(1, 101)) <= set(pairs[0]))
        self.assertEqual((lists, mandatory, popularity), original)

    def test_candidate_union_rejects_insufficient_distinct_candidates(self):
        with self.assertRaises(ValueError):
            self.reranker.candidate_union(
                {"ppr": [[1, 2, 3, 4, 5]], "repeat": [[1, 2, 3, 4, 5]]},
                [[1, 2, 3, 4, 5]],
                [1, 2, 3, 4, 5],
            )

    def test_candidate_oracle_and_recall_keep_full_ground_truth_denominators(self):
        users = ["00012345678901234567", "partial", "more_than_five", "unavailable"]
        pairs = np.array(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 8, 9, 10], [1, 2, 3, 4, 5]],
            dtype=np.int64,
        )
        truth = {users[0]: set(), users[1]: {1, 99}, users[2]: set(range(1, 8)), users[3]: {98, 99}}
        summary = self.reranker.candidate_diagnostics(pairs, users, truth)
        self.assertAlmostEqual(summary["macro_recall_at_200"], (0.5 + 2 / 7) / 4)
        self.assertAlmostEqual(summary["micro_recall_at_200"], 3 / 11)
        self.assertAlmostEqual(summary["nonempty_macro_recall_at_200"], (0.5 + 2 / 7) / 3)
        self.assertAlmostEqual(summary["any_hit_fraction"], 0.5)
        self.assertAlmostEqual(summary["oracle_map_at_5"], (0.5 + 2 / 5) / 4)

    def test_final_ranking_returns_exact_five_numeric_ties_and_forced_cold_fallback(self):
        pairs = np.array([[10, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 10]], dtype=np.int64)
        probabilities = np.array([[0.5, 0.5, 0.2, 0.1, 0.05, 0.01], [0, 0, 0, 0, 0, 1]])
        fallback = [6, 5, 4, 3, 2]
        predicted = self.reranker.rank_candidate_predictions(
            probabilities, pairs, [True, False], fallback
        )
        self.assertEqual(predicted, [[2, 10, 3, 4, 5], fallback])
        for row in predicted:
            self.assertEqual(len(row), 5)
            self.assertEqual(len(set(row)), 5)

    def test_warm_ranking_requires_five_finite_model_scores(self):
        pairs = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        probabilities = np.array([[0.9, 0.8, 0.7, 0.6, np.nan, np.nan]])
        with self.assertRaises(ValueError):
            self.reranker.rank_candidate_predictions(
                probabilities, pairs, [True], [1, 2, 3, 4, 5]
            )

    def feature_fixture(self):
        user = "00012345678901234567"
        history = {
            "counts": sparse.csr_matrix([[2, 1]], dtype=np.float32),
            "last": sparse.csr_matrix(
                [[wall_seconds("2020-04-05 00:50:00"), wall_seconds("2020-04-05 06:50:00")]],
                dtype=np.float64,
            ),
            "users": {user: 0},
            "items": np.array([11, 22], dtype=np.int64),
            "lower": "2020-03-29 00:00:00",
            "cutoff": wall_seconds("2020-04-06 00:00:00"),
            "period": "train",
        }
        users = [user, "98765432109876543210"]
        pairs = np.array([[11, 22, 33, 44], [11, 22, 33, 44]], dtype=np.int64)
        with tempfile.TemporaryDirectory(prefix="sweet-tv-hgb-features-") as temporary:
            folder = Path(temporary)
            airings = folder / "airings.csv"
            with airings.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["user_id", "tv_show_id", "channel_id", "airing_start", "airing_end", "watch_start", "watch_end", "period"]
                )
                writer.writerows(
                    [
                        [user, 11, 7, "2020-04-04 00:20:00", "2020-04-04 00:50:00", "2020-04-04 00:30:00", "2020-04-04 00:50:00", "train"],
                        [user, 11, 7, "2020-04-05 00:20:00", "2020-04-05 00:50:00", "2020-04-05 00:30:00", "2020-04-05 00:50:00", "train"],
                        [user, 22, 9, "2020-04-05 06:20:00", "2020-04-05 06:50:00", "2020-04-05 06:30:00", "2020-04-05 06:50:00", "train"],
                    ]
                )
            context = load_train_context(airings, history)
            schedule_path = folder / "schedule.csv"
            with schedule_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["channel_id", "start_time", "duration", "tv_show_id", "tv_show_title"])
                writer.writerows(
                    [
                        [7, "12.04.2020 00:30:00", 1800, 11, "A"],
                        [9, "12.04.2020 06:30:00", 1800, 22, "B"],
                        [99, "13.04.2020 09:30:00", 1800, 33, "C"],
                    ]
                )
            schedule = schedule_context(
                [schedule_path], "2020-04-06 00:00:00", "2020-04-14 00:00:00", pairs[0]
            )
        return history, context, users, pairs, schedule

    def test_temporal_channel_features_use_original_calendar_and_manual_shares(self):
        history, context, users, pairs, schedule = self.feature_fixture()
        values, names = self.reranker.history_features(
            history, context, users, pairs, schedule, forecast_days=8
        )
        features = {name: values[:, :, column] for column, name in enumerate(names)}
        # Эта календарная точка — воскресенье/bin0. Прямое чтение наших
        # нормализованных координат как Unix seconds ошибочно дало бы субботу/bin7.
        naive = datetime(1970, 1, 1) + timedelta(seconds=wall_seconds("2020-04-05 00:30:00"))
        self.assertEqual((naive.weekday(), naive.hour // 3), (5, 7))
        self.assertEqual(context["weekday"][0][6], 2)
        self.assertEqual(context["hour"][0][0], 2)
        self.assertEqual(schedule["items"][11]["joint"][(7, 6, 0)], 1)
        for name, expected in (
            ("hour_alignment", [2 / 3, 1 / 3, 0, 0]),
            ("weekday_alignment", [2 / 3, 2 / 3, 0, 0]),
            ("weekday_hour_alignment", [1 / 3, 1 / 3, 0, 0]),
            ("channel_alignment", [2 / 3, 1 / 3, 0, 0]),
            ("channel_hour_alignment", [2 / 3, 1 / 3, 0, 0]),
            ("joint_alignment", [1 / 3, 1 / 3, 0, 0]),
        ):
            with self.subTest(feature=name):
                np.testing.assert_allclose(features[name][0], expected, rtol=1e-6)
        for name in ("hour_concentration", "weekday_concentration", "channel_concentration"):
            np.testing.assert_allclose(features[name][0], [5 / 9] * 4, rtol=1e-6)
        np.testing.assert_allclose(features["weekday_hour_concentration"][0], [1 / 3] * 4, rtol=1e-6)
        np.testing.assert_array_equal(features["history_nchannels"][0], [2] * 4)
        np.testing.assert_array_equal(features["item_distinct_days"][0], [2, 1, 0, 0])
        np.testing.assert_array_equal(features["item_day_count_regularity"][0], [1, 1, 0, 0])
        train_days = 8 - 1 / 24
        np.testing.assert_allclose(features["watches_per_train_day"][0], [3 / train_days] * 4, rtol=1e-6)
        np.testing.assert_allclose(features["item_watch_rate"][0], [2 / train_days, 1 / train_days, 0, 0], rtol=1e-6)
        np.testing.assert_allclose(features["item_last_age_days"][0, :2], [23 / 24 + 10 / 1440, 17 / 24 + 10 / 1440], rtol=1e-6)
        self.assertTrue(np.isnan(features["item_last_age_days"][0, 2:]).all())

    def test_three_dimensional_feature_flatten_and_group_masks_keep_pair_alignment(self):
        history, context, users, pairs, schedule = self.feature_fixture()
        values, names = self.reranker.history_features(
            history, context, users, pairs, schedule, forecast_days=8
        )
        expected_names = sum(self.reranker.FEATURE_GROUPS.values(), [])
        self.assertEqual(names, expected_names)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(values.shape, (2, 4, len(names)))
        self.assertEqual(values.dtype, np.dtype(np.float32))
        flat = values.reshape(-1, len(names))
        days = names.index("item_distinct_days")
        np.testing.assert_array_equal(pairs.ravel(), [11, 22, 33, 44, 11, 22, 33, 44])
        np.testing.assert_array_equal(flat[:, days], [2, 1, 0, 0, 0, 0, 0, 0])
        group_columns = {group: [names.index(name) for name in columns]
                         for group, columns in self.reranker.FEATURE_GROUPS.items()}
        all_columns = sum(group_columns.values(), [])
        self.assertEqual(sorted(all_columns), list(range(len(names))))
        self.assertEqual([len(group_columns[group]) for group in ("B", "C", "D")], [7, 6, 5])
        cold_context_columns = group_columns["C"] + group_columns["D"]
        np.testing.assert_array_equal(flat[4:, cold_context_columns], 0)
        self.assertTrue(np.isnan(flat[4:, names.index("item_last_age_days")]).all())

    def test_temporal_guard_accepts_touching_cutoff_but_rejects_overlap_or_empty_target(self):
        runner = importlib.import_module("scripts.experiment_hgb")
        runner.temporal_check(
            "2020-03-09 00:00:00", "2020-04-13 00:00:00",
            "2020-04-13 00:00:00", "2020-05-18 00:00:00",
        )
        for cutoff, target_start, target_end in (
            ("2020-04-13 00:00:01", "2020-04-13 00:00:00", "2020-05-18 00:00:00"),
            ("2020-04-13 00:00:00", "2020-04-13 00:00:00", "2020-04-13 00:00:00"),
        ):
            with self.subTest(cutoff=cutoff, target_start=target_start, target_end=target_end):
                with self.assertRaises(ValueError):
                    runner.temporal_check("2020-03-09 00:00:00", cutoff, target_start, target_end)

    def test_runner_feature_groups_are_nested_and_select_columns_by_name(self):
        runner = importlib.import_module("scripts.experiment_hgb")
        # Меняем физический порядок колонок: ablation должна выбирать имена,
        # а не считать, что нужные признаки всегда лежат первыми в массиве.
        names = list(reversed(runner.BASE_NAMES + sum(self.reranker.FEATURE_GROUPS.values(), [])))
        expected = runner.BASE_NAMES.copy()
        previous = []
        for variant, width in (("A", 15), ("B", 22), ("C", 28), ("D", 33)):
            if variant != "A":
                expected.extend(self.reranker.FEATURE_GROUPS[variant])
            columns = runner.feature_columns(names, variant)
            self.assertEqual(len(columns), width)
            self.assertEqual([names[column] for column in columns], expected)
            self.assertEqual(columns[:len(previous)], previous)
            previous = columns
        with self.assertRaises(ValueError):
            runner.feature_columns(names, "E")

    def test_runner_user_selection_preserves_candidate_and_flattened_row_alignment(self):
        runner = importlib.import_module("scripts.experiment_hgb")
        users = np.array(["00012345678901234567", "98765432109876543210", "cold"])
        pairs = np.array([list(range(10, 15)), list(range(20, 25)), list(range(10, 15))], dtype=np.int64)
        feature_cube = np.stack(
            [np.column_stack((row, np.full(5, marker)))
             for row, marker in zip(pairs, (100, 200, 300))]
        ).astype(np.float32)
        data = {
            "X": feature_cube.reshape(-1, 2),
            "pairs": pairs,
            "users": users,
            "warm": np.array([True, True, False]),
            "v3": pairs.copy(),
            "feature_names": np.array(["score", "aux"]),
            "fallback": np.arange(10, 15),
            "candidate_pool": np.array(list(range(10, 15)) + list(range(20, 25))),
        }
        requested = [str(users[1]), str(users[2]), str(users[0])]
        selected = runner.select_rows(data, requested)
        np.testing.assert_array_equal(selected["users"], requested)
        np.testing.assert_array_equal(selected["pairs"], pairs[[1, 2, 0]])
        np.testing.assert_array_equal(selected["v3"], pairs[[1, 2, 0]])
        np.testing.assert_array_equal(selected["warm"], [True, False, True])
        np.testing.assert_array_equal(
            selected["X"].reshape(3, 5, 2), feature_cube[[1, 2, 0]]
        )
        np.testing.assert_array_equal(selected["X"][:, 0], selected["pairs"].ravel())
        with self.assertRaises(ValueError):
            runner.select_rows({**data, "X": data["X"][:-1]}, requested)

    def test_saved_oof_reuses_selected_cache_and_fails_closed_without_rebuilding(self):
        runner = importlib.import_module("scripts.experiment_hgb")
        with tempfile.TemporaryDirectory(prefix="sweet-tv-hgb-oof-") as temporary:
            root = Path(temporary)
            data = root / "cache"
            data.mkdir()
            source = root / "source.csv"
            source.write_bytes(b"frozen historical source\n")
            source_stat = source.stat()
            signature = {
                "implementation": {"fixture": "selected run"},
                "inputs": {"source.csv": [source_stat.st_size, source_stat.st_mtime_ns]},
            }
            meta = {"signature": signature, "target_interval": ["2020-04-13", "2020-05-18"]}
            expected = copy.deepcopy(meta)
            arrays = {
                "X": np.arange(20, dtype=np.float32).reshape(10, 2),
                "y": np.arange(10, dtype=np.uint8) % 2,
                "pairs": np.array([list(range(11, 16)), list(range(21, 26))], dtype=np.int64),
                "users": np.array(["00012345678901234567", "98765432109876543210"]),
            }
            path, meta_path = data / "early.npz", data / "early.json"
            np.savez_compressed(path, **arrays)
            meta_text = json.dumps(meta)
            meta_path.write_text(meta_text, encoding="utf-8")
            cached_bytes = path.read_bytes()

            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner, "DATA", data),
                patch.object(runner, "snapshot", side_effect=AssertionError("OOF regeneration prohibited")) as regenerate,
                patch.object(runner.HistGradientBoostingClassifier, "fit", side_effect=AssertionError("OOF refit prohibited")) as refit,
            ):
                loaded, loaded_meta = runner.load_saved_oof("early", expected)
                self.assertEqual(loaded_meta, meta)
                self.assertEqual(set(loaded), set(arrays))
                for name, values in arrays.items():
                    np.testing.assert_array_equal(loaded[name], values)
                self.assertEqual(path.read_bytes(), cached_bytes)

                with self.subTest(invalid="missing_npz"):
                    path.unlink()
                    with self.assertRaises(ValueError):
                        runner.load_saved_oof("early", expected)
                    self.assertFalse(path.exists())
                    path.write_bytes(cached_bytes)
                with self.subTest(invalid="missing_metadata"):
                    meta_path.unlink()
                    with self.assertRaises(ValueError):
                        runner.load_saved_oof("early", expected)
                    self.assertFalse(meta_path.exists())
                    meta_path.write_text(meta_text, encoding="utf-8")
                with self.subTest(invalid="different_signature"):
                    different = copy.deepcopy(meta)
                    different["signature"]["implementation"]["fixture"] = "another experiment"
                    meta_path.write_text(json.dumps(different), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runner.load_saved_oof("early", expected)
                    self.assertEqual(path.read_bytes(), cached_bytes)
                    meta_path.write_text(meta_text, encoding="utf-8")
                with self.subTest(invalid="changed_source"):
                    source.write_bytes(b"changed source contents and size\n")
                    with self.assertRaises(ValueError):
                        runner.load_saved_oof("early", expected)
                    self.assertEqual(path.read_bytes(), cached_bytes)
                regenerate.assert_not_called()
                refit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
