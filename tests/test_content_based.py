"""Контролируемые проверки независимого content-based профиля."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.prepare_interactions import SCHEDULE_COLUMNS
from src.content_based import fit_content_model, load_train_metadata, score_content


class ContentBasedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sweet-tv-content-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "schedule.csv"

    def write_schedule(self, changes):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEDULE_COLUMNS)
            writer.writeheader()
            for values in changes:
                row = dict.fromkeys(SCHEDULE_COLUMNS, "")
                row.update(channel_id="1", start_time="10.03.2020 12:00:00", duration="3600", tv_show_id="11")
                row.update(values)
                writer.writerow(row)

    def load(self):
        return load_train_metadata([self.path], "2020-03-09 00:00:00", "2020-05-18 00:00:00")

    def test_calendar_train_filter_excludes_future_metadata_and_boundary_touch(self):
        self.write_schedule([
            {"start_time": "17.05.2020 23:30:00", "tv_show_category": "TRAIN"},
            {"start_time": "18.05.2020 00:00:00", "tv_show_category": "FUTURE", "tv_show_id": "12"},
            {"start_time": "08.03.2020 23:00:00", "tv_show_category": "BEFORE", "tv_show_id": "13"},
            {"duration": "0", "tv_show_category": "ZERO", "tv_show_id": "14"},
            {"tv_show_id": "0", "tv_show_category": "UNLABELLED"},
        ])
        metadata, audit = self.load()
        self.assertEqual(set(metadata), {11})
        self.assertEqual(metadata[11]["basic"], {"category:train"})
        self.assertEqual(audit["included_train_airing_rows"], 1)
        model = fit_content_model(metadata, sparse.csr_matrix([[1]]), [11])
        self.assertEqual(model["feature_names"], ["category:train"])
        self.assertTrue(np.isnan(score_content(model, [0], [12])[0, 0]))

    def test_metadata_union_handles_genre_order_names_and_initial_year(self):
        self.write_schedule([
            {"tv_show_category": "  Драма ", "tv_show_genre_1": "Боевик", "tv_show_genre_2": " ДРАМА ",
             "director": "  Имя   Фамилия ", "actors": "А Первый, Б Второй; В Третий / Г Четвёртый",
             "year_of_production": "1989-0000"},
            {"tv_show_category": "Комедия", "tv_show_genre_1": "драма", "tv_show_genre_3": "боевик",
             "year_of_production": "2010-"},
            {"year_of_production": "0000-"},
        ])
        metadata, audit = self.load()
        self.assertEqual(metadata[11]["basic"], {"category:драма", "category:комедия", "genre:боевик", "genre:драма"})
        self.assertTrue(metadata[11]["basic"].issubset(metadata[11]["extended"]))
        self.assertEqual(metadata[11]["extended"] - metadata[11]["basic"], {
            "director:имя фамилия", "actor:а первый", "actor:б второй", "actor:в третий", "actor:г четвёртый",
            "year_decade:1980", "year_decade:2010",
        })
        self.assertEqual(audit["included_rows_without_valid_initial_year"], 1)

    def test_tfidf_uses_train_item_documents_and_scores_unobserved_known_item(self):
        metadata = {
            1: {"basic": {"genre:a"}},
            2: {"basic": {"genre:a", "genre:b"}},
            3: {"basic": set()},
        }
        model = fit_content_model(metadata, sparse.csr_matrix([[2]]), [1])
        scores = score_content(model, [0], [1, 2, 3, 99])
        idf_a, idf_b = np.log(4 / 3) + 1, np.log(4 / 2) + 1
        self.assertAlmostEqual(float(scores[0, 0]), 1)
        self.assertAlmostEqual(float(scores[0, 1]), idf_a / np.sqrt(idf_a**2 + idf_b**2), places=6)
        self.assertTrue(np.all(np.isnan(scores[0, 2:])))
        self.assertEqual(scores.dtype, np.float32)

    def test_binary_and_log_profiles_match_manual_cosines_and_unknown_users(self):
        metadata = {1: {"basic": {"a"}}, 2: {"basic": {"b"}}}
        counts = sparse.csr_matrix([[1, 3], [0, 0]])
        binary = fit_content_model(metadata, counts, [1, 2])
        weighted = fit_content_model(metadata, counts, [1, 2], history_weighting="log1p")
        np.testing.assert_allclose(score_content(binary, [0], [1, 2]), [[1 / np.sqrt(2)] * 2], rtol=1e-6)
        scores = score_content(weighted, [0, 1, -1, 99], [1, 2])
        np.testing.assert_allclose(scores[0], [1 / np.sqrt(5), 2 / np.sqrt(5)], rtol=1e-6)
        self.assertTrue(np.all(np.isnan(scores[1:])))

    def test_extended_feature_set_and_missing_observed_metadata(self):
        metadata = {1: {"basic": set(), "extended": {"director:a"}}}
        basic = fit_content_model(metadata, sparse.csr_matrix([[1], [0]]), [1])
        extended = fit_content_model(metadata, sparse.csr_matrix([[1], [0]]), [1], feature_set="extended")
        self.assertTrue(np.isnan(score_content(basic, [0], [1])[0, 0]))
        self.assertAlmostEqual(float(score_content(extended, [0], [1])[0, 0]), 1)
        missing = fit_content_model(metadata, sparse.csr_matrix([[1]]), [99], feature_set="extended")
        self.assertEqual(missing["audit"]["train_observed_items_without_metadata"], 1)
        self.assertTrue(np.isnan(score_content(missing, [0], [1])[0, 0]))

    def test_batch_boundary_and_invalid_configuration(self):
        metadata = {1: {"basic": {"a"}}}
        model = fit_content_model(metadata, sparse.csr_matrix(np.ones((129, 1))), [1])
        scores = score_content(model, list(range(129)), [1, 99])
        np.testing.assert_allclose(scores[:, 0], np.ones(129))
        self.assertTrue(np.all(np.isnan(scores[:, 1])))
        for kwargs in ({"feature_set": "future"}, {"history_weighting": "duration"}):
            with self.assertRaises(ValueError):
                fit_content_model(metadata, sparse.csr_matrix([[1]]), [1], **kwargs)
        with self.assertRaises(ValueError):
            fit_content_model(metadata, sparse.csr_matrix([[-1]]), [1])


if __name__ == "__main__":
    unittest.main()
