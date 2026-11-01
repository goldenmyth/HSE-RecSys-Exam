"""Пустая comparison-группа не доказывает полезность вспомогательной модели."""

import unittest

import numpy as np

from scripts.complement_ppr import diagnostics


class ComplementDiagnosticsTests(unittest.TestCase):
    def test_identical_full_top20_with_no_unflagged_items_fails_closed(self):
        users, candidates = [str(i) for i in range(8)], np.arange(1, 21)
        scores = np.tile(np.linspace(1, .9, 20), (8, 1))
        truth = {user: set(candidates) for user in users}
        result = diagnostics(scores, scores, users, candidates, truth, {user: 30 for user in users})
        self.assertGreaterEqual(result["top_lists"]["20"]["actionable_new_hits"], 63)
        self.assertEqual(result["boundary"]["unflagged_slots"], 0)
        self.assertFalse(result["useful_for_small_boost"])


if __name__ == "__main__":
    unittest.main()
