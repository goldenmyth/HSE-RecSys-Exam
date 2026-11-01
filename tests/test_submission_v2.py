"""Персональный submission: точные ID/порядок шаблона и строгая валидация."""

import csv
import importlib
import tempfile
import unittest
from pathlib import Path


class PersonalSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sweet-tv-submission-v2-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.users = ["00012345678901234567", "98765432109876543210"]
        self.candidates = {1, 2, 3, 4, 5, 6}
        # Порядок словаря намеренно отличается от порядка строк шаблона.
        self.predictions = {
            self.users[1]: [5, 3, 1, 4, 2],
            self.users[0]: [1, 2, 3, 4, 5],
        }
        self.template = self.folder / "template.csv"
        self.output = self.folder / "submission_v2.csv"
        self.write_template()

    def write_template(self, users=None, header=None, extra_column=False):
        with self.template.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header if header is not None else ["user_id", "tv_show_id"])
            for user in self.users if users is None else users:
                row = [user, "0 0 0 0 0"]
                if extra_column:
                    row.append("unexpected")
                writer.writerow(row)

    def write_submission(self, predictions=None, output=None, expected_rows=2):
        helper = importlib.import_module("scripts.build_submission_v2").write_personal_submission
        return helper(
            self.template,
            self.output if output is None else output,
            self.predictions if predictions is None else predictions,
            self.candidates,
            expected_rows=expected_rows,
        )

    def test_per_user_lists_preserve_twenty_digit_ids_leading_zeros_and_order(self):
        original = self.template.read_bytes()
        checks = self.write_submission()
        with self.output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["user_id", "tv_show_id"])
        self.assertEqual(
            rows[1:],
            [[self.users[0], "1 2 3 4 5"], [self.users[1], "5 3 1 4 2"]],
        )
        self.assertEqual(self.template.read_bytes(), original)
        self.assertIsInstance(checks, dict)

    def test_invalid_programme_lists_leave_existing_output_unchanged(self):
        sentinel = b"existing output must survive failed validation\n"
        self.output.write_bytes(sentinel)
        for invalid in (
            [1, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [-1, 1, 2, 3, 4],
            [1, 2, 3, 4, 99],
            [1.0, 2, 3, 4, 5],
            ["1", 2, 3, 4, 5],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6],
        ):
            with self.subTest(programmes=invalid):
                predictions = dict(self.predictions)
                predictions[self.users[1]] = invalid
                with self.assertRaises(ValueError):
                    self.write_submission(predictions)
                self.assertEqual(self.output.read_bytes(), sentinel)

    def test_missing_extra_duplicate_users_schema_and_row_count_are_rejected(self):
        missing = {self.users[0]: self.predictions[self.users[0]]}
        extra = {**self.predictions, "extra_user": [1, 2, 3, 4, 5]}
        numeric_user = {
            int(self.users[0]): self.predictions[self.users[0]],
            self.users[1]: self.predictions[self.users[1]],
        }
        for predictions in (missing, extra, numeric_user):
            with self.subTest(predictions=predictions):
                with self.assertRaises(ValueError):
                    self.write_submission(predictions)
        with self.assertRaises(ValueError):
            self.write_submission(expected_rows=3)

        self.write_template(users=[self.users[0], self.users[0]])
        with self.assertRaises(ValueError):
            self.write_submission({self.users[0]: [1, 2, 3, 4, 5]})
        self.write_template(header=["tv_show_id", "user_id"])
        with self.assertRaises(ValueError):
            self.write_submission()
        self.write_template(extra_column=True)
        with self.assertRaises(ValueError):
            self.write_submission()

    def test_original_template_cannot_be_overwritten(self):
        original = self.template.read_bytes()
        with self.assertRaises(ValueError):
            self.write_submission(output=self.template)
        self.assertEqual(self.template.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
