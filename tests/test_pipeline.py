import tempfile
import unittest
from pathlib import Path

from nsl_kdd_ids.pipeline import (
    ATTACK_FAMILIES,
    FEATURE_COLUMNS,
    _read_nsl_file,
)


class NslKddDataTests(unittest.TestCase):
    def test_feature_schema_and_attack_family_coverage(self) -> None:
        self.assertEqual(len(FEATURE_COLUMNS), 41)
        self.assertEqual(ATTACK_FAMILIES["normal"], "Normal")
        self.assertEqual(ATTACK_FAMILIES["neptune"], "DoS")
        self.assertEqual(ATTACK_FAMILIES["satan"], "Probe")
        self.assertEqual(ATTACK_FAMILIES["guess_passwd"], "R2L")
        self.assertEqual(ATTACK_FAMILIES["buffer_overflow"], "U2R")

    def test_reader_normalizes_labels_and_drops_difficulty(self) -> None:
        row = ["0"] * len(FEATURE_COLUMNS)
        row[1:4] = ["tcp", "http", "SF"]
        rows = [
            ",".join([*row, "normal.", "20"]),
            ",".join([*row, "neptune", "19"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "sample.txt"
            data_path.write_text("\n".join(rows), encoding="utf-8")

            data = _read_nsl_file(data_path)

        self.assertEqual(list(data.columns[-3:]), ["attack_label", "attack_category", "is_attack"])
        self.assertEqual(data["attack_label"].tolist(), ["normal", "neptune"])
        self.assertEqual(data["attack_category"].tolist(), ["Normal", "DoS"])
        self.assertEqual(data["is_attack"].tolist(), [0, 1])
        self.assertNotIn("difficulty", data.columns)


if __name__ == "__main__":
    unittest.main()
