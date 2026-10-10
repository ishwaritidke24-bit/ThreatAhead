import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nsl_kdd_ids.demo import (
    _normalize_record,
    _render_page,
    load_training_sample,
    predict_payload,
)
from nsl_kdd_ids.pipeline import FEATURE_COLUMNS


def valid_payload() -> dict[str, object]:
    payload: dict[str, object] = {name: 0 for name in FEATURE_COLUMNS}
    payload.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
    return payload


class DemoInterfaceTests(unittest.TestCase):
    def test_page_renders_every_feature_and_baseline_indicator(self) -> None:
        page = _render_page().decode("utf-8")
        self.assertIn("ThreatAhead", page)
        self.assertIn("Baseline Model", page)
        for feature in FEATURE_COLUMNS:
            self.assertIn(f'id="{feature}"', page)

    def test_payload_is_normalized_and_sent_to_baseline_inference(self) -> None:
        payload = valid_payload()
        payload["duration"] = "1.25"
        with patch(
            "nsl_kdd_ids.demo.predict_record",
            return_value={
                "model_config": "baseline",
                "decision_threshold": 0.5,
                "threshold_semantics": "default estimator decision",
                "binary_prediction": "Attack",
                "attack_family": "DoS",
            },
        ) as predict:
            result = predict_payload(payload)

        self.assertEqual(result["binary_prediction"], "Attack")
        record = predict.call_args.args[0]
        self.assertEqual(record["duration"], 1.25)
        predict.assert_called_once_with(record, model_config="baseline")

    def test_payload_rejects_missing_fields_and_invalid_values(self) -> None:
        payload = valid_payload()
        with self.assertRaisesRegex(ValueError, "Missing"):
            _normalize_record({key: value for key, value in payload.items() if key != "duration"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            _normalize_record({**payload, "extra": 1})
        for feature, invalid in (
            ("protocol_type", ""),
            ("service", 5),
            ("duration", True),
            ("duration", "nan"),
        ):
            with self.subTest(feature=feature, invalid=invalid):
                with self.assertRaisesRegex(ValueError, feature):
                    _normalize_record({**payload, feature: invalid})

    def test_sample_is_loaded_only_from_training_split(self) -> None:
        training_row = ["0"] * len(FEATURE_COLUMNS) + ["normal", "20"]
        training_row[1:4] = ["tcp", "http", "SF"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "KDDTrain+.txt").write_text(
                ",".join(training_row) + "\n", encoding="utf-8"
            )
            # A test file exists deliberately; the sample loader must not read it.
            (root / "KDDTest+.txt").write_text("not training data", encoding="utf-8")
            with patch("nsl_kdd_ids.demo.dataset_dir", return_value=root):
                sample = load_training_sample()

        self.assertEqual(sample["protocol_type"], "tcp")
        self.assertEqual(sample["service"], "http")
        self.assertEqual(sample["flag"], "SF")
        self.assertEqual(len(sample), len(FEATURE_COLUMNS))
        self.assertIsInstance(sample["duration"], float)


if __name__ == "__main__":
    unittest.main()
