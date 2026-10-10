import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nsl_kdd_ids.demo import (
    DemoPrerequisiteError,
    _normalize_record,
    _render_page,
    analyze_training_sample,
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
        self.assertIn("<details class=\"advanced\">", page)
        self.assertIn("Advanced: Manual Feature Entry", page)
        self.assertIn('id="record-form"', page)
        self.assertIn("Correct prediction", page)
        self.assertIn("Prediction mismatch", page)
        self.assertIn("Cannot reach the local demo server", page)
        self.assertIn("nsl_kdd_ids.demo", page)
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

    def test_sample_selection_uses_actual_training_labels_and_41_features(self) -> None:
        normal_row = ["0"] * len(FEATURE_COLUMNS) + ["normal.", "20"]
        attack_row = ["0"] * len(FEATURE_COLUMNS) + ["neptune", "21"]
        normal_row[1:4] = ["tcp", "http", "SF"]
        attack_row[1:4] = ["tcp", "private", "S0"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "KDDTrain+.txt").write_text(
                "\n".join([",".join(normal_row), ",".join(attack_row)]) + "\n",
                encoding="utf-8",
            )
            # A test file exists deliberately; the sample loader must not read it.
            (root / "KDDTest+.txt").write_text("not training data", encoding="utf-8")
            with patch("nsl_kdd_ids.demo.dataset_dir", return_value=root):
                normal_record, normal_label, normal_category = load_training_sample(
                    "normal"
                )
                attack_record, attack_label, attack_category = load_training_sample(
                    "attack"
                )

        self.assertEqual(normal_label, "normal")
        self.assertEqual(normal_category, "Normal")
        self.assertEqual(attack_label, "neptune")
        self.assertEqual(attack_category, "DoS")
        self.assertEqual(set(normal_record), set(FEATURE_COLUMNS))
        self.assertEqual(set(attack_record), set(FEATURE_COLUMNS))
        self.assertEqual(len(normal_record), 41)
        self.assertEqual(len(attack_record), 41)
        self.assertEqual(normal_record["service"], "http")
        self.assertEqual(attack_record["service"], "private")
        self.assertIsInstance(normal_record["duration"], float)

    def test_sample_result_displays_prediction_and_actual_label_separately(self) -> None:
        record = valid_payload()
        with (
            patch(
                "nsl_kdd_ids.demo.load_training_sample",
                return_value=(record, "neptune", "DoS"),
            ),
            patch(
                "nsl_kdd_ids.demo.predict_record",
                return_value={
                    "model_config": "baseline",
                    "decision_threshold": 0.5,
                    "threshold_semantics": "default",
                    "binary_prediction": "Attack",
                    "attack_family": "DoS",
                },
            ) as predict,
        ):
            result = analyze_training_sample("attack")

        self.assertEqual(len(predict.call_args.args[0]), 41)
        self.assertEqual(set(predict.call_args.args[0]), set(FEATURE_COLUMNS))
        predict.assert_called_once_with(record, model_config="baseline")
        self.assertEqual(result["predicted_class"], "DoS")
        self.assertEqual(result["actual_dataset_label"], "neptune")
        self.assertEqual(result["actual_category"], "DoS")
        self.assertTrue(result["is_correct"])

    def test_sample_result_reports_mismatch_from_labels(self) -> None:
        with (
            patch(
                "nsl_kdd_ids.demo.load_training_sample",
                return_value=(valid_payload(), "neptune", "DoS"),
            ),
            patch(
                "nsl_kdd_ids.demo.predict_record",
                return_value={
                    "model_config": "baseline",
                    "decision_threshold": 0.5,
                    "threshold_semantics": "default",
                    "binary_prediction": "Attack",
                    "attack_family": "Probe",
                },
            ),
        ):
            result = analyze_training_sample("attack")
        self.assertEqual(result["predicted_class"], "Probe")
        self.assertEqual(result["actual_dataset_label"], "neptune")
        self.assertFalse(result["is_correct"])

    def test_missing_training_data_and_models_have_actionable_errors(self) -> None:
        with patch(
            "nsl_kdd_ids.demo.dataset_dir",
            return_value=Path("missing-training-directory"),
        ):
            with self.assertRaisesRegex(DemoPrerequisiteError, "Add KDDTrain") as error:
                load_training_sample("normal")
        self.assertNotIn("missing-training-directory", str(error.exception))

        with (
            patch(
                "nsl_kdd_ids.demo.load_training_sample",
                return_value=(valid_payload(), "normal", "Normal"),
            ),
            patch(
                "nsl_kdd_ids.demo.predict_record",
                side_effect=FileNotFoundError("C:/private/path/model.joblib"),
            ),
        ):
            with self.assertRaisesRegex(DemoPrerequisiteError, "Generate them") as error:
                analyze_training_sample("normal")
        self.assertNotIn("C:/private/path", str(error.exception))


if __name__ == "__main__":
    unittest.main()
