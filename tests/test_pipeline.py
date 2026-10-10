import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from nsl_kdd_ids.pipeline import (
    ATTACK_FAMILIES,
    FEATURE_COLUMNS,
    _evaluate_predictions,
    _map_system_predictions,
    _metrics_match,
    _model_pipeline,
    _read_nsl_file,
    _threshold_predictions,
    _nested_group_audit,
    _validation_threshold,
    _select_binary_thresholds,
    _validate_model_feature_contract,
    _training_validation_indices,
    attack_model_dir,
    feature_dir,
    models_dir,
    predict_record,
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


class IntrusionEvaluationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_record() -> dict[str, object]:
        record: dict[str, object] = {feature: 0 for feature in FEATURE_COLUMNS}
        record.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
        return record

    def test_baseline_inference_matches_saved_models_and_is_deterministic(self) -> None:
        record = self._synthetic_record()
        result = predict_record(record)
        repeated = predict_record(record)
        binary_model = joblib.load(models_dir() / "random_forest.joblib")

        self.assertEqual(result, repeated)
        self.assertEqual(result["model_config"], "baseline")
        self.assertEqual(result["decision_threshold"], 0.5)
        model_prediction = int(
            binary_model.predict(
                pd.DataFrame([record])[list(binary_model.feature_names_in_)]
            )[0]
        )
        self.assertEqual(
            result["binary_prediction"], "Attack" if model_prediction else "Normal"
        )
        self.assertNotIn("probability", result)

    def test_saved_baseline_manifests_match_both_models(self) -> None:
        with (feature_dir() / "selected_features.json").open(encoding="utf-8") as stream:
            features = json.load(stream)["selected_features"]
        binary = joblib.load(models_dir() / "random_forest.joblib")
        family = joblib.load(attack_model_dir() / "random_forest_attack_type.joblib")
        self.assertEqual(list(binary.feature_names_in_), features)
        self.assertEqual(list(family.feature_names_in_), features)

    def test_candidate_models_match_their_own_manifest(self) -> None:
        candidate_dir = models_dir() / "controlled_experiment_grouped_validation"
        with (candidate_dir / "selected_features.json").open(encoding="utf-8") as stream:
            features = json.load(stream)["selected_features"]
        binary = joblib.load(candidate_dir / "binary_random_forest.joblib")
        family = joblib.load(candidate_dir / "attack_family_random_forest.joblib")
        self.assertEqual(list(binary.feature_names_in_), features)
        self.assertEqual(list(family.feature_names_in_), features)

    def test_candidate_inference_requires_explicit_threshold_and_own_manifest(self) -> None:
        record = self._synthetic_record()
        with self.assertRaisesRegex(ValueError, "explicit threshold"):
            predict_record(record, model_config="grouped_candidate")
        with self.assertRaisesRegex(ValueError, "numeric value"):
            predict_record(
                record, model_config="grouped_candidate", threshold=True
            )

        result = predict_record(
            record, model_config="grouped_candidate", threshold=0.31
        )
        self.assertEqual(result["model_config"], "grouped_candidate")
        self.assertEqual(result["decision_threshold"], 0.31)

    def test_inference_does_not_modify_saved_baseline_evaluation(self) -> None:
        paths = (
            Path("artifacts/evaluation/test_predictions.csv"),
            Path("artifacts/evaluation/complete_system_report.json"),
        )
        before = {
            path: hashlib.sha256(path.read_bytes()).digest()
            for path in paths
        }
        predict_record(self._synthetic_record())
        after = {
            path: hashlib.sha256(path.read_bytes()).digest()
            for path in paths
        }
        self.assertEqual(before, after)

    def test_inference_rejects_missing_extra_and_invalid_values(self) -> None:
        record = self._synthetic_record()
        with self.assertRaisesRegex(ValueError, "missing features"):
            predict_record({key: value for key, value in record.items() if key != "duration"})
        with self.assertRaisesRegex(ValueError, "unexpected features"):
            predict_record({**record, "unexpected": 1})
        for feature, invalid in (
            ("duration", True),
            ("duration", np.nan),
            ("service", 2),
            ("flag", ""),
        ):
            with self.subTest(feature=feature, invalid=invalid):
                with self.assertRaises(TypeError):
                    predict_record({**record, feature: invalid})

    def test_family_model_is_not_called_for_normal_inference(self) -> None:
        record = self._synthetic_record()
        binary_model = Mock()
        binary_model.predict.return_value = np.array([0])
        family_model = Mock()
        with patch(
            "nsl_kdd_ids.pipeline._load_inference_artifacts",
            return_value=(
                binary_model,
                family_model,
                ["duration"],
                0.5,
                "test threshold",
            ),
        ):
            result = predict_record(record)
        self.assertEqual(result["binary_prediction"], "Normal")
        self.assertIsNone(result["attack_family"])
        family_model.predict.assert_not_called()

    def test_family_model_runs_only_after_attack_detection(self) -> None:
        binary_model = Mock()
        binary_model.predict.return_value = np.array([1])
        family_model = Mock()
        family_model.predict.return_value = np.array(["Probe"])
        with patch(
            "nsl_kdd_ids.pipeline._load_inference_artifacts",
            return_value=(
                binary_model,
                family_model,
                ["duration"],
                0.5,
                "test threshold",
            ),
        ):
            result = predict_record(self._synthetic_record())
        self.assertEqual(result["binary_prediction"], "Attack")
        self.assertEqual(result["attack_family"], "Probe")
        family_model.predict.assert_called_once()

    def test_saved_metric_report_must_match_recomputed_predictions(self) -> None:
        expected = {
            "accuracy": 0.8,
            "macro avg": {"f1-score": 0.7, "support": 5},
        }
        self.assertTrue(_metrics_match(expected, {**expected, "extra_metric": 1.0}))
        self.assertFalse(
            _metrics_match(expected, {"accuracy": 0.8, "macro avg": {"f1-score": 0.6, "support": 5}})
        )
        self.assertFalse(_metrics_match({"missing": 1.0}, expected))

    def test_threshold_predictions_and_edge_values(self) -> None:
        probabilities = np.array([0.0, 0.4, 0.5, 1.0])
        np.testing.assert_array_equal(
            _threshold_predictions(probabilities, 0.5), [0, 0, 1, 1]
        )
        np.testing.assert_array_equal(
            _threshold_predictions(probabilities, 0.0), [1, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            _threshold_predictions(probabilities, 1.0), [0, 0, 0, 1]
        )

    def test_threshold_rejects_invalid_thresholds_and_probabilities(self) -> None:
        for threshold in (-0.1, 1.1, np.nan, np.inf, "invalid"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    _threshold_predictions(np.array([0.5]), threshold)
        for probabilities in (
            np.array([[0.5]]),
            np.array([np.nan]),
            np.array([-0.1]),
            np.array([1.1]),
        ):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    _threshold_predictions(probabilities, 0.5)

    def test_system_mapping_uses_family_only_for_predicted_attacks(self) -> None:
        np.testing.assert_array_equal(
            _map_system_predictions(
                np.array([0, 1, 1, 0]), np.array(["DoS", "U2R"])
            ),
            ["Normal", "DoS", "U2R", "Normal"],
        )
        with self.assertRaises(ValueError):
            _map_system_predictions(np.array([0, 1]), np.array(["DoS", "R2L"]))
        with self.assertRaises(ValueError):
            _map_system_predictions(np.array([0, 2]), np.array([]))

    def test_evaluation_counts_stage_failures_and_reports_confusion_matrices(self) -> None:
        test = pd.DataFrame(
            {
                "attack_label": ["normal", "neptune", "guess_passwd", "normal", "buffer_overflow"],
                "attack_category": ["Normal", "DoS", "R2L", "Normal", "U2R"],
                "is_attack": [0, 1, 1, 0, 1],
            }
        )
        report, binary_predictions, final_predictions = _evaluate_predictions(
            test,
            np.array([0.1, 0.9, 0.2, 0.8, 0.9]),
            np.array(["DoS", "DoS", "R2L", "Probe", "DoS"]),
            0.5,
        )

        self.assertEqual(report["attacks_missed_by_detector"], 1)
        self.assertEqual(report["attacks_detected_but_wrong_family"], 1)
        self.assertEqual(report["false_positives_among_normal"], 1)
        self.assertEqual(report["binary_intrusion_detection"]["false_positive_rate"], 0.5)
        self.assertEqual(report["binary_intrusion_detection"]["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(
            report["attack_family"]["confusion_matrix"],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
        )
        np.testing.assert_array_equal(binary_predictions, [0, 1, 0, 1, 1])
        np.testing.assert_array_equal(
            final_predictions, ["Normal", "DoS", "Normal", "Probe", "DoS"]
        )

    def test_validation_split_contains_only_training_rows_and_preserves_strata(self) -> None:
        categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        labels = [category for category in categories for _ in range(20)]
        attack_labels = [
            "normal" if category == "Normal" else f"{category}_{index // 5}"
            for category, index in (
                (category, index) for category in categories for index in range(20)
            )
        ]
        features = pd.DataFrame(
            {name: np.zeros(len(labels), dtype="int64") for name in FEATURE_COLUMNS}
        )
        features["duration"] = np.arange(len(labels))
        train = pd.DataFrame(
            {
                **{column: features[column] for column in FEATURE_COLUMNS},
                "attack_label": attack_labels,
                "attack_category": labels,
                "is_attack": [int(label != "Normal") for label in labels],
            }
        )
        fit_indices, validation_indices = _training_validation_indices(train)
        repeated_fit_indices, repeated_validation_indices = _training_validation_indices(
            train, random_state=42
        )
        alternate_fit_indices, alternate_validation_indices = (
            _training_validation_indices(
                train,
                random_state=17,
                allow_missing_classes=True,
                fold_index=1,
            )
        )

        self.assertEqual(set(fit_indices) & set(validation_indices), set())
        np.testing.assert_array_equal(fit_indices, repeated_fit_indices)
        np.testing.assert_array_equal(validation_indices, repeated_validation_indices)
        self.assertNotEqual(
            set(validation_indices), set(alternate_validation_indices)
        )
        self.assertEqual(
            set(fit_indices) | set(validation_indices), set(range(len(train)))
        )
        self.assertGreater(len(validation_indices), 0)
        self.assertLess(len(validation_indices), len(train))
        self.assertEqual(
            set(train.iloc[validation_indices]["attack_category"]),
            set(labels),
        )
        self.assertFalse(
            set(train.iloc[fit_indices]["attack_label"])
            & set(train.iloc[validation_indices]["attack_label"])
            - {"normal"}
        )
        fit_hashes = set(
            pd.util.hash_pandas_object(
                train.iloc[fit_indices][FEATURE_COLUMNS], index=False
            )
        )
        validation_hashes = set(
            pd.util.hash_pandas_object(
                train.iloc[validation_indices][FEATURE_COLUMNS], index=False
            )
        )
        self.assertFalse(fit_hashes & validation_hashes)
        alternate_fit_labels = set(train.iloc[alternate_fit_indices]["attack_label"])
        alternate_validation_labels = set(
            train.iloc[alternate_validation_indices]["attack_label"]
        )
        self.assertFalse(
            (alternate_fit_labels & alternate_validation_labels) - {"normal"}
        )
        alternate_fit_hashes = set(
            pd.util.hash_pandas_object(
                train.iloc[alternate_fit_indices][FEATURE_COLUMNS], index=False
            )
        )
        alternate_validation_hashes = set(
            pd.util.hash_pandas_object(
                train.iloc[alternate_validation_indices][FEATURE_COLUMNS],
                index=False,
            )
        )
        self.assertFalse(alternate_fit_hashes & alternate_validation_hashes)

    def test_validation_threshold_maximizes_system_macro_f1(self) -> None:
        class FixedBinaryModel:
            classes_ = np.array([0, 1])

            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                positive = features["probability"].to_numpy()
                return np.column_stack((1 - positive, positive))

        class FixedFamilyModel:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return features["family"].to_numpy()

        validation = pd.DataFrame(
            {
                "probability": [0.11, 0.40, 0.70, 0.90],
                "family": ["DoS", "DoS", "R2L", "U2R"],
                "attack_category": ["Normal", "DoS", "R2L", "U2R"],
                "is_attack": [0, 1, 1, 1],
                "attack_label": ["normal", "neptune", "guess_passwd", "buffer_overflow"],
            }
        )

        threshold, metrics = _validation_threshold(
            validation,
            FixedBinaryModel(),
            FixedFamilyModel(),
            ["probability", "family"],
        )

        self.assertEqual(threshold, 0.4)
        self.assertEqual(metrics["complete_system_macro_f1"], 0.8)

    def test_nested_binary_threshold_policies_use_inner_data_objectives(self) -> None:
        validation = pd.DataFrame(
            {
                "is_attack": [0, 0, 1, 1],
            }
        )
        selected = _select_binary_thresholds(
            validation, np.array([0.05, 0.4, 0.4, 0.8])
        )

        self.assertEqual(selected["fixed_0_5"]["threshold"], 0.5)
        self.assertEqual(selected["inner_binary_f1"]["threshold"], 0.4)
        self.assertEqual(selected["inner_binary_f1"]["inner_metrics"]["f1"], 0.8)
        self.assertEqual(selected["inner_recall_at_fpr_limit"]["threshold"], 0.8)
        self.assertEqual(
            selected["inner_recall_at_fpr_limit"]["inner_metrics"][
                "false_positive_rate"
            ],
            0.0,
        )
        with self.assertRaises(ValueError):
            _select_binary_thresholds(validation, np.array([0.05]), 0.01)
        with self.assertRaises(ValueError):
            _select_binary_thresholds(validation, np.array([0.05, 0.4, 0.4, 0.8]), 1.1)

    def test_nested_group_audit_detects_partition_group_overlap(self) -> None:
        frames = []
        for duration, label, category in (
            (0, "normal", "Normal"),
            (1, "neptune", "DoS"),
            (2, "ipsweep", "Probe"),
        ):
            row = {column: 0 for column in FEATURE_COLUMNS}
            row["duration"] = duration
            frames.append(
                pd.DataFrame(
                    [
                        {
                            **row,
                            "attack_label": label,
                            "attack_category": category,
                            "is_attack": int(label != "normal"),
                        }
                    ]
                )
            )
        audit = _nested_group_audit(frames[0], frames[1], frames[2])
        self.assertTrue(
            all(not overlap for overlap in audit["attack_name_overlap"].values())
        )
        self.assertTrue(
            all(
                overlap == 0
                for overlap in audit["shared_exact_full_feature_patterns"].values()
            )
        )

    def test_fitted_model_feature_contract_matches_manifest_and_order(self) -> None:
        features = ["service", "duration", "protocol_type"]
        training = pd.DataFrame(
            [
                ["http", 0, "tcp"],
                ["ftp", 1, "udp"],
                ["http", 1, "tcp"],
                ["ftp", 0, "udp"],
            ],
            columns=features,
        )
        binary_model = _model_pipeline(
            features, DecisionTreeClassifier(random_state=42)
        ).fit(training, [0, 1, 0, 1])
        family_model = _model_pipeline(
            features, DecisionTreeClassifier(random_state=42)
        ).fit(training, ["DoS", "Probe", "DoS", "Probe"])

        with tempfile.TemporaryDirectory() as directory:
            binary_path = Path(directory) / "binary.joblib"
            family_path = Path(directory) / "family.joblib"
            joblib.dump(binary_model, binary_path)
            joblib.dump(family_model, family_path)
            _validate_model_feature_contract(
                joblib.load(binary_path), joblib.load(family_path), features
            )
        with self.assertRaises(ValueError):
            _validate_model_feature_contract(binary_model, family_model, features[::-1])

    def test_preprocessing_uses_the_same_named_feature_order_at_inference(self) -> None:
        features = ["service", "duration", "protocol_type"]
        training = pd.DataFrame(
            [
                ["http", 0, "tcp"],
                ["ftp", 1, "udp"],
                ["http", 1, "tcp"],
                ["ftp", 0, "udp"],
            ],
            columns=features,
        )
        model = _model_pipeline(features, DecisionTreeClassifier(random_state=42))
        model.fit(training[features], [0, 1, 0, 1])

        reordered = training[["protocol_type", "duration", "service"]]
        expected = model.predict(training[features])
        actual = model.predict(reordered[features])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
