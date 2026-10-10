from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TOP_FEATURES = 20
KNN_SAMPLE_SIZE = 25_000
VALIDATION_SIZE = 0.2
THRESHOLD_CANDIDATES = np.round(np.linspace(0.05, 0.95, 91), 2)
REPEATED_VALIDATION_SPLITS = ((42, 1), (17, 2), (2025, 4))
NESTED_OUTER_FOLDS = 3
NESTED_INNER_FOLDS = 3
NESTED_OUTER_FOLD_INDICES = (0, 1, 2)
NESTED_INNER_FOLD_INDEX = 1
NESTED_MAX_FALSE_POSITIVE_RATE = 0.01
BINARY_LABELS = [0, 1]
FAMILY_LABELS = ["DoS", "Probe", "R2L", "U2R"]
SYSTEM_LABELS = ["Normal", *FAMILY_LABELS]

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
RAW_COLUMNS = [*FEATURE_COLUMNS, "attack_label", "difficulty"]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
LABEL_COLUMNS = ["attack_label", "attack_category", "is_attack"]

ATTACK_FAMILIES = {
    "normal": "Normal",
    "apache2": "DoS",
    "back": "DoS",
    "land": "DoS",
    "mailbomb": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "processtable": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "httptunnel": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "named": "R2L",
    "phf": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "ps": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def project_root() -> Path:
    current = Path.cwd()
    return current.parent if current.name == "notebooks" else current


def dataset_dir() -> Path:
    return project_root() / "dataset" / "NSL-KDD"


def processed_dir() -> Path:
    return project_root() / "dataset" / "processed"


def feature_dir() -> Path:
    return project_root() / "artifacts" / "feature_selection"


def models_dir() -> Path:
    return project_root() / "artifacts" / "models"


def attack_model_dir() -> Path:
    return project_root() / "artifacts" / "attack_type_model"


def evaluation_dir() -> Path:
    return project_root() / "artifacts" / "evaluation"


def explainability_dir() -> Path:
    return project_root() / "artifacts" / "explainability"


def _require_file(path: Path, instruction: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}\n{instruction}")


def _read_nsl_file(path: Path) -> pd.DataFrame:
    _require_file(
        path,
        "Place KDDTrain+.txt and KDDTest+.txt in dataset/NSL-KDD, then rerun.",
    )
    data = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    if len(data.columns) != 43 or data.isna().any().any():
        raise ValueError(f"{path.name} does not have the expected 43 clean columns.")

    data["attack_label"] = (
        data["attack_label"].astype("string").str.strip().str.lower().str.rstrip(".")
    )
    unknown = sorted(set(data["attack_label"]) - set(ATTACK_FAMILIES))
    if unknown:
        raise ValueError(f"Unmapped NSL-KDD labels in {path.name}: {unknown}")

    data.drop(columns="difficulty", inplace=True)
    data["attack_category"] = data["attack_label"].map(ATTACK_FAMILIES)
    data["is_attack"] = (data["attack_label"] != "normal").astype("int8")
    return data[[*FEATURE_COLUMNS, *LABEL_COLUMNS]]


def inspect_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = []
    for split, name in (("train", "KDDTrain+.txt"), ("test", "KDDTest+.txt")):
        data = _read_nsl_file(dataset_dir() / name)
        counts = data["attack_category"].value_counts().rename_axis("class").to_frame(split)
        splits.append(counts)
        print(f"{split}: {len(data):,} records, {len(FEATURE_COLUMNS)} input features")
    distribution = pd.concat(splits, axis=1).fillna(0).astype("int64")
    return distribution, data.head()


def prepare_data() -> dict[str, Any]:
    train = _read_nsl_file(dataset_dir() / "KDDTrain+.txt")
    test = _read_nsl_file(dataset_dir() / "KDDTest+.txt")
    processed_dir().mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir() / "train.parquet", index=False)
    test.to_parquet(processed_dir() / "test.parquet", index=False)

    report = {
        "dataset": "NSL-KDD",
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "train_attack_categories": {
            str(key): int(value)
            for key, value in train["attack_category"].value_counts().items()
        },
        "test_attack_categories": {
            str(key): int(value)
            for key, value in test["attack_category"].value_counts().items()
        },
        "train_attack_labels": {
            str(key): int(value)
            for key, value in train["attack_label"].value_counts().items()
        },
        "test_attack_labels": {
            str(key): int(value)
            for key, value in test["attack_label"].value_counts().items()
        },
        "difficulty_used_as_feature": False,
    }
    (processed_dir() / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir() / "train.parquet"
    test_path = processed_dir() / "test.parquet"
    _require_file(
        train_path,
        "Run notebook 01 or `python -m nsl_kdd_ids.pipeline --step prepare` first.",
    )
    _require_file(
        test_path,
        "Run notebook 01 or `python -m nsl_kdd_ids.pipeline --step prepare` first.",
    )
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def _load_processed_training() -> pd.DataFrame:
    train_path = processed_dir() / "train.parquet"
    _require_file(
        train_path,
        "Run notebook 01 or `python -m nsl_kdd_ids.pipeline --step prepare` first.",
    )
    return pd.read_parquet(train_path)


def _selected_features() -> list[str]:
    path = feature_dir() / "selected_features.json"
    _require_file(
        path,
        "Run notebook 02 or `python -m nsl_kdd_ids.pipeline --step select` first.",
    )
    selected = json.loads(path.read_text(encoding="utf-8"))["selected_features"]
    if not selected or not set(selected).issubset(FEATURE_COLUMNS):
        raise ValueError(f"Invalid selected feature list in {path}")
    return selected


def _build_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in CATEGORICAL_FEATURES]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _aggregate_importances(
    model: Pipeline, features: list[str]
) -> pd.Series:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    scores: dict[str, float] = {}
    for feature in features:
        positions = [
            index
            for index, name in enumerate(names)
            if name == f"numeric__{feature}"
            or name == f"categorical__{feature}"
            or name.startswith(f"categorical__{feature}_")
        ]
        scores[feature] = float(importances[positions].sum()) if positions else 0.0
    return pd.Series(scores, dtype="float64").sort_values(ascending=False)


def _select_top_features(train: pd.DataFrame) -> tuple[list[str], pd.Series]:
    selector = Pipeline(
        [
            ("preprocessor", _build_preprocessor(FEATURE_COLUMNS)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    selector.fit(train[FEATURE_COLUMNS], train["is_attack"])
    importance = _aggregate_importances(selector, FEATURE_COLUMNS)
    selected = importance.head(min(TOP_FEATURES, len(importance))).index.tolist()
    return selected, importance


def select_features() -> pd.DataFrame:
    train, _ = _load_processed()
    selected, importance = _select_top_features(train)

    feature_dir().mkdir(parents=True, exist_ok=True)
    importance.rename("importance").to_csv(feature_dir() / "feature_scores.csv")
    report = {
        "dataset": "NSL-KDD",
        "method": "Random Forest feature importance aggregated to the original feature",
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "feature_importance": {key: float(value) for key, value in importance.items()},
    }
    (feature_dir() / "selected_features.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return importance.rename("importance").to_frame()


def _model_pipeline(features: list[str], classifier: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _build_preprocessor(features)),
            ("classifier", classifier),
        ]
    )


def _validate_model_feature_contract(
    binary_model: Pipeline, family_model: Pipeline, features: list[str]
) -> None:
    if not features or len(set(features)) != len(features):
        raise ValueError("Model feature contract requires unique, non-empty features.")
    for name, model in (("binary", binary_model), ("family", family_model)):
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features != features:
            raise ValueError(
                f"Saved {name} model feature order does not match its feature manifest."
            )


def _load_inference_artifacts(
    model_config: str, threshold: float | None
) -> tuple[Pipeline, Pipeline, list[str], float, str]:
    if model_config == "baseline":
        if threshold is not None:
            raise ValueError("The baseline uses its saved estimator's default predict() decision.")
        model_root = models_dir()
        family_root = attack_model_dir()
        binary_path = model_root / "random_forest.joblib"
        family_path = family_root / "random_forest_attack_type.joblib"
        manifest_path = feature_dir() / "selected_features.json"
        decision_threshold = 0.5
        threshold_semantics = (
            "Informational 0.5 boundary; binary decisions use the estimator's "
            "default predict(), including its tie behavior."
        )
    elif model_config == "grouped_candidate":
        if threshold is None:
            raise ValueError("The grouped_candidate configuration requires an explicit threshold.")
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a finite numeric value between 0 and 1.")
        _threshold_predictions(np.array([], dtype="float64"), threshold)
        model_root = models_dir() / "controlled_experiment_grouped_validation"
        binary_path = model_root / "binary_random_forest.joblib"
        family_path = model_root / "attack_family_random_forest.joblib"
        manifest_path = model_root / "selected_features.json"
        decision_threshold = float(threshold)
        threshold_semantics = (
            "predict_proba positive-class output >= caller-supplied cutoff; "
            "not a calibrated real-world risk."
        )
    else:
        raise ValueError(
            "model_config must be 'baseline' or 'grouped_candidate'."
        )

    for path in (binary_path, family_path, manifest_path):
        _require_file(path, f"Required artifacts for model configuration {model_config!r} are missing.")
    with manifest_path.open(encoding="utf-8") as stream:
        features = json.load(stream)["selected_features"]
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(feature, str) for feature in features)
        or len(set(features)) != len(features)
        or not set(features).issubset(FEATURE_COLUMNS)
    ):
        raise ValueError(f"Invalid ordered feature manifest for {model_config!r}.")

    binary_model = joblib.load(binary_path)
    family_model = joblib.load(family_path)
    _validate_model_feature_contract(binary_model, family_model, features)
    return (
        binary_model,
        family_model,
        features,
        decision_threshold,
        threshold_semantics,
    )


def predict_record(
    record: Mapping[str, Any],
    model_config: str = "baseline",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict one raw NSL-KDD record with an explicitly named saved model set."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of all 41 NSL-KDD feature names to values.")
    missing = sorted(set(FEATURE_COLUMNS) - set(record))
    extra = sorted(set(record) - set(FEATURE_COLUMNS))
    if missing or extra:
        raise ValueError(f"Invalid record schema; missing features: {missing}; unexpected features: {extra}.")
    for feature in FEATURE_COLUMNS:
        value = record[feature]
        if feature in CATEGORICAL_FEATURES:
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{feature} must be a non-empty string.")
        elif (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise TypeError(f"{feature} must be a finite numeric scalar.")

    binary_model, family_model, features, decision_threshold, semantics = (
        _load_inference_artifacts(model_config, threshold)
    )
    row = pd.DataFrame([record], columns=FEATURE_COLUMNS)[features]
    if model_config == "baseline":
        binary_prediction = int(binary_model.predict(row)[0])
    else:
        positive_score = _positive_probabilities(binary_model, row)
        binary_prediction = int(
            _threshold_predictions(positive_score, decision_threshold)[0]
        )
    if binary_prediction not in BINARY_LABELS:
        raise ValueError(f"Binary model returned an unsupported prediction: {binary_prediction!r}.")

    family_prediction = str(family_model.predict(row)[0]) if binary_prediction == 1 else None
    if family_prediction is not None and family_prediction not in FAMILY_LABELS:
        raise ValueError(
            f"Family model returned an unsupported prediction: {family_prediction!r}."
        )
    return {
        "model_config": model_config,
        "decision_threshold": decision_threshold,
        "threshold_semantics": semantics,
        "binary_prediction": "Attack" if binary_prediction else "Normal",
        "attack_family": family_prediction,
    }


def _binary_metrics(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    false_positive_rate = float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "false_positive_rate": false_positive_rate,
    }


def _threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isscalar(threshold):
        raise ValueError("threshold must be a finite number between 0 and 1.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("threshold must be a finite number between 0 and 1.") from error
    if not np.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1.")
    scores = np.asarray(probabilities, dtype="float64")
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be a one-dimensional array in [0, 1].")
    return (scores >= threshold_value).astype("int8")


def _positive_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Binary model must expose exactly one positive class labeled 1.")
    return model.predict_proba(features)[:, int(positive_columns[0])]


def _map_system_predictions(
    binary_predictions: np.ndarray, family_predictions: np.ndarray
) -> np.ndarray:
    binary = np.asarray(binary_predictions)
    families = np.asarray(family_predictions)
    if binary.ndim != 1 or not np.isin(binary, BINARY_LABELS).all():
        raise ValueError("binary_predictions must be a one-dimensional array of 0/1 values.")
    attack_count = int((binary == 1).sum())
    if families.ndim != 1 or len(families) != attack_count:
        raise ValueError("family_predictions must contain one result per predicted attack.")
    if not set(families).issubset(FAMILY_LABELS):
        raise ValueError("family_predictions contains an unknown attack family.")

    final_predictions = np.full(len(binary), "Normal", dtype=object)
    final_predictions[np.flatnonzero(binary == 1)] = families
    return final_predictions


def _training_validation_indices(
    train: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    allow_missing_classes: bool = False,
    fold_index: int | None = None,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = [column for column in FEATURE_COLUMNS if column in train]
    if not feature_columns or "attack_label" not in train:
        raise ValueError("Grouped validation requires input features and attack_label.")
    if fold_index is not None and (
        not isinstance(fold_index, int) or not 0 <= fold_index < n_splits
    ):
        raise ValueError(f"fold_index must be an integer from 0 through {n_splits - 1}.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    parent = np.arange(len(train))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_attack: dict[str, int] = {}
    first_by_features: dict[int, int] = {}
    feature_hashes = pd.util.hash_pandas_object(train[feature_columns], index=False).to_numpy()
    labels = train["attack_label"].astype(str).to_numpy()
    for index, (label, feature_hash) in enumerate(zip(labels, feature_hashes)):
        if label != "normal":
            if label in first_by_attack:
                union(index, first_by_attack[label])
            else:
                first_by_attack[label] = index
        feature_key = int(feature_hash)
        if feature_key in first_by_features:
            union(index, first_by_features[feature_key])
        else:
            first_by_features[feature_key] = index

    groups = pd.factorize(np.asarray([find(index) for index in range(len(train))]))[0]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score: tuple[int, int, float, int] | None = None
    fallback_split: tuple[np.ndarray, np.ndarray] | None = None
    fallback_score: tuple[int, int, int, float, int] | None = None
    for fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train[feature_columns], train["attack_category"], groups)
    ):
        if fold == fold_index:
            return fit_indices, validation_indices
        fit_support = train.iloc[fit_indices]["attack_category"].value_counts()
        validation_support = train.iloc[validation_indices]["attack_category"].value_counts()
        if any(
            fit_support.get(category, 0) == 0
            or validation_support.get(category, 0) == 0
            for category in SYSTEM_LABELS
        ):
            if allow_missing_classes:
                represented_classes = sum(
                    fit_support.get(category, 0) > 0
                    and validation_support.get(category, 0) > 0
                    for category in SYSTEM_LABELS
                )
                present_categories = [
                    category
                    for category in SYSTEM_LABELS
                    if fit_support.get(category, 0) > 0
                    and validation_support.get(category, 0) > 0
                ]
                minimum_validation_support = min(
                    (int(validation_support[category]) for category in present_categories),
                    default=0,
                )
                minimum_fit_support = min(
                    (int(fit_support[category]) for category in present_categories),
                    default=0,
                )
                distance_from_target = abs(
                    len(validation_indices) / len(train) - VALIDATION_SIZE
                )
                score_with_missing = (
                    represented_classes,
                    minimum_validation_support,
                    minimum_fit_support,
                    -distance_from_target,
                    -fold,
                )
                if fallback_score is None or score_with_missing > fallback_score:
                    fallback_score = score_with_missing
                    fallback_split = fit_indices, validation_indices
            continue
        minimum_validation_support = min(
            int(validation_support.get(category, 0)) for category in SYSTEM_LABELS
        )
        minimum_fit_support = min(int(fit_support.get(category, 0)) for category in SYSTEM_LABELS)
        distance_from_target = abs(len(validation_indices) / len(train) - VALIDATION_SIZE)
        score = (
            minimum_validation_support,
            minimum_fit_support,
            -distance_from_target,
            -fold,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_split = fit_indices, validation_indices
    if best_split is None:
        if allow_missing_classes and fallback_split is not None:
            return fallback_split
        raise ValueError(
            "Cannot create grouped validation with every system class represented in fit and validation."
        )
    return best_split


def _experiment_binary_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _experiment_family_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sample_standard_deviation": None}
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
    }


def _select_binary_thresholds(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    max_false_positive_rate: float = NESTED_MAX_FALSE_POSITIVE_RATE,
) -> dict[str, dict[str, Any]]:
    if (
        not np.isfinite(max_false_positive_rate)
        or not 0 <= max_false_positive_rate <= 1
    ):
        raise ValueError("max_false_positive_rate must be finite and in [0, 1].")
    scores = np.asarray(probabilities, dtype="float64")
    if scores.ndim != 1 or len(scores) != len(validation):
        raise ValueError("probabilities must be one-dimensional and match validation rows.")
    actual = validation["is_attack"].to_numpy(dtype="int8")
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("Inner validation must contain both normal and attack rows.")

    def metrics_at(threshold: float) -> dict[str, float]:
        predicted = _threshold_predictions(scores, threshold)
        matrix = confusion_matrix(actual, predicted, labels=BINARY_LABELS)
        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "false_positive_rate": float(matrix[0, 1] / matrix[0].sum()),
        }

    candidates = sorted(set(THRESHOLD_CANDIDATES.tolist()) | {1.0})
    evaluated = [(float(threshold), metrics_at(float(threshold))) for threshold in candidates]
    best_f1_threshold, best_f1_metrics = max(
        evaluated, key=lambda pair: (pair[1]["f1"], pair[0])
    )
    feasible = [
        (threshold, metrics)
        for threshold, metrics in evaluated
        if metrics["false_positive_rate"] <= max_false_positive_rate
    ]
    if not feasible:
        raise ValueError(
            "No threshold candidate satisfies the requested validation false-positive constraint."
        )
    best_fpr_threshold, best_fpr_metrics = max(
        feasible, key=lambda pair: (pair[1]["recall"], pair[0])
    )
    return {
        "fixed_0_5": {"threshold": 0.5, "inner_metrics": metrics_at(0.5)},
        "inner_binary_f1": {
            "threshold": best_f1_threshold,
            "inner_metrics": best_f1_metrics,
            "selection_objective": "maximize binary attack F1; ties choose higher threshold",
        },
        "inner_recall_at_fpr_limit": {
            "threshold": best_fpr_threshold,
            "inner_metrics": best_fpr_metrics,
            "selection_objective": (
                f"maximize binary attack recall subject to inner validation FPR <= "
                f"{max_false_positive_rate:.3f}; ties choose higher threshold"
            ),
            "maximum_false_positive_rate": max_false_positive_rate,
        },
    }


def _nested_group_audit(
    fit: pd.DataFrame, inner_validation: pd.DataFrame, outer_validation: pd.DataFrame
) -> dict[str, Any]:
    def attack_names(frame: pd.DataFrame) -> set[str]:
        return set(frame.loc[frame["is_attack"] == 1, "attack_label"].astype(str))

    def feature_hashes(frame: pd.DataFrame) -> set[int]:
        return set(
            pd.util.hash_pandas_object(frame[FEATURE_COLUMNS], index=False)
            .to_numpy()
            .tolist()
        )

    partitions = {
        "fit": fit,
        "inner_validation": inner_validation,
        "outer_validation": outer_validation,
    }
    name_sets = {key: attack_names(frame) for key, frame in partitions.items()}
    hash_sets = {key: feature_hashes(frame) for key, frame in partitions.items()}
    return {
        "attack_name_overlap": {
            f"{left}_and_{right}": sorted(name_sets[left] & name_sets[right])
            for left, right in (
                ("fit", "inner_validation"),
                ("fit", "outer_validation"),
                ("inner_validation", "outer_validation"),
            )
        },
        "shared_exact_full_feature_patterns": {
            f"{left}_and_{right}": len(hash_sets[left] & hash_sets[right])
            for left, right in (
                ("fit", "inner_validation"),
                ("fit", "outer_validation"),
                ("inner_validation", "outer_validation"),
            )
        },
        "class_support": {
            key: {label: int(frame["attack_category"].value_counts().get(label, 0)) for label in SYSTEM_LABELS}
            for key, frame in partitions.items()
        },
    }


def run_nested_grouped_validation() -> dict[str, Any]:
    train = _load_processed_training()
    repetitions: list[dict[str, Any]] = []
    output_dir = evaluation_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_outer_splits = [
        _training_validation_indices(
            train,
            random_state=RANDOM_STATE,
            allow_missing_classes=True,
            fold_index=fold_index,
            n_splits=NESTED_OUTER_FOLDS,
        )
        for fold_index in NESTED_OUTER_FOLD_INDICES
    ]

    for outer_fold, (outer_fit_indices, outer_validation_indices) in enumerate(
        grouped_outer_splits
    ):
        outer_fit = train.iloc[outer_fit_indices]
        outer_validation = train.iloc[outer_validation_indices]
        inner_fit_indices, inner_validation_indices = _training_validation_indices(
            outer_fit,
            random_state=RANDOM_STATE,
            allow_missing_classes=True,
            fold_index=NESTED_INNER_FOLD_INDEX,
            n_splits=NESTED_INNER_FOLDS,
        )
        inner_fit = outer_fit.iloc[inner_fit_indices]
        inner_validation = outer_fit.iloc[inner_validation_indices]
        inner_features, _ = _select_top_features(inner_fit)
        inner_binary = _model_pipeline(
            inner_features, _experiment_binary_classifier()
        )
        inner_binary.fit(inner_fit[inner_features], inner_fit["is_attack"])
        inner_probabilities = _positive_probabilities(
            inner_binary, inner_validation[inner_features]
        )
        policies = _select_binary_thresholds(
            inner_validation, inner_probabilities
        )

        outer_features, _ = _select_top_features(outer_fit)
        outer_binary = _model_pipeline(
            outer_features, _experiment_binary_classifier()
        )
        outer_family = _model_pipeline(
            outer_features, _experiment_family_classifier()
        )
        outer_family_fit = outer_fit.loc[outer_fit["is_attack"] == 1]
        outer_binary.fit(outer_fit[outer_features], outer_fit["is_attack"])
        outer_family.fit(
            outer_family_fit[outer_features], outer_family_fit["attack_category"]
        )
        outer_probabilities = _positive_probabilities(
            outer_binary, outer_validation[outer_features]
        )
        outer_families = outer_family.predict(outer_validation[outer_features])
        outer_reports: dict[str, Any] = {}
        for policy_name, policy in policies.items():
            outer_reports[policy_name] = _evaluate_predictions(
                outer_validation,
                outer_probabilities,
                outer_families,
                float(policy["threshold"]),
            )[0]

        audit = _nested_group_audit(
            inner_fit, inner_validation, outer_validation
        )
        held_out_outer = sorted(
            set(outer_validation.loc[outer_validation["is_attack"] == 1, "attack_label"])
            - set(outer_fit.loc[outer_fit["is_attack"] == 1, "attack_label"])
        )
        repetitions.append(
            {
                "outer_fold": outer_fold,
                "outer_fit_rows": len(outer_fit),
                "inner_fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_validation_rows": len(outer_validation),
                "inner_threshold_policies": policies,
                "outer_held_out_attack_names": held_out_outer,
                "grouping_audit": audit,
                "outer_results": outer_reports,
                "missing_outer_validation_classes": [
                    label
                    for label in SYSTEM_LABELS
                    if audit["class_support"]["outer_validation"][label] == 0
                ],
                "missing_inner_validation_classes": [
                    label
                    for label in SYSTEM_LABELS
                    if audit["class_support"]["inner_validation"][label] == 0
                ],
            }
        )

    policy_names = ("fixed_0_5", "inner_binary_f1", "inner_recall_at_fpr_limit")
    aggregate: dict[str, Any] = {}
    requested_metrics = {
        "binary_precision": lambda report: report["binary_intrusion_detection"]["precision"],
        "binary_recall": lambda report: report["binary_intrusion_detection"]["recall"],
        "binary_f1": lambda report: report["binary_intrusion_detection"]["f1"],
        "binary_roc_auc": lambda report: report["binary_intrusion_detection"]["roc_auc"],
        "binary_false_positive_rate": lambda report: report["binary_intrusion_detection"]["false_positive_rate"],
        "system_accuracy": lambda report: report["complete_system"]["metrics"]["accuracy"],
        "system_macro_f1": lambda report: report["complete_system"]["metrics"]["macro avg"]["f1-score"],
        "system_weighted_f1": lambda report: report["complete_system"]["metrics"]["weighted avg"]["f1-score"],
    }
    for policy_name in policy_names:
        reports = [fold["outer_results"][policy_name] for fold in repetitions]
        aggregate[policy_name] = {
            metric: _metric_summary([float(getter(report)) for report in reports])
            for metric, getter in requested_metrics.items()
        }
        for family in ("R2L", "U2R"):
            family_metrics = [
                report["complete_system"]["metrics"][family] for report in reports
            ]
            for key in ("precision", "recall", "f1-score", "support"):
                aggregate[policy_name][f"{family}_{key}"] = _metric_summary(
                    [float(metrics[key]) for metrics in family_metrics]
                )
        if policy_name != "fixed_0_5":
            aggregate[policy_name]["selected_threshold"] = _metric_summary(
                [
                    float(fold["inner_threshold_policies"][policy_name]["threshold"])
                    for fold in repetitions
                ]
            )

    report: dict[str, Any] = {
        "dataset": "KDDTrain+ processed training rows only",
        "official_test_data_loaded_or_used": False,
        "model_settings": {
            "binary": {
                "n_estimators": 100,
                "max_depth": 20,
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
            },
            "family": {
                "n_estimators": 150,
                "max_depth": 25,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "protocol": {
            "outer": (
                "Three-fold StratifiedGroupKFold, random_state 42. Each fold is "
                "evaluated once; groups connect all rows of an attack name and all "
                "identical full-feature rows. Outer fold 0 has no Probe examples."
            ),
            "inner": (
                "Within each outer training partition, select the prespecified "
                "grouped fold 1 of 3 for threshold selection. Inner validation always "
                "contains normal and attack rows, but may omit attack families; "
                "therefore threshold objectives are binary, not five-class macro F1."
            ),
            "threshold_objectives": {
                "fixed_0_5": "No selection; fixed probability threshold 0.5.",
                "inner_binary_f1": "Maximize inner validation binary attack F1; ties choose higher threshold.",
                "inner_recall_at_fpr_limit": (
                    "Maximize inner validation attack recall subject to FPR <= 1%; "
                    "ties choose higher threshold."
                ),
            },
            "fit_boundaries": (
                "Inner feature selection, preprocessing and binary model fit on "
                "inner-fit only. Outer feature selection, preprocessing, binary "
                "model and family model fit on outer-fit only. Outer labels are used "
                "by StratifiedGroupKFold for fold assignment and then for reporting, "
                "never for feature, model, or threshold selection."
            ),
            "limitations": [
                "The giant duplicate-linked attack-name groups make full five-class coverage impossible in every outer or inner grouped fold.",
                "Only two outer folds contain every system category; the third is retained and Probe is reported as unavailable rather than dropped.",
                "R2L/U2R support is highly uneven across outer folds, so aggregated rare-family metrics have substantial uncertainty.",
                "These results estimate performance on held-out NSL-KDD attack-name groups only, not real-world attack generalization.",
                "The saved production evaluation path still uses baseline models and default .predict(); candidate artifacts and tuned threshold have not been promoted.",
            ],
            "official_test_labels_used_for_selection": False,
        },
        "outer_folds": repetitions,
        "aggregate_mean_and_sample_standard_deviation": aggregate,
        "reproduction_command": (
            ".\\.venv\\Scripts\\python.exe -m nsl_kdd_ids.pipeline "
            "--step nested-validation"
        ),
    }
    (output_dir / "nested_grouped_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for fold in repetitions:
        for policy_name in policy_names:
            result = fold["outer_results"][policy_name]
            system_metrics = result["complete_system"]["metrics"]
            binary_metrics = result["binary_intrusion_detection"]
            for label in SYSTEM_LABELS:
                family_metrics = system_metrics[label]
                rows.append(
                    {
                        "outer_fold": fold["outer_fold"],
                        "policy": policy_name,
                        "inner_selected_threshold": (
                            0.5
                            if policy_name == "fixed_0_5"
                            else fold["inner_threshold_policies"][policy_name]["threshold"]
                        ),
                        "binary_precision": binary_metrics["precision"],
                        "binary_recall": binary_metrics["recall"],
                        "binary_f1": binary_metrics["f1"],
                        "binary_roc_auc": binary_metrics["roc_auc"],
                        "binary_false_positive_rate": binary_metrics["false_positive_rate"],
                        "system_accuracy": system_metrics["accuracy"],
                        "system_macro_f1": system_metrics["macro avg"]["f1-score"],
                        "system_weighted_f1": system_metrics["weighted avg"]["f1-score"],
                        "class": label,
                        "class_precision": family_metrics["precision"],
                        "class_recall": family_metrics["recall"],
                        "class_f1": family_metrics["f1-score"],
                        "class_support": int(family_metrics["support"]),
                        "missing_outer_classes": ",".join(fold["missing_outer_validation_classes"]),
                    }
                )
    pd.DataFrame(rows).to_csv(
        output_dir / "nested_grouped_validation_metrics.csv", index=False
    )
    markdown = [
        "# Nested attack-name-grouped model-selection audit",
        "",
        "Only processed KDDTrain+ was loaded. KDDTest+ was not loaded or used.",
        "Outer-fold rows are held out from model and threshold selection. Labels "
        "are used by the stratified grouping algorithm to construct folds and then "
        "for reporting.",
        "",
        "## Outer-fold results",
        "",
        "| Fold | Policy | Inner threshold | Binary precision | Recall | F1 | ROC-AUC | FPR | System accuracy | Macro F1 | Weighted F1 | R2L P/R/F1 (support) | U2R P/R/F1 (support) | Missing classes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for fold in repetitions:
        for policy_name in policy_names:
            result = fold["outer_results"][policy_name]
            binary = result["binary_intrusion_detection"]
            system = result["complete_system"]["metrics"]
            threshold = fold["inner_threshold_policies"][policy_name]["threshold"]
            r2l, u2r = system["R2L"], system["U2R"]
            markdown.append(
                f"| {fold['outer_fold']} | {policy_name} | {threshold:.2f} "
                f"| {binary['precision']:.4f} | {binary['recall']:.4f} "
                f"| {binary['f1']:.4f} | {binary['roc_auc']:.4f} "
                f"| {binary['false_positive_rate']:.4f} "
                f"| {system['accuracy']:.4f} | {system['macro avg']['f1-score']:.4f} "
                f"| {system['weighted avg']['f1-score']:.4f} "
                f"| {r2l['precision']:.4f}/{r2l['recall']:.4f}/{r2l['f1-score']:.4f} ({int(r2l['support'])}) "
                f"| {u2r['precision']:.4f}/{u2r['recall']:.4f}/{u2r['f1-score']:.4f} ({int(u2r['support'])}) "
                f"| {', '.join(fold['missing_outer_validation_classes']) or 'None'} |"
            )
    markdown.extend(["", "## Aggregate mean ± sample SD", "", "| Policy | Metric | Mean | Sample SD |", "|---|---|---:|---:|"])
    for policy_name in policy_names:
        for metric, values in aggregate[policy_name].items():
            markdown.append(
                f"| {policy_name} | {metric} | {values['mean']:.4f} | "
                f"{values['sample_standard_deviation']:.4f} |"
            )
    markdown.extend(
        [
            "",
            "## Protocol and limitations",
            "",
            "The inner threshold objectives are binary because grouping prevents "
            "all attack families from appearing in every inner validation partition. "
            "The 1% FPR limit is an inner-validation constraint only. Outer labels "
            "are used for stratified fold assignment and reporting, never threshold "
            "or model selection. The third outer fold has no Probe "
            "examples; its fixed-label macro F1 includes Probe as zero and should "
            "not be treated as a complete five-class estimate.",
            "Model settings are fixed. Feature selection and preprocessing are "
            "re-fitted only on the corresponding inner-fit or outer-fit data. Exact "
            "attack-name and full-feature grouping audits are included in JSON.",
            "The current production evaluator continues to load baseline models "
            "and uses their default .predict() decision. The reliable candidate's "
            "separate model files and 0.28 threshold do not affect production "
            "evaluation. Do not promote a candidate until a production inference "
            "artifact contract (model, ordered features, and threshold) is explicitly "
            "implemented and verified.",
            "Results are specific to held-out NSL-KDD groups and do not establish "
            "generalization to real-world attacks.",
            "",
            f"Reproduce with: `{report['reproduction_command']}`",
            "",
        ]
    )
    (output_dir / "nested_grouped_validation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report


def run_repeated_grouped_validation() -> dict[str, Any]:
    train = _load_processed_training()
    evaluation_dir().mkdir(parents=True, exist_ok=True)
    repetitions: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for seed, fold_index in REPEATED_VALIDATION_SPLITS:
        fit_indices, validation_indices = _training_validation_indices(
            train,
            random_state=seed,
            allow_missing_classes=True,
            fold_index=fold_index,
        )
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        features, _ = _select_top_features(fit)
        binary_model = _model_pipeline(features, _experiment_binary_classifier())
        family_model = _model_pipeline(features, _experiment_family_classifier())
        family_fit = fit.loc[fit["is_attack"] == 1]
        binary_model.fit(fit[features], fit["is_attack"])
        family_model.fit(family_fit[features], family_fit["attack_category"])

        probabilities = _positive_probabilities(binary_model, validation[features])
        family_predictions = family_model.predict(validation[features])
        threshold, threshold_metrics = _validation_threshold(
            validation, binary_model, family_model, features
        )
        baseline_report, _, _ = _evaluate_predictions(
            validation, probabilities, family_predictions, 0.5
        )
        candidate_report, _, _ = _evaluate_predictions(
            validation, probabilities, family_predictions, threshold
        )

        fit_labels = set(fit["attack_label"])
        validation_labels = set(validation["attack_label"])
        fit_hashes = set(
            pd.util.hash_pandas_object(fit[FEATURE_COLUMNS], index=False).to_numpy()
        )
        validation_hashes = set(
            pd.util.hash_pandas_object(
                validation[FEATURE_COLUMNS], index=False
            ).to_numpy()
        )
        fit_support = fit["attack_category"].value_counts()
        validation_support = validation["attack_category"].value_counts()
        missing_fit = [
            label for label in SYSTEM_LABELS if fit_support.get(label, 0) == 0
        ]
        missing_validation = [
            label
            for label in SYSTEM_LABELS
            if validation_support.get(label, 0) == 0
        ]
        split_record = {
            "seed": seed,
            "fold_index": fold_index,
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "validation_threshold": threshold,
            "threshold_selection_metrics": threshold_metrics,
            "grouping_audit": {
                "held_out_attack_names": sorted(
                    (validation_labels - fit_labels) - {"normal"}
                ),
                "attack_name_overlap": sorted(
                    (validation_labels & fit_labels) - {"normal"}
                ),
                "shared_exact_full_feature_patterns": len(fit_hashes & validation_hashes),
                "missing_fit_classes": missing_fit,
                "missing_validation_classes": missing_validation,
                "fit_class_support": {
                    label: int(fit_support.get(label, 0)) for label in SYSTEM_LABELS
                },
                "validation_class_support": {
                    label: int(validation_support.get(label, 0))
                    for label in SYSTEM_LABELS
                },
            },
            "baseline_default_threshold": baseline_report,
            "candidate_validation_selected_threshold": candidate_report,
        }
        repetitions.append(split_record)

        for strategy, report, applied_threshold in (
            ("baseline_default_threshold", baseline_report, 0.5),
            ("candidate_validation_selected_threshold", candidate_report, threshold),
        ):
            binary = report["binary_intrusion_detection"]
            system = report["complete_system"]["metrics"]
            for label in SYSTEM_LABELS:
                class_metrics = system[label]
                csv_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "threshold": applied_threshold,
                        "binary_accuracy": binary["accuracy"],
                        "binary_precision": binary["precision"],
                        "binary_recall": binary["recall"],
                        "binary_f1": binary["f1"],
                        "binary_false_positive_rate": binary["false_positive_rate"],
                        "binary_roc_auc": binary["roc_auc"],
                        "system_accuracy": system["accuracy"],
                        "system_macro_f1": system["macro avg"]["f1-score"],
                        "system_weighted_f1": system["weighted avg"]["f1-score"],
                        "class": label,
                        "class_precision": class_metrics["precision"],
                        "class_recall": class_metrics["recall"],
                        "class_f1": class_metrics["f1-score"],
                        "class_support": int(class_metrics["support"]),
                    }
                )

    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for strategy_key in (
        "baseline_default_threshold",
        "candidate_validation_selected_threshold",
    ):
        reports = [record[strategy_key] for record in repetitions]
        summary[strategy_key] = {}
        metric_getters = {
            "binary_accuracy": lambda report: report["binary_intrusion_detection"]["accuracy"],
            "binary_precision": lambda report: report["binary_intrusion_detection"]["precision"],
            "binary_recall": lambda report: report["binary_intrusion_detection"]["recall"],
            "binary_f1": lambda report: report["binary_intrusion_detection"]["f1"],
            "binary_false_positive_rate": lambda report: report["binary_intrusion_detection"]["false_positive_rate"],
            "binary_roc_auc": lambda report: report["binary_intrusion_detection"]["roc_auc"],
            "system_accuracy": lambda report: report["complete_system"]["metrics"]["accuracy"],
            "system_macro_f1": lambda report: report["complete_system"]["metrics"]["macro avg"]["f1-score"],
            "system_weighted_f1": lambda report: report["complete_system"]["metrics"]["weighted avg"]["f1-score"],
        }
        for metric_name, getter in metric_getters.items():
            summary[strategy_key][metric_name] = _metric_summary(
                [float(getter(report)) for report in reports]
            )
        for label in SYSTEM_LABELS:
            for metric_name in ("precision", "recall", "f1-score", "support"):
                summary[strategy_key][f"{label}_{metric_name}"] = _metric_summary(
                    [
                        float(report["complete_system"]["metrics"][label][metric_name])
                        for report in reports
                    ]
                )
    summary["candidate_validation_selected_threshold"][
        "selected_threshold"
    ] = _metric_summary([float(record["validation_threshold"]) for record in repetitions])

    report: dict[str, Any] = {
        "dataset": "KDDTrain+ processed training rows only",
        "official_test_data_loaded_or_used": False,
        "repetitions": repetitions,
        "summary_mean_and_sample_standard_deviation": summary,
        "protocol": {
            "seed_and_fold_index_pairs": [
                {"seed": seed, "fold_index": fold_index}
                for seed, fold_index in REPEATED_VALIDATION_SPLITS
            ],
            "split": (
                "For each seed, generate five-fold StratifiedGroupKFold partitions "
                "and use the specified fold index so repetitions have distinct "
                "validation groups. Attack names and identical full-feature rows "
                "are unioned into groups. The fold indices were selected to retain "
                "representation where feasible and include a rare-family-support "
                "fold; any missing class is reported."
            ),
            "observed_seed_effect": (
                "On this dataset, changing random_state alone yielded identical "
                "fold assignments, so distinct fold indices were specified explicitly."
            ),
            "fit_boundary": (
                "Feature selection, preprocessing, binary model, and family model "
                "are fitted only on each repetition's fit partition."
            ),
            "comparison": (
                "Paired comparison using the same fit-only models and validation "
                "predictions: baseline uses threshold 0.5; candidate selects the "
                "threshold maximizing five-class validation macro F1, with ties "
                "favoring the higher threshold."
            ),
            "interpretation_limit": (
                "Candidate metrics are measured on the same validation partition "
                "used to select its threshold, so they are selection-set estimates "
                "and may be optimistic. Across-seed variation is descriptive, not "
                "an independent test estimate."
            ),
            "model_random_state": RANDOM_STATE,
            "official_test_labels_used_for_selection": False,
        },
        "reproduction_command": (
            ".\\.venv\\Scripts\\python.exe -m nsl_kdd_ids.pipeline "
            "--step repeated-validation"
        ),
    }
    (evaluation_dir() / "repeated_grouped_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(csv_rows).to_csv(
        evaluation_dir() / "repeated_grouped_validation_metrics.csv", index=False
    )

    markdown = [
        "# Repeated attack-name-grouped validation",
        "",
        "Training data only; the official test split was not loaded or used.",
        "For each seed, the baseline threshold (0.5) and validation-selected",
        "threshold are evaluated on the same validation rows. Selected-threshold",
        "metrics are selection-set estimates, not independent generalization estimates.",
        "",
        "## Split results",
        "",
        "| Seed / fold | Strategy | Threshold | Binary accuracy | Binary precision | Binary recall | Binary F1 | ROC-AUC | FPR | System accuracy | Macro F1 | Weighted F1 | R2L recall (support) | U2R recall (support) | Missing validation classes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in repetitions:
        for strategy_key, strategy_label, applied_threshold in (
            ("baseline_default_threshold", "Baseline (0.5)", 0.5),
            (
                "candidate_validation_selected_threshold",
                "Candidate (validation-selected)",
                record["validation_threshold"],
            ),
        ):
            result = record[strategy_key]
            binary = result["binary_intrusion_detection"]
            system = result["complete_system"]["metrics"]
            r2l = system["R2L"]
            u2r = system["U2R"]
            markdown.append(
                f"| {record['seed']} / {record['fold_index']} "
                f"| {strategy_label} | {applied_threshold:.2f} "
                f"| {binary['accuracy']:.4f} | {binary['precision']:.4f} "
                f"| {binary['recall']:.4f} | {binary['f1']:.4f} "
                f"| {binary['roc_auc']:.4f} | {binary['false_positive_rate']:.4f} "
                f"| {system['accuracy']:.4f} | {system['macro avg']['f1-score']:.4f} "
                f"| {system['weighted avg']['f1-score']:.4f} "
                f"| {r2l['recall']:.4f} ({int(r2l['support'])}) "
                f"| {u2r['recall']:.4f} ({int(u2r['support'])}) "
                f"| {', '.join(record['grouping_audit']['missing_validation_classes']) or 'None'} |"
            )
    markdown.extend(["", "## Main metric means and sample standard deviations", ""])
    markdown.append("| Strategy | Metric | Mean | Sample SD |")
    markdown.append("|---|---|---:|---:|")
    main_metrics = (
        "binary_accuracy",
        "binary_precision",
        "binary_recall",
        "binary_f1",
        "binary_roc_auc",
        "binary_false_positive_rate",
        "system_accuracy",
        "system_macro_f1",
        "system_weighted_f1",
        "R2L_recall",
        "U2R_recall",
        "R2L_support",
        "U2R_support",
    )
    for strategy, strategy_summary in summary.items():
        for key in main_metrics:
            metric = strategy_summary[key]
            markdown.append(
                f"| {strategy} | {key} | {metric['mean']:.4f} | "
                f"{metric['sample_standard_deviation']:.4f} |"
            )
    threshold_metric = summary["candidate_validation_selected_threshold"][
        "selected_threshold"
    ]
    markdown.append(
        f"| candidate_validation_selected_threshold | selected_threshold "
        f"| {threshold_metric['mean']:.4f} "
        f"| {threshold_metric['sample_standard_deviation']:.4f} |"
    )
    markdown.extend(
        [
            "",
            "## Per-class metrics by repetition",
            "",
            "| Seed / fold | Strategy | Class | Precision | Recall | F1 | Support |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in repetitions:
        for strategy_key, strategy_label in (
            ("baseline_default_threshold", "Baseline (0.5)"),
            (
                "candidate_validation_selected_threshold",
                "Candidate (validation-selected)",
            ),
        ):
            class_metrics = record[strategy_key]["complete_system"]["metrics"]
            for label in SYSTEM_LABELS:
                metric = class_metrics[label]
                markdown.append(
                    f"| {record['seed']} / {record['fold_index']} "
                    f"| {strategy_label} | {label} | {metric['precision']:.4f} "
                    f"| {metric['recall']:.4f} | {metric['f1-score']:.4f} "
                    f"| {int(metric['support'])} |"
                )
    markdown.extend(
        [
            "",
            "## Grouping and limitations",
            "",
            "Attack names and identical full-feature rows are assigned as indivisible "
            "groups. Per-seed held-out names, overlap, class support, and missing "
            "classes are recorded in the JSON report. Feature selection and all "
            "preprocessing/model fitting use only that split's fit rows.",
            "Different random_state values produced the same fold assignment here; "
            "the experiment therefore uses distinct fold indices (1, 2, and 4). "
            "Fold 4 lacks Probe in validation, and its metrics must not be treated "
            "as a complete five-class estimate.",
            "The threshold is selected and evaluated on the same validation rows; "
            "this can overstate candidate metrics. R2L/U2R support and their few "
            "attack-name groups limit confidence. This experiment compares threshold "
            "selection with the default threshold for fixed model settings; it does "
            "not establish generalization to unseen data.",
            "",
            "Reproduce with: "
            + f"`{report['reproduction_command']}`",
            "",
        ]
    )
    (evaluation_dir() / "repeated_grouped_validation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report


def _evaluate_predictions(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    family_predictions_for_rows: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    binary_predictions = _threshold_predictions(probabilities, threshold)
    actual_attack = test["is_attack"].to_numpy(dtype="int8")
    actual_categories = test["attack_category"].to_numpy()
    if len(binary_predictions) != len(test):
        raise ValueError("Binary prediction count must match the evaluation rows.")
    if len(family_predictions_for_rows) != len(test):
        raise ValueError("Family prediction count must match the evaluation rows.")

    attack_rows = np.flatnonzero(actual_attack == 1)
    family_report = classification_report(
        actual_categories[attack_rows],
        family_predictions_for_rows[attack_rows],
        labels=FAMILY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    detected_family_predictions = family_predictions_for_rows[binary_predictions == 1]
    final_predictions = _map_system_predictions(binary_predictions, detected_family_predictions)
    final_report = classification_report(
        actual_categories,
        final_predictions,
        labels=SYSTEM_LABELS,
        output_dict=True,
        zero_division=0,
    )
    binary_matrix = confusion_matrix(actual_attack, binary_predictions, labels=BINARY_LABELS)
    family_matrix = confusion_matrix(
        actual_categories[attack_rows],
        family_predictions_for_rows[attack_rows],
        labels=FAMILY_LABELS,
    )
    final_matrix = confusion_matrix(actual_categories, final_predictions, labels=SYSTEM_LABELS)
    binary_report = _binary_metrics(
        pd.Series(actual_attack),
        binary_predictions,
        np.asarray(probabilities, dtype="float64"),
    )
    binary_report["confusion_matrix"] = binary_matrix.tolist()

    report = {
        "binary_intrusion_detection": binary_report,
        "attack_family": {
            "labels": FAMILY_LABELS,
            "metrics": family_report,
            "confusion_matrix": family_matrix.tolist(),
        },
        "complete_system": {
            "labels": SYSTEM_LABELS,
            "metrics": final_report,
            "confusion_matrix": final_matrix.tolist(),
        },
        "attacks_missed_by_detector": int(((actual_attack == 1) & (binary_predictions == 0)).sum()),
        "attacks_detected_but_wrong_family": int(
            (
                (actual_attack == 1)
                & (binary_predictions == 1)
                & (final_predictions != actual_categories)
            ).sum()
        ),
        "false_positives_among_normal": int(
            ((actual_attack == 0) & (binary_predictions == 1)).sum()
        ),
    }
    return report, binary_predictions, final_predictions


def _validation_threshold(
    validation: pd.DataFrame,
    binary_model: Pipeline,
    family_model: Pipeline,
    features: list[str],
) -> tuple[float, dict[str, float]]:
    probabilities = _positive_probabilities(binary_model, validation[features])
    family_predictions = family_model.predict(validation[features])
    best_threshold: float | None = None
    best_score = -1.0
    best_binary_predictions: np.ndarray | None = None
    for threshold in THRESHOLD_CANDIDATES:
        binary_predictions = _threshold_predictions(probabilities, float(threshold))
        detected_families = family_predictions[binary_predictions == 1]
        final_predictions = _map_system_predictions(binary_predictions, detected_families)
        score = f1_score(
            validation["attack_category"],
            final_predictions,
            labels=SYSTEM_LABELS,
            average="macro",
            zero_division=0,
        )
        if score > best_score or (score == best_score and (best_threshold is None or threshold > best_threshold)):
            best_threshold = float(threshold)
            best_score = float(score)
            best_binary_predictions = binary_predictions

    if best_threshold is None or best_binary_predictions is None:
        raise RuntimeError("Threshold selection did not evaluate any candidates.")
    validation_report = _binary_metrics(
        validation["is_attack"],
        best_binary_predictions,
        probabilities,
    )
    validation_report["complete_system_macro_f1"] = best_score
    return best_threshold, validation_report


def _flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "confusion_matrix":
                continue
            flattened.update(_flatten_numeric_metrics(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def _metrics_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = _flatten_numeric_metrics(left)
    right_values = _flatten_numeric_metrics(right)
    return set(left_values).issubset(right_values) and all(
        np.isclose(left_values[key], right_values[key], rtol=1e-10, atol=1e-12)
        for key in left_values
    )


def _comparison_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> pd.DataFrame:
    baseline_values = _flatten_numeric_metrics(baseline)
    candidate_values = _flatten_numeric_metrics(candidate)
    rows = []
    for metric in sorted(set(baseline_values) | set(candidate_values)):
        before = baseline_values.get(metric)
        after = candidate_values.get(metric)
        rows.append(
            {
                "metric": metric,
                "baseline": before,
                "candidate": after,
                "delta": after - before if before is not None and after is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _validation_split_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    feature_hashes = pd.util.hash_pandas_object(
        train[FEATURE_COLUMNS], index=False
    ).to_numpy()
    fit_hashes = set(feature_hashes[fit_indices])
    validation_hashes = set(feature_hashes[validation_indices])
    fit_labels = set(train.iloc[fit_indices]["attack_label"])
    validation_labels = set(train.iloc[validation_indices]["attack_label"])

    historical_fit, historical_validation_indices = train_test_split(
        np.arange(len(train)),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train["attack_category"],
    )
    historical_fit_labels = set(train.iloc[historical_fit]["attack_label"])
    historical_validation = train.iloc[historical_validation_indices]
    historical_validation_attacks = historical_validation.loc[
        historical_validation["is_attack"] == 1
    ]
    training_attack_labels = set(train.loc[train["is_attack"] == 1, "attack_label"])
    unseen_test_counts = (
        test.loc[
            (test["is_attack"] == 1)
            & ~test["attack_label"].isin(training_attack_labels),
            "attack_label",
        ]
        .value_counts()
        .sort_index()
    )
    return {
        "grouped_validation": {
            "fit_support": {
                str(key): int(value)
                for key, value in train.iloc[fit_indices]["attack_category"].value_counts().items()
            },
            "validation_support": {
                str(key): int(value)
                for key, value in train.iloc[validation_indices]["attack_category"].value_counts().items()
            },
            "held_out_attack_labels": sorted(
                (validation_labels - fit_labels) - {"normal"}
            ),
            "attack_label_overlap": sorted(
                (validation_labels & fit_labels) - {"normal"}
            ),
            "exact_full_feature_patterns_shared": len(fit_hashes & validation_hashes),
        },
        "historical_random_row_validation": {
            "validation_rows": len(historical_validation),
            "validation_attack_rows": len(historical_validation_attacks),
            "validation_attack_rows_with_attack_name_seen_in_fit": int(
                historical_validation_attacks["attack_label"]
                .isin(historical_fit_labels)
                .sum()
            ),
            "exact_full_feature_patterns_shared": len(
                set(feature_hashes[historical_fit])
                & set(feature_hashes[historical_validation_indices])
            ),
        },
        "official_test_distribution_diagnostic": {
            "unseen_attack_label_count": int(len(unseen_test_counts)),
            "unseen_attack_rows": int(unseen_test_counts.sum()),
            "unseen_attack_labels_and_support": {
                str(key): int(value) for key, value in unseen_test_counts.items()
            },
            "used_for_model_or_threshold_selection": False,
        },
    }


def _load_recorded_baseline(test: pd.DataFrame) -> dict[str, Any]:
    prediction_path = evaluation_dir() / "test_predictions.csv"
    binary_report_path = models_dir() / "binary_model_results.json"
    binary_matrix_path = models_dir() / "random_forest_confusion_matrix.csv"
    family_report_path = attack_model_dir() / "report.json"
    family_matrix_path = attack_model_dir() / "confusion_matrix.csv"
    system_report_path = evaluation_dir() / "complete_system_report.json"
    system_matrix_path = evaluation_dir() / "complete_system_confusion_matrix.csv"
    for path in (
        prediction_path,
        binary_report_path,
        binary_matrix_path,
        family_report_path,
        family_matrix_path,
        system_report_path,
        system_matrix_path,
    ):
        _require_file(path, "Run the existing binary, family, and complete-system evaluation first.")

    predictions = pd.read_csv(prediction_path)
    if len(predictions) != len(test):
        raise ValueError("Recorded baseline prediction count does not match KDDTest+.")
    for prediction_column, test_column in (
        ("actual_label", "attack_label"),
        ("actual_category", "attack_category"),
        ("actual_is_attack", "is_attack"),
    ):
        if prediction_column not in predictions or not np.array_equal(
            predictions[prediction_column].to_numpy(), test[test_column].to_numpy()
        ):
            raise ValueError(
                f"Recorded baseline {prediction_column} values do not align with the current KDDTest+ rows."
            )

    with binary_report_path.open(encoding="utf-8") as stream:
        binary_summary = json.load(stream)
    binary_results = binary_summary["models"]["Random Forest"]
    baseline_features = binary_summary["selected_features"]
    with family_report_path.open(encoding="utf-8") as stream:
        family_metrics = json.load(stream)["metrics"]
    with system_report_path.open(encoding="utf-8") as stream:
        system_metrics = json.load(stream)["complete_system"]
    binary_matrix = pd.read_csv(binary_matrix_path, index_col=0).to_numpy(dtype="int64")
    family_matrix = pd.read_csv(family_matrix_path, index_col=0).to_numpy(dtype="int64")
    system_matrix = pd.read_csv(system_matrix_path, index_col=0).to_numpy(dtype="int64")
    if binary_matrix.shape != (2, 2) or family_matrix.shape != (4, 4) or system_matrix.shape != (5, 5):
        raise ValueError("Recorded baseline confusion matrices have unexpected dimensions.")

    binary_predictions = predictions["predicted_is_attack"].to_numpy(dtype="int8")
    actual_attack = test["is_attack"].to_numpy(dtype="int8")
    actual_categories = test["attack_category"].to_numpy()
    predicted_categories = predictions["predicted_category"].to_numpy()
    persisted_binary_model = joblib.load(models_dir() / "random_forest.joblib")
    persisted_family_model = joblib.load(
        attack_model_dir() / "random_forest_attack_type.joblib"
    )
    model_binary_predictions = persisted_binary_model.predict(
        test[baseline_features]
    ).astype("int8")
    persisted_family_predictions = persisted_family_model.predict(
        test.loc[actual_attack == 1, baseline_features]
    )
    gated_family_predictions = persisted_family_model.predict(
        test.loc[binary_predictions == 1, baseline_features]
    )
    recorded_binary_matrix = confusion_matrix(
        actual_attack, binary_predictions, labels=BINARY_LABELS
    )
    recorded_family_matrix = confusion_matrix(
        actual_categories[actual_attack == 1],
        persisted_family_predictions,
        labels=FAMILY_LABELS,
    )
    recorded_system_matrix = confusion_matrix(
        actual_categories, predicted_categories, labels=SYSTEM_LABELS
    )
    if not np.array_equal(model_binary_predictions, binary_predictions):
        raise ValueError("Persisted baseline binary model does not reproduce recorded predictions.")
    if not np.array_equal(gated_family_predictions, predicted_categories[binary_predictions == 1]):
        raise ValueError("Persisted baseline family model does not reproduce gated predictions.")
    if not (
        np.array_equal(recorded_binary_matrix, binary_matrix)
        and np.array_equal(recorded_family_matrix, family_matrix)
        and np.array_equal(recorded_system_matrix, system_matrix)
    ):
        raise ValueError("Baseline reports or confusion matrices do not match saved predictions/models.")
    probabilities = _positive_probabilities(persisted_binary_model, test[baseline_features])
    ties_at_default_threshold = int(np.count_nonzero(probabilities == 0.5))
    recomputed_binary_metrics = _binary_metrics(
        pd.Series(actual_attack),
        binary_predictions,
        probabilities,
    )
    recomputed_family_metrics = classification_report(
        actual_categories[actual_attack == 1],
        persisted_family_predictions,
        labels=FAMILY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    recomputed_system_metrics = classification_report(
        actual_categories,
        predicted_categories,
        labels=SYSTEM_LABELS,
        output_dict=True,
        zero_division=0,
    )
    if not (
        _metrics_match(binary_results, recomputed_binary_metrics)
        and _metrics_match(family_metrics, recomputed_family_metrics)
        and _metrics_match(system_metrics, recomputed_system_metrics)
    ):
        raise ValueError("A baseline metrics report does not match its recorded predictions.")
    binary_metrics = dict(binary_results)
    normal_count = int(binary_matrix[0].sum())
    binary_metrics["false_positive_rate"] = (
        float(binary_matrix[0, 1] / normal_count) if normal_count else 0.0
    )
    binary_metrics["confusion_matrix"] = binary_matrix.tolist()
    return {
        "binary_intrusion_detection": binary_metrics,
        "attack_family": {
            "labels": FAMILY_LABELS,
            "metrics": family_metrics,
            "confusion_matrix": family_matrix.tolist(),
        },
        "complete_system": {
            "labels": SYSTEM_LABELS,
            "metrics": system_metrics,
            "confusion_matrix": system_matrix.tolist(),
        },
        "attacks_missed_by_detector": int(
            ((actual_attack == 1) & (binary_predictions == 0)).sum()
        ),
        "attacks_detected_but_wrong_family": int(
            (
                (actual_attack == 1)
                & (binary_predictions == 1)
                & (predicted_categories != actual_categories)
            ).sum()
        ),
        "false_positives_among_normal": int(
            ((actual_attack == 0) & (binary_predictions == 1)).sum()
        ),
        "artifact_consistency": {
            "binary_model_predictions_match_recorded": True,
            "family_model_predictions_match_recorded_when_gated": True,
            "all_confusion_matrices_match": True,
            "binary_probability_ties_at_0_5": ties_at_default_threshold,
        },
    }


def run_controlled_experiment() -> dict[str, Any]:
    train, test = _load_processed()

    fit_indices, validation_indices = _training_validation_indices(train)
    fit = train.iloc[fit_indices]
    validation = train.iloc[validation_indices]
    features, _ = _select_top_features(fit)
    binary_validation_model = _model_pipeline(
        features,
        _experiment_binary_classifier(),
    )
    family_fit = fit.loc[fit["is_attack"] == 1]
    family_validation_model = _model_pipeline(
        features,
        _experiment_family_classifier(),
    )
    binary_validation_model.fit(fit[features], fit["is_attack"])
    family_validation_model.fit(family_fit[features], family_fit["attack_category"])
    threshold, validation_report = _validation_threshold(
        validation, binary_validation_model, family_validation_model, features
    )

    candidate_binary = _model_pipeline(
        features,
        _experiment_binary_classifier(),
    )
    family_train = train.loc[train["is_attack"] == 1]
    candidate_family = _model_pipeline(
        features,
        _experiment_family_classifier(),
    )
    candidate_binary.fit(train[features], train["is_attack"])
    candidate_family.fit(family_train[features], family_train["attack_category"])
    _validate_model_feature_contract(candidate_binary, candidate_family, features)
    baseline_report = _load_recorded_baseline(test)
    candidate_probabilities = _positive_probabilities(candidate_binary, test[features])
    candidate_family_predictions = candidate_family.predict(test[features])
    candidate_report, candidate_binary_predictions, candidate_final_predictions = _evaluate_predictions(
        test, candidate_probabilities, candidate_family_predictions, threshold
    )

    experiment_models_dir = models_dir() / "controlled_experiment_grouped_validation"
    experiment_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_binary, experiment_models_dir / "binary_random_forest.joblib")
    joblib.dump(candidate_family, experiment_models_dir / "attack_family_random_forest.joblib")
    (experiment_models_dir / "selected_features.json").write_text(
        json.dumps({"selected_features": features}, indent=2), encoding="utf-8"
    )
    evaluation_dir().mkdir(parents=True, exist_ok=True)
    candidate_predictions = pd.DataFrame(
        {
            "actual_label": test["attack_label"],
            "actual_category": test["attack_category"],
            "actual_is_attack": test["is_attack"],
            "binary_probability": candidate_probabilities,
            "predicted_is_attack": candidate_binary_predictions,
            "predicted_category": candidate_final_predictions,
        }
    )
    candidate_predictions.to_csv(
        evaluation_dir() / "reliable_experiment_candidate_predictions.csv", index=False
    )
    comparison = _comparison_table(baseline_report, candidate_report)
    comparison.to_csv(evaluation_dir() / "reliable_experiment_comparison.csv", index=False)
    historical_validation = None
    historical_report_path = evaluation_dir() / "controlled_experiment_report.json"
    if historical_report_path.is_file():
        with historical_report_path.open(encoding="utf-8") as stream:
            historical_validation = json.load(stream).get(
                "threshold_selection", {}
            ).get("validation_metrics")
    split_audit = _validation_split_audit(
        train, test, fit_indices, validation_indices
    )
    historical_split_audit = split_audit["historical_random_row_validation"]
    unseen_test_audit = split_audit["official_test_distribution_diagnostic"]
    report = {
        "problem_investigated": (
            "Historical random-row validation scored approximately 99.9% binary "
            "accuracy while the official-test candidate scored approximately 79.8%."
        ),
        "root_cause_and_evidence": {
            "finding": (
                "No fit/validation preprocessing contamination was found: feature "
                "selection and both validation pipelines are fitted on the fit partition. "
                "The historical validation design was optimistic for unseen attack "
                "types because every one of its 11,726 validation attack rows shared "
                "an attack name with fit. KDDTest+ has attack names absent from training."
            ),
            "historical_attack_name_overlap_rows": historical_split_audit[
                "validation_attack_rows_with_attack_name_seen_in_fit"
            ],
            "historical_validation_attack_rows": historical_split_audit[
                "validation_attack_rows"
            ],
            "historical_exact_feature_duplicates_across_fit_validation": historical_split_audit[
                "exact_full_feature_patterns_shared"
            ],
            "official_test_unseen_attack_rows": unseen_test_audit["unseen_attack_rows"],
            "official_test_unseen_attack_names": unseen_test_audit[
                "unseen_attack_label_count"
            ],
            "interpretation": (
                "The rare exact duplicate overlap is not sufficient evidence to explain "
                "the score gap. The principal confirmed issue is validation/test "
                "distribution mismatch and lack of attack-name holdout in validation; "
                "the full gap cannot be attributed to a single factor."
            ),
        },
        "changes_made": [
            "Validation attack names are grouped so a non-normal attack name cannot occur in both fit and validation.",
            "Exact full-feature duplicate rows are grouped across fit and validation, including rows with different labels.",
            "Feature selection and preprocessing remain fitted only on the selected fit partition.",
            "Persisted baseline models, predictions, metrics, and confusion matrices are cross-validated for consistency.",
            "Corrected results are saved under new reliable_experiment_* paths; historical random-row experiment artifacts remain unchanged.",
        ],
        "dataset": "NSL-KDD official test split; model and threshold choices made using KDDTrain+ only",
        "test_rows": len(test),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "baseline_source": (
            "Existing baseline predictions/reports and persisted models; labels, model "
            "predictions, gated family outputs, and confusion matrices were cross-checked."
        ),
        "candidate_features": features,
        "validation_split_audit": split_audit,
        "threshold_selection": {
            "objective": "maximize five-class complete-system macro F1 on grouped validation data; ties choose the higher threshold",
            "threshold": threshold,
            "validation_metrics": validation_report,
            "method": (
                "StratifiedGroupKFold with five folds. Every non-normal attack label "
                "is grouped, and identical full-feature records are grouped across "
                "labels. A fold is selected by class support only, maximizing the "
                "minimum support in both fit and validation partitions."
            ),
            "historical_random_split_metrics_not_used_for_selection": historical_validation,
            "threshold_search": {
                "minimum": float(THRESHOLD_CANDIDATES[0]),
                "maximum": float(THRESHOLD_CANDIDATES[-1]),
                "step": round(float(THRESHOLD_CANDIDATES[1] - THRESHOLD_CANDIDATES[0]), 2),
            },
        },
        "baseline": baseline_report,
        "candidate": candidate_report,
        "reproduction": {
            "command": "python -m nsl_kdd_ids.pipeline --step experiment",
            "historical_random_split_outputs_preserved": True,
            "new_outputs": [
                "artifacts/evaluation/reliable_experiment_report.json",
                "artifacts/evaluation/reliable_experiment_comparison.csv",
                "artifacts/evaluation/reliable_experiment_candidate_predictions.csv",
                "artifacts/models/controlled_experiment_grouped_validation/",
            ],
        },
    }
    (evaluation_dir() / "reliable_experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train_binary_models() -> pd.DataFrame:
    train, test = _load_processed()
    features = _selected_features()
    y_train = train["is_attack"]
    y_test = test["is_attack"]
    x_train = train[features]
    x_test = test[features]

    knn_train = train.groupby("is_attack", group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), max(1, round(KNN_SAMPLE_SIZE * len(group) / len(train)))),
            random_state=RANDOM_STATE,
        ),
        include_groups=False,
    )
    # Keep labels aligned after sampling without including the target in model input.
    knn_x = train.loc[knn_train.index, features]
    knn_y = train.loc[knn_train.index, "is_attack"]

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
    }
    results = {}
    models_dir().mkdir(parents=True, exist_ok=True)
    for name, classifier in classifiers.items():
        model = _model_pipeline(features, classifier)
        fit_x, fit_y = (knn_x, knn_y) if name == "K-Nearest Neighbors" else (x_train, y_train)
        model.fit(fit_x, fit_y)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        results[name] = _binary_metrics(y_test, predictions, probabilities)
        joblib.dump(model, models_dir() / f"{name.lower().replace(' ', '_').replace('-', '')}.joblib")
        pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=[0, 1]),
            index=["actual_normal", "actual_attack"],
            columns=["predicted_normal", "predicted_attack"],
        ).to_csv(models_dir() / f"{name.lower().replace(' ', '_').replace('-', '')}_confusion_matrix.csv")

    report = {
        "dataset": "NSL-KDD official test split",
        "test_rows": len(test),
        "training_rows": len(train),
        "knn_training_rows": len(knn_x),
        "selected_features": features,
        "models": results,
    }
    (models_dir() / "binary_model_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(results).T


def train_attack_classifier() -> pd.DataFrame:
    train, test = _load_processed()
    features = _selected_features()
    attack_train = train.loc[train["is_attack"] == 1]
    attack_test = test.loc[test["is_attack"] == 1]
    model = _model_pipeline(
        features,
        RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(attack_train[features], attack_train["attack_category"])
    predictions = model.predict(attack_test[features])
    labels = ["DoS", "Probe", "R2L", "U2R"]
    report = classification_report(
        attack_test["attack_category"],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    attack_model_dir().mkdir(parents=True, exist_ok=True)
    joblib.dump(model, attack_model_dir() / "random_forest_attack_type.joblib")
    pd.DataFrame(
        confusion_matrix(attack_test["attack_category"], predictions, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ).to_csv(attack_model_dir() / "confusion_matrix.csv")
    summary = {
        "dataset": "NSL-KDD official test split, attack records only",
        "train_attack_rows": len(attack_train),
        "test_attack_rows": len(attack_test),
        "labels": labels,
        "metrics": report,
    }
    (attack_model_dir() / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(report).T


def explain_models() -> dict[str, pd.DataFrame]:
    features = _selected_features()
    model_paths = {
        "binary_random_forest": models_dir() / "random_forest.joblib",
        "attack_type_random_forest": attack_model_dir() / "random_forest_attack_type.joblib",
    }
    explainability_dir().mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, path in model_paths.items():
        _require_file(path, "Train the binary and attack-family models first.")
        model = joblib.load(path)
        importance = _aggregate_importances(model, features).rename("importance")
        outputs[name] = importance.to_frame()
        importance.to_csv(explainability_dir() / f"{name}_feature_importance.csv")
        figure, axis = plt.subplots(figsize=(10, 6))
        importance.sort_values().plot.barh(ax=axis, color="steelblue")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Aggregated Random Forest feature importance")
        figure.tight_layout()
        figure.savefig(explainability_dir() / f"{name}_feature_importance.png", dpi=150)
        plt.close(figure)
    return outputs


def verify_full_system() -> dict[str, Any]:
    _, test = _load_processed()
    features = _selected_features()
    binary_path = models_dir() / "random_forest.joblib"
    family_path = attack_model_dir() / "random_forest_attack_type.joblib"
    _require_file(binary_path, "Train binary models before evaluation.")
    _require_file(family_path, "Train the attack-family model before evaluation.")

    binary_model = joblib.load(binary_path)
    family_model = joblib.load(family_path)
    _validate_model_feature_contract(binary_model, family_model, features)
    binary_predictions = binary_model.predict(test[features]).astype("int8")
    predicted_attack = np.flatnonzero(binary_predictions == 1)
    predicted_families = family_model.predict(test.iloc[predicted_attack][features])
    final_predictions = _map_system_predictions(binary_predictions, predicted_families)

    categories = SYSTEM_LABELS
    y_final = test["attack_category"].to_numpy()
    final_report = classification_report(
        y_final, final_predictions, labels=categories, output_dict=True, zero_division=0
    )
    binary_report = {
        "accuracy": float(accuracy_score(test["is_attack"], binary_predictions)),
        "precision": float(precision_score(test["is_attack"], binary_predictions, zero_division=0)),
        "recall": float(recall_score(test["is_attack"], binary_predictions, zero_division=0)),
        "f1": float(f1_score(test["is_attack"], binary_predictions, zero_division=0)),
    }
    evaluation_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        confusion_matrix(y_final, final_predictions, labels=categories),
        index=[f"actual_{label}" for label in categories],
        columns=[f"predicted_{label}" for label in categories],
    ).to_csv(evaluation_dir() / "complete_system_confusion_matrix.csv")
    predictions = pd.DataFrame(
        {
            "actual_label": test["attack_label"],
            "actual_category": test["attack_category"],
            "actual_is_attack": test["is_attack"],
            "predicted_is_attack": binary_predictions,
            "predicted_category": final_predictions,
        }
    )
    predictions.to_csv(evaluation_dir() / "test_predictions.csv", index=False)
    report = {
        "dataset": "NSL-KDD official test split",
        "test_rows": len(test),
        "binary_intrusion_detection": binary_report,
        "complete_system": final_report,
        "predicted_attacks": int(binary_predictions.sum()),
        "wrong_final_categories": int((y_final != final_predictions).sum()),
    }
    (evaluation_dir() / "complete_system_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def run_all() -> None:
    print("\n== NSL-KDD dataset ==")
    distribution, sample = inspect_dataset()
    print(distribution)
    print(sample)
    print("\n== Prepare official train/test splits ==")
    print(prepare_data())
    print("\n== Select features ==")
    print(select_features().head(TOP_FEATURES))
    print("\n== Train binary intrusion detectors ==")
    print(train_binary_models())
    print("\n== Train attack-family classifier ==")
    print(train_attack_classifier())
    print("\n== Explain model features ==")
    explain_models()
    print("\n== Verify complete system ==")
    print(json.dumps(verify_full_system(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSL-KDD IDS notebook pipeline.")
    parser.add_argument(
        "--step",
        choices=(
            "all",
            "inspect",
            "prepare",
            "select",
            "train-binary",
            "train-family",
            "explain",
            "verify",
            "experiment",
            "repeated-validation",
            "nested-validation",
        ),
        default="all",
    )
    args = parser.parse_args()
    actions = {
        "inspect": inspect_dataset,
        "prepare": prepare_data,
        "select": select_features,
        "train-binary": train_binary_models,
        "train-family": train_attack_classifier,
        "explain": explain_models,
        "verify": verify_full_system,
        "experiment": run_controlled_experiment,
        "repeated-validation": run_repeated_grouped_validation,
        "nested-validation": run_nested_grouped_validation,
    }
    if args.step == "all":
        run_all()
    else:
        result = actions[args.step]()
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
