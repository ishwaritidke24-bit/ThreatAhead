from __future__ import annotations

import argparse
import json
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TOP_FEATURES = 20
KNN_SAMPLE_SIZE = 25_000

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


def select_features() -> pd.DataFrame:
    train, _ = _load_processed()
    features = FEATURE_COLUMNS
    selector = Pipeline(
        [
            ("preprocessor", _build_preprocessor(features)),
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
    selector.fit(train[features], train["is_attack"])
    importance = _aggregate_importances(selector, features)
    selected = importance.head(min(TOP_FEATURES, len(importance))).index.tolist()

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


def _binary_metrics(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


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
    binary_predictions = binary_model.predict(test[features]).astype("int8")
    final_predictions = np.full(len(test), "Normal", dtype=object)
    predicted_attack = np.flatnonzero(binary_predictions == 1)
    if len(predicted_attack):
        final_predictions[predicted_attack] = family_model.predict(
            test.iloc[predicted_attack][features]
        )

    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
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
        choices=("all", "inspect", "prepare", "select", "train-binary", "train-family", "explain", "verify"),
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
    }
    if args.step == "all":
        run_all()
    else:
        result = actions[args.step]()
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
