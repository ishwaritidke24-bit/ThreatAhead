# ThreatAhead

The pipeline detects normal traffic versus attacks, then classifies
detected attacks into the four NSL-KDD families: DoS, Probe, R2L, and U2R.

NSL-KDD is much smaller than CIC-IDS2017. This project uses the benchmark's
official `KDDTrain+` and `KDDTest+` split; it does not combine and randomly
resplit the two files.

## Dataset

The project expects these two files:

```text
dataset/
└── NSL-KDD/
    ├── KDDTrain+.txt
    └── KDDTest+.txt
```

The provided `archive (4).zip` contained both files, and they have been
extracted into `dataset/NSL-KDD/`. The archive also contains `.arff` copies,
smaller subsets, and duplicate paths; those are not needed here. Only the
official full train and test text files are used.

Each row contains 41 connection features, an attack label, and a difficulty
score. The difficulty score is excluded from model input. The three categorical
features (`protocol_type`, `service`, and `flag`) are one-hot encoded; numeric
features are standardized inside each model pipeline using training data only.

Attack labels are normalized to lowercase and mapped using the standard
NSL-KDD families:

| Output class | Contents |
| --- | --- |
| Normal | `normal` |
| DoS | Denial-of-service attacks |
| Probe | Scanning and probing attacks |
| R2L | Remote-to-local attacks |
| U2R | User-to-root attacks |

The binary detector uses `0 = normal`, `1 = attack`. The attack-family model
uses only attack records. It predicts the four attack families rather than
individual attack names.

## Setup

Requires Python 3.11 or later. Install the project dependencies and notebook
tools with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --extra notebook
```

The existing `.venv` can also be used directly:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step all
```

The full pipeline can take a few minutes, depending on the computer. K-Nearest
Neighbors uses a deterministic, stratified sample of at most 25,000 training
records to keep prediction time manageable. The other binary models and the
attack-family model use the official training split.

## Run the notebooks

Start Jupyter Lab from the repository root:

```powershell
uv run --extra notebook jupyter lab
```

Run these notebooks in order:

| Order | Notebook | Work |
| --- | --- | --- |
| 1 | `00_explore_data.ipynb` | Inspect the supplied train/test files |
| 2 | `01_load_and_clean_data.ipynb` | Validate labels and save processed Parquet splits |
| 3 | `02_feature_selection.ipynb` | Rank features and select the top 20 |
| 4 | `03_train_models.ipynb` | Train and compare binary detectors |
| 5 | `04_attack_type_classification.ipynb` | Train the four-family classifier |
| 6 | `05_model_explainability.ipynb` | Save aggregated Random Forest feature importances |
| 7 | `06_verify_trained_models.ipynb` | Evaluate the full two-stage system on KDDTest+ |

The same stages can be run individually from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step inspect
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step prepare
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step select
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step train-binary
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step train-family
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step explain
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step verify
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step experiment
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step repeated-validation
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step nested-validation
```

`--step all` executes every stage in order. Later stages require the output of
earlier stages and report a clear error if a prerequisite is missing.

The controlled experiment compares the saved baseline with a candidate on the
same untouched `KDDTest+` split. Validation uses a deterministic
`StratifiedGroupKFold` split of `KDDTrain+`: each attack name is held together,
and rows with identical full input features are kept in the same group. This
tests generalization to attack names not seen during fitting and prevents exact
duplicate feature rows crossing fit and validation. The fold is selected by
class support only; model and feature selection fit exclusively on its fit
partition. The binary probability threshold is selected on validation data to
maximize five-class complete-system macro F1. Ties favor the higher threshold.
After selection, the candidate models are refit on all of `KDDTrain+`; saved
baseline models and historical reports are kept unchanged. Baseline model
predictions, family predictions, and confusion matrices are checked against
the existing recorded evaluation. A threshold of exactly 0.5 can differ from
scikit-learn `predict()` when probabilities tie; the baseline comparison uses
the recorded default predictions. The macro-F1 objective balances all five
categories but trades normal false positives against attack recall.

The latest audited run writes `reliable_experiment_report.json`,
`reliable_experiment_comparison.csv`, and
`reliable_experiment_candidate_predictions.csv` under
`artifacts/evaluation/`; candidate models are saved under
`artifacts/models/controlled_experiment_grouped_validation/`. The earlier
random-row validation experiment remains in its original `controlled_experiment_*`
artifacts for historical comparison and is not overwritten.

`--step repeated-validation` runs three deterministic attack-name-grouped
splits on `KDDTrain+` only. It compares the same fixed, fit-only two-stage
models at the default 0.5 threshold and at a threshold selected for validation
macro F1. Since threshold selection and candidate scoring use the same
validation partition, these are selection-set results, not independent
generalization estimates. The JSON, CSV, and Markdown results are written to
`repeated_grouped_validation_report.json`,
`repeated_grouped_validation_metrics.csv`, and
`repeated_grouped_validation_report.md` under `artifacts/evaluation/`.
The selected fold indices are distinct because changing the random seed alone
produced identical grouped-fold assignments for the current data. One selected
fold has no Probe validation examples; that limitation is called out rather
than treated as a complete five-class estimate.

`--step nested-validation` is the preferred training-only protocol for comparing
threshold policies. It uses three grouped outer folds and a grouped inner split
inside each outer training partition. Feature selection, preprocessing, and
models are fitted only within their corresponding fit partition. Outer labels
are used for stratified fold construction and reporting, never model or threshold
selection. It compares a fixed 0.5 threshold, an inner-binary
F1-selected threshold, and an inner threshold maximizing attack recall subject
to a 1% empirical false-positive-rate limit. The 1% is an experimental
constraint, not a production requirement. The inner folds cannot consistently
represent every attack family, so threshold selection uses binary metrics;
one outer fold has no Probe records and is explicitly flagged. The JSON and
CSV/Markdown reports are written as `nested_grouped_validation_report.json`,
`nested_grouped_validation_metrics.csv`, and
`nested_grouped_validation_report.md` under `artifacts/evaluation/`.
In the current run, the inner-F1 policy averaged 0.926 binary recall and 0.0148
outer-fold FPR; the inner 1%-FPR policy averaged 0.914 recall and 0.0102 outer
FPR. That policy exceeded its 1% limit on one outer fold (1.36%), demonstrating
that an inner empirical constraint is not a deployment guarantee. Five-class
macro F1 averaged 0.442 and 0.449, respectively, with high fold variability
and missing Probe support in one fold. No policy is promoted based on these
outer results; the intended normal-traffic alert budget still needs to be
defined independently.

The candidate from the controlled experiment is not the model used by the
production evaluation step. `verify_full_system()` loads the baseline
`artifacts/models/random_forest.joblib` and
`artifacts/attack_type_model/random_forest_attack_type.joblib`, then uses
scikit-learn's default `.predict()` behavior. The reliable candidate models
are stored separately under
`artifacts/models/controlled_experiment_grouped_validation/`, and its tuned
threshold (0.28) is recorded in the reliable experiment report rather than
being applied by the production evaluator. Therefore baseline production
metrics and candidate experiment metrics must not be presented as the same
deployed inference path. Do not promote the candidate until model, ordered
feature list, and threshold are packaged and validated together.
The production and candidate manifests each match their corresponding saved
models, but their feature sets differ.

## Single-record inference

`nsl_kdd_ids.pipeline.predict_record(record, model_config="baseline")`
accepts one mapping containing exactly the 41 NSL-KDD input features (not the
attack label or difficulty field). The categorical values
`protocol_type`, `service`, and `flag` must be non-empty strings; all other
values must be finite numeric scalars. Missing or extra keys, booleans in
numeric fields, and invalid value types raise an error. Feature selection and
ordering come from the selected configuration's saved manifest, and the
already-fitted model pipeline applies its own preprocessing. Unknown
categorical values follow the saved encoder's `handle_unknown="ignore"`
behavior.

The result contains `model_config`, `decision_threshold`,
`threshold_semantics`, `binary_prediction` (`Normal` or `Attack`), and
`attack_family` (`DoS`, `Probe`, `R2L`, or `U2R` only for a detected attack;
otherwise `null`). It does not return a probability or claim calibrated
real-world risk.

The default `baseline` configuration uses the saved production baseline
models and scikit-learn's default binary `.predict()` behavior. Its reported
0.5 threshold is informational; exact ties retain the estimator's native
decision semantics. The optional `grouped_candidate` configuration loads its
own binary/family models and ordered manifest. It requires an explicit
`threshold=` argument on every call and applies the candidate positive-class
`predict_proba` cutoff, which is not a calibrated real-world risk; the value
0.28 in the historical experiment report is not implicitly applied or
promoted. Neither configuration changes the existing production evaluation
path.

Illustrative call:

```python
from nsl_kdd_ids.pipeline import FEATURE_COLUMNS, predict_record

record = {feature: 0 for feature in FEATURE_COLUMNS}
record.update(protocol_type="tcp", service="http", flag="SF")
result = predict_record(record)
```

For this synthetic record, the returned mapping has the baseline configuration,
an informational 0.5 decision threshold, and a `Normal`/`Attack` decision with
a family only when it is `Attack`. The actual synthetic prediction depends on
the saved model artifacts. This interface is a reproducibility contract, not
evidence of deployment readiness or real-world intrusion detection.

## Run the local demo interface

The demo uses only Python's standard library for its local HTTP server and the
project's existing dependencies for model inference; no web framework is
required. From the repository root, start it with:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.demo
```

Open `http://127.0.0.1:8501` in a browser. Use **Load training sample** to
populate the form from the first record in `dataset/NSL-KDD/KDDTrain+.txt`, or
enter all 41 features yourself; then choose **Classify record**. The interface
validates all values and sends one JSON record to a local endpoint, which calls
`predict_record(..., model_config="baseline")`. Model preprocessing, selected
features, and prediction logic remain in the shared pipeline; the interface
does not offer candidate selection, file uploads, or code execution. Stop the
server with Ctrl+C.

Architecture: browser form → loopback-only Python standard-library HTTP server
→ feature validation → saved baseline `predict_record` contract → rendered
binary decision and optional family. The sample endpoint reads the training
file only; the official test file is not used by the demo. It is a single-record
offline demonstration, not live traffic monitoring, an alerting service, or a
security guarantee. A local prediction is only as representative as the
NSL-KDD-derived model and its input data.

## Outputs

Generated files are written to ignored directories:

```text
dataset/processed/              Parquet train/test splits and preprocessing report
artifacts/feature_selection/    Feature scores and selected feature list
artifacts/models/               Binary models, metrics, and confusion matrices
artifacts/attack_type_model/    Attack-family model and evaluation report
artifacts/explainability/       Aggregated feature scores and plots
artifacts/evaluation/           End-to-end predictions and system report
artifacts/models/controlled_experiment_grouped_validation/  Audited candidate models
```

All metrics are measured on `KDDTest+`, not a random split of training records.
Results should not be compared directly to scores from CIC-IDS2017: the datasets,
attack taxonomy, feature sets, and split methodology differ.

The historical random-row experiment writes `controlled_experiment_*` files.
The audited grouped-validation experiment writes `reliable_experiment_report.json`,
`reliable_experiment_comparison.csv`, and candidate test predictions under
`artifacts/evaluation/`. Its validation metrics and test metrics are reported
separately. U2R has very low support, so its measured recall has substantial
statistical uncertainty.

### Historical baseline results

The original run on the 22,544 official test records reported binary Random
Forest 0.765
accuracy, 0.967 precision, and 0.608 recall. The complete two-stage system
achieved 0.755 accuracy and 0.499 macro F1. R2L and U2R recall were especially
low (0.020 and 0.030); these families are rare in `KDDTrain+`, and some attack
names in `KDDTest+` do not occur in training. Refer to the `reliable_experiment`
report for the audited grouped-validation candidate comparison; do not interpret
the historical random-row validation score as an estimate of unseen-attack
performance.

## Repository structure

```text
.
├── dataset/NSL-KDD/             NSL-KDD official train/test text files
├── nsl_kdd_ids/pipeline.py      Shared implementation and command-line runner
├── notebooks/                   Ordered exploration, training, and evaluation notebooks
├── tests/                       Focused data and label-mapping tests
├── pyproject.toml
└── README.md
```
