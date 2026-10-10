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

Dataset files are intentionally excluded from Git. Place the official full
`KDDTrain+` and `KDDTest+` text files at the paths above before running the
pipeline; `.arff` copies and smaller subsets are not used. NSL-KDD is a
benchmark associated with the Canadian Institute for Cybersecurity at the
University of New Brunswick. This repository does not redistribute the data
or provide an unverified download URL. Obtain the official files from an
authorized dataset source and check its current access and reuse terms; place
the files at the exact paths shown above.

Each row contains 41 connection features, an attack label, and a difficulty
score. The difficulty score is excluded from model input. The three categorical
features (`protocol_type`, `service`, and `flag`) are one-hot encoded; numeric
features are standardized inside each model pipeline using training data only.
Random Forest feature importance selects 20 of the 41 inputs for the saved
baseline models; that ordered feature list is stored with the model artifacts.
The controlled candidate has its own feature selection and matching manifest.

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

Requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From the
repository root, these PowerShell commands create a Python 3.12 environment
and install dependencies from `pyproject.toml` and the tracked `uv.lock`:

```powershell
uv venv --python 3.12
uv sync --locked --extra notebook
```

### Minimum setup for the demo

The demo needs these generated baseline artifacts:

```text
artifacts/models/random_forest.joblib
artifacts/attack_type_model/random_forest_attack_type.joblib
artifacts/feature_selection/selected_features.json
```

They are produced by the complete baseline pipeline below. The **Load training
sample** button also needs `dataset/NSL-KDD/KDDTrain+.txt`; without it, a
reviewer can enter a record manually. The demo does not need `KDDTest+.txt`,
candidate artifacts, or experiment reports.

### Reproduce the baseline and candidate evaluation

Dataset files and generated artifacts are not tracked. For a fresh checkout,
first obtain and place both official dataset files as described above. Run the
complete baseline pipeline to read the official split, create processed
datasets and feature manifest, train the baseline models, and generate baseline
evaluation artifacts:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step all
```

The full test suite also inspects the baseline predictions and saved
grouped-candidate artifacts. Generate the candidate models and reliable
comparison report after the baseline pipeline:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step experiment
```

`--step experiment` selects using a validation split from `KDDTrain+` and then
evaluates the finalized baseline/candidate on the official test split for
reporting. Never use those test results to tune or select models.

To regenerate the training-only grouped-validation reports as well, run these
optional commands; they use `KDDTrain+` only:

```powershell
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step repeated-validation
.\.venv\Scripts\python.exe -m nsl_kdd_ids.pipeline --step nested-validation
```

After generating the required artifacts, run the tests and launch the demo:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m nsl_kdd_ids.demo
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

Open `http://127.0.0.1:8501` in a browser. Select **Try Normal Connection**
or **Try Attack Connection**, then choose **Analyze Connection**. The app
selects a matching historical row using its actual label in
`KDDTrain+.txt`, passes its 41 features through the baseline
`predict_record` contract, and shows the predicted class separately from the
dataset label and whether they match. Manual entry remains available under
**Advanced: Manual Feature Entry**. The interface validates all values and
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
This sentence applies to official-test evaluation reports only: repeated and
nested grouped-validation reports use `KDDTrain+` validation folds, not
`KDDTest+`. Results should not be compared directly to scores from CIC-IDS2017:
the datasets, attack taxonomy, feature sets, and split methodology differ.

The historical random-row experiment writes `controlled_experiment_*` files.
The audited grouped-validation experiment writes `reliable_experiment_report.json`,
`reliable_experiment_comparison.csv`, and candidate test predictions under
`artifacts/evaluation/`. Its validation metrics and test metrics are reported
separately. U2R has very low support, so its measured recall has substantial
statistical uncertainty.

### Saved official-test evaluation results

The audited `reliable_experiment_report.json` evaluates the saved baseline and
the grouped-validation candidate on the same 22,544 official test rows after
model/threshold selection was completed using `KDDTrain+`. Values below are
read from that report; they are dataset-specific and do not establish
real-world performance.

| Official-test metric | Baseline | Grouped candidate |
| --- | ---: | ---: |
| Binary accuracy | 76.50% | 82.18% |
| Binary precision | 96.73% | 96.40% |
| Binary attack recall | 60.77% | 71.36% |
| Binary F1 | 74.65% | 82.01% |
| Binary ROC-AUC | 96.25% | 96.22% |
| Binary false-positive rate | 2.72% | 3.52% |
| Complete-system accuracy | 75.51% | 80.37% |
| Complete-system macro F1 | 49.89% | 57.03% |
| Complete-system weighted F1 | 70.98% | 77.04% |

Rare-family complete-system results (precision / recall / F1; support) were:

| Family | Baseline | Grouped candidate |
| --- | --- | --- |
| R2L | 90.48% / 1.98% / 3.87%; 2,885 | 97.23% / 12.17% / 21.63%; 2,885 |
| U2R | 50.00% / 2.99% / 5.63%; 67 | 25.00% / 5.97% / 9.64%; 67 |

The candidate improved several metrics on this benchmark but increased the
normal-traffic false-positive rate. R2L and especially U2R remain weak; U2R
support is very small. The candidate remains separate from the production
baseline and is not selected automatically by the demo.

Training-only nested grouped validation gives a separate view of split
variability. Across three outer folds, the inner-binary-F1 policy averaged
92.63% attack recall (sample SD 6.00%) and 1.48% false-positive rate (SD
0.32%); inner recall subject to a 1% validation FPR constraint averaged 91.36%
recall (SD 7.04%) and 1.02% outer-fold FPR (SD 0.33%). The latter exceeded
1% in one outer fold. Five-class macro F1 averaged 44.23% (SD 19.04%) and
44.90% (SD 19.64%) respectively. One outer fold lacks Probe examples, and
rare-family support varies. These grouped folds hold attack names and identical
full-feature records together, but are not evidence of real-world generalization.

The historical baseline-only results were binary accuracy 76.50%, precision
96.73%, recall 60.77%, and complete-system accuracy 75.51% with macro F1
49.89%. These values match the baseline column above; the earlier random-row
validation score near 99.9% is not an estimate of performance on unseen attack
names.

The experiment JSON/CSV reports and model artifacts live under `artifacts/`,
which is ignored by Git. They are not present in a fresh clone; regenerate
them with the Setup commands before running the full artifact-dependent test
suite. Reports are reproducible outputs, not checked-in inputs.

## Repository structure

```text
.
├── dataset/NSL-KDD/             NSL-KDD official train/test text files
├── nsl_kdd_ids/pipeline.py      Shared implementation and command-line runner
├── notebooks/                   Ordered exploration, training, and evaluation notebooks
├── tests/                       Focused data and label-mapping tests
├── pyproject.toml
├── uv.lock
└── README.md
```
