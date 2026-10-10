# ThreadAhead

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
```

`--step all` executes every stage in order. Later stages require the output of
earlier stages and report a clear error if a prerequisite is missing.

## Outputs

Generated files are written to ignored directories:

```text
dataset/processed/              Parquet train/test splits and preprocessing report
artifacts/feature_selection/    Feature scores and selected feature list
artifacts/models/               Binary models, metrics, and confusion matrices
artifacts/attack_type_model/    Attack-family model and evaluation report
artifacts/explainability/       Aggregated feature scores and plots
artifacts/evaluation/           End-to-end predictions and system report
```

All metrics are measured on `KDDTest+`, not a random split of training records.
Results should not be compared directly to scores from CIC-IDS2017: the datasets,
attack taxonomy, feature sets, and split methodology differ.

### Results from the completed run

On the 22,544 official test records, the binary Random Forest achieved 0.765
accuracy, 0.967 precision, and 0.608 recall. The complete two-stage system
achieved 0.755 accuracy and 0.499 macro F1. R2L and U2R recall were especially
low (0.020 and 0.030); these families are rare in `KDDTrain+`, and some attack
names in `KDDTest+` do not occur in training. The full per-model metrics and
test predictions are saved under `artifacts/`.

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
