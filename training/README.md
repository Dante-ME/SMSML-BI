# Training stage — Heart Failure Clinical Records

Trains and tunes a binary classifier on the preprocessed dataset produced by the
experiment stage (`data/processed/`), with everything tracked in MLflow.

## Prerequisites

The processed dataset must exist. If `data/processed/` is empty, generate it first:

```bash
python experiment/automate.py
```

Install the training dependencies:

```bash
pip install -r training/requirements.txt
```

## Files

| File | Role |
|---|---|
| `modelling.py` | Baseline run — Logistic Regression, MLflow autolog + explicit test metrics. |
| `modelling_tuning.py` | Hyperparameter tuning — `RandomizedSearchCV`, compares against the baseline. |
| `common.py` | Shared data loading, metrics, plots and MLflow setup used by both scripts. |
| `requirements.txt` | Pinned dependencies for this stage. |
| `models/` | Serialised models (`.joblib`) and metric summaries (`.json`), including the promoted `best_model.joblib`. |
| `reports/` | Confusion matrices, ROC curves, classification reports, CV results, comparison table. |

## How to run

Both scripts resolve paths from their own location, so the working directory
does not matter:

```bash
# from the project root
python training/modelling.py
python training/modelling_tuning.py

# or from inside training/
cd training
python modelling.py
python modelling_tuning.py
```

Run `modelling.py` **first** — `modelling_tuning.py` reads its metrics file to
build the baseline-vs-tuned comparison. The tuning script still runs without it,
it just skips the comparison and says so.

### Useful options

```bash
python training/modelling.py --cv-folds 10 --run-name baseline_v2

python training/modelling_tuning.py --model gradient_boosting --n-iter 40
python training/modelling_tuning.py --model logistic_regression --scoring f1
```

`--model` accepts `random_forest` (default), `gradient_boosting` and
`logistic_regression`. `--scoring` accepts any scikit-learn scorer name.

### The promoted best model

`models/best_model.joblib` is the stable entry point for the deployment stage.
It is **only** overwritten when a tuning run beats the recorded `test_roc_auc`
in `models/best_model_metrics.json` — so experimenting with a weaker model
family cannot silently demote what gets served. Use `--force-promote` to
override that guard deliberately.

## MLflow

Tracking is local. Because MLflow 3.x put the plain-directory file store into
maintenance mode, the backend is SQLite — but everything still lives inside
`mlruns/`:

```
mlruns/
├── mlflow.db          # runs, parameters, metrics
└── artifacts/         # models, plots, CSV/JSON artifacts
```

Both scripts log to a single experiment, **`heart-failure-classification`**, so
the baseline and the tuned run appear side by side in the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open <http://127.0.0.1:5000>.

To log somewhere else (a remote tracking server, DagsHub, a CI artifact store),
set the environment variable — no code change needed:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow   # bash
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/<user>/<repo>.mlflow"  # PowerShell
```

### What gets logged

| Category | Contents |
|---|---|
| **Parameters** | Model hyperparameters, model type, stage, target column, row/feature counts, CV folds, seed. Tuning also logs the winning params under `best__*`. |
| **Metrics** | `train_*` and `test_*` accuracy, precision, recall, F1, ROC-AUC, average precision; cross-validated score with its standard deviation; `delta_test_*` for tuned-minus-baseline. |
| **Model** | Logged via `mlflow.sklearn.log_model` with an input example and inferred signature, ready for `mlflow models serve`. |
| **Artifacts** | Confusion matrix, ROC curve, classification report, feature importances / coefficients, the full `cv_results` table, best params, and `baseline_vs_tuned.csv`. |

## Methodology

- **No leakage.** The features were already imputed, encoded and scaled during
  the experiment stage, where the transformers were fit on the training split
  only. Here, cross-validation runs on the training split and the test set is
  touched exactly once, for the final evaluation.
- **Reproducible.** `random_state=42` throughout — the model, the CV folds and
  the `RandomizedSearchCV` candidate sampling. Re-running produces the same
  numbers.
- **Fair comparison.** Both scripts load the same split via `common.py` and
  compute metrics with the same function, so the deltas are meaningful.
- **Imbalance.** The dataset is roughly 68/32. The baseline uses
  `class_weight="balanced"`; the tuning search treats `class_weight` as a
  hyperparameter. Neither resamples the data.

## Serving the trained model

The tuning script writes a stable `models/best_model.joblib` for the deployment
stage, and the MLflow model can be served directly from its run:

```bash
mlflow models serve -m "runs:/<run_id>/model" --port 5001 --no-conda
```

The run id is printed at the end of each script and shown in the MLflow UI.
