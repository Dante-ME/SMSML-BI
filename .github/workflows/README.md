# Continuous Integration

Two GitHub Actions workflows cover this project. Both run on `ubuntu-latest`
with **Python 3.12** and cached pip dependencies, and both fail the run as soon
as any step exits non-zero.

| Workflow | File | Triggers | Typical duration |
|---|---|---|---|
| **ML CI Pipeline** | `ml-ci.yml` | push to `main`, pull request to `main`, manual | ~3-5 min |
| **ML Hyperparameter Tuning** | `ml-tuning.yml` | manual only | ~10-20 min |

They are split on purpose. The baseline pipeline is fast enough to gate every
pull request; a `RandomizedSearchCV` over 50 candidates x 5 folds takes minutes
on a 2-core runner, which is too slow to sit in front of every change.

---

## 1. ML CI Pipeline (`ml-ci.yml`)

The everyday pipeline: it rebuilds everything from the raw CSV and proves the
project still runs end to end on a clean machine.

### Triggers

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

A new push to the same branch cancels the run already in flight
(`concurrency.cancel-in-progress`), so only the newest commit is tested.

### What it does

1. **Checkout repository** — `actions/checkout@v4`.
2. **Set up Python 3.12** — `actions/setup-python@v5` with `cache: pip`, keyed on
   `experiment/requirements.txt` and `training/requirements.txt`. The first run
   populates the cache; later runs reuse the wheels.
3. **Install dependencies** — installs both requirements files, so CI tests the
   environment the project actually declares rather than a guessed subset.
4. **Show environment** — prints `python --version` and `pip list` for debugging.
5. **Run preprocessing** — `python experiment/automate.py`, regenerating
   `data/processed/` from `data/raw/`.
6. **Verify the processed dataset** — checks `train.csv`, `test.csv`,
   `heart_failure_preprocessed.csv` and `metadata.json` all exist and are
   non-empty; otherwise emits `::error::` and exits 1.
7. **Train baseline** — `python training/modelling.py`.
8. **Verify the trained model** — checks the `.joblib` model, its metrics JSON,
   the confusion matrix, the ROC curve and the classification report.
9. **Check the MLflow run was recorded** — verifies `mlruns/mlflow.db` exists.
10. **Publish metrics to the job summary** — renders a train/test metrics table
    straight onto the run's summary page, so results are visible without
    downloading anything.
11. **Upload artifacts** — four separate artifacts (below).

### Failure behaviour

GitHub Actions stops a job at the first step that exits non-zero, and every
`run` block uses `bash` with `set -euo pipefail`, so an error inside a
multi-line script fails the step rather than being swallowed. If a Python script
raises, the job goes red immediately and later steps are skipped.

The artifact uploads use `if: always()`, so a failed run still uploads whatever
was produced — that is usually what you need to diagnose it.

---

## 2. ML Hyperparameter Tuning (`ml-tuning.yml`)

Manual, parameterised tuning that produces the tuned model and a fair
baseline-vs-tuned comparison.

### Inputs

| Input | Type | Default | Meaning |
|---|---|---|---|
| `model` | choice | `random_forest` | `random_forest`, `gradient_boosting` or `logistic_regression` |
| `n_iter` | string | `50` | Number of `RandomizedSearchCV` candidates |
| `cv_folds` | string | `5` | Stratified cross-validation folds |
| `scoring` | string | `roc_auc` | Metric optimised during the search |

`n_iter` and `cv_folds` are validated in a dedicated step before any training
starts, so a typo fails in seconds rather than after a long install.

### What it does

Same setup steps, then: preprocessing → **baseline training** → tuning. The
baseline run is not redundant: `modelling_tuning.py` reads
`training/models/baseline_logistic_regression_metrics.json` to build the
comparison table, so the workflow regenerates it on the clean runner.

The job summary shows the winning hyperparameters and a baseline-vs-tuned table
with each delta marked *improved* / *same* / *worse*.

---

## How to trigger a workflow manually

Both workflows expose `workflow_dispatch`.

**From the GitHub UI**

1. Open the repository → **Actions** tab.
2. Pick **ML CI Pipeline** or **ML Hyperparameter Tuning** in the left sidebar.
3. Click **Run workflow** (top right).
4. Choose the branch, fill in the inputs (tuning only), then **Run workflow**.

> The **Run workflow** button only appears once the workflow file exists on the
> repository's **default branch**. After the first push to `main`, refresh the
> Actions tab.

**From the GitHub CLI**

```bash
gh workflow run ml-ci.yml --ref main

gh workflow run ml-tuning.yml --ref main \
  -f model=random_forest \
  -f n_iter=50 \
  -f cv_folds=5 \
  -f scoring=roc_auc

# watch the newest run
gh run watch
```

---

## Expected outputs

### Job summary

Every run writes a metrics table to its summary page (**Actions** → the run →
scroll past the job list). The CI pipeline shows train/test accuracy, precision,
recall, F1, ROC-AUC and average precision plus the cross-validated ROC-AUC; the
tuning workflow shows the best hyperparameters and the comparison table.

### Downloadable artifacts

Found at the bottom of a run's page under **Artifacts**. `<n>` is the run number.

| Artifact | Contents | Retention |
|---|---|---|
| `baseline-model-<n>` | `training/models/*.joblib` and `*_metrics.json` | 30 days |
| `evaluation-reports-<n>` | All of `training/reports/` — confusion matrix PNG, ROC curve PNG, classification report, coefficients CSV | 30 days |
| `processed-dataset-<n>` | All of `data/processed/` | 14 days |
| `mlruns-<n>` | The MLflow store: `mlflow.db` plus logged models and artifacts | 14 days |

The tuning workflow produces the same set, named
`tuned-model-<model>-<n>`, `tuning-reports-<model>-<n>` and
`mlruns-tuning-<model>-<n>`, and its reports artifact additionally contains
`*_cv_results.csv`, `*_best_params.json`, `*_feature_importance.csv` and
`baseline_vs_tuned.csv`.

### Inspecting a downloaded MLflow store

```bash
unzip mlruns-42.zip -d ./ci-mlruns
mlflow ui --backend-store-uri sqlite:///ci-mlruns/mlflow.db
```

---

## Verifying the workflows on GitHub

1. **Push the workflow files to `main`.** Workflows only register once they are
   on the default branch.

   ```bash
   git add .github/workflows
   git commit -m "add CI workflows"
   git push origin main
   ```

2. **Watch the automatic run.** The push to `main` triggers *ML CI Pipeline*
   immediately. Open **Actions** and click the newest run.

3. **Check each step is green.** Expand *Verify the processed dataset was
   produced* and *Verify the trained model and reports exist* — both print an
   `OK <path> (<bytes>)` line per required file.

4. **Read the job summary** for the metrics table. On a healthy run the baseline
   test ROC-AUC is about **0.86**.

5. **Download an artifact** and confirm the `.joblib` model and the PNG plots
   open correctly.

6. **Test the manual trigger** using the steps above, and confirm the tuning run
   reports a tuned test ROC-AUC around **0.90**, above the baseline.

7. **Test the failure path** (optional, on a branch): open a pull request that
   breaks something on purpose — for example renaming the target column in
   `data/raw/` — and confirm the run goes red at the verification step with the
   `::error::` annotation shown inline on the PR.

### Note on committed outputs

`.gitignore` excludes `mlruns/` and `*.joblib`, so trained models and the MLflow
store are **not** committed — they exist only as workflow artifacts. That keeps
the repository small; download the artifacts when you need the binaries. The
metrics JSON files and the report PNG/CSV/TXT files are committed normally.

### Note on install size

CI installs both `experiment/requirements.txt` and `training/requirements.txt`,
which pulls the Jupyter stack even though CI never executes the notebook. This
is deliberate — it tests the declared environments rather than a hand-picked
subset. If you want a leaner run, split the notebook-only pins out of
`experiment/requirements.txt` into a separate file and install just the core
ones here.
