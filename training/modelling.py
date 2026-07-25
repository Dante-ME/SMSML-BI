"""Baseline training run — Logistic Regression, tracked with MLflow.

Trains a deliberately simple, defensible baseline on the preprocessed heart
failure dataset so that the tuned model in ``modelling_tuning.py`` has a
meaningful reference point. Logistic Regression is a sound default here: the
features are already scaled by the experiment stage, the task is binary, and
the coefficients stay interpretable for a clinical dataset.

Run from the project root or from training/ — both work:

    python training/modelling.py
    cd training && python modelling.py

Optional overrides::

    python training/modelling.py --run-name my_baseline --cv-folds 10

MLflow logs to ``mlruns/`` at the project root (override with MLFLOW_TRACKING_URI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python training/modelling.py` and `cd training && python modelling.py`
# to resolve `common` the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import (
    EXPERIMENT_NAME,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    compute_metrics,
    load_processed_data,
    log_artifacts,
    print_metrics,
    save_evaluation_artifacts,
    save_metrics_json,
    setup_mlflow,
)

MODEL_TAG = "baseline_logistic_regression"

# class_weight="balanced" counteracts the 68/32 class imbalance without
# resampling, which keeps the dataset itself untouched.
BASELINE_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}


def train_baseline(cv_folds: int = 5, run_name: str = MODEL_TAG) -> dict:
    """Train, evaluate and log the baseline model. Returns the test metrics."""
    X_train, X_test, y_train, y_test, target = load_processed_data()

    setup_mlflow(EXPERIMENT_NAME)

    # Autolog captures the estimator parameters and the fitted model for us;
    # the test metrics below are logged explicitly so their names are stable.
    mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)

    with mlflow.start_run(run_name=run_name) as run:
        model = LogisticRegression(**BASELINE_PARAMS)

        # Cross-validation on the TRAINING split only — the test set is never
        # touched until the final evaluation below.
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        model.fit(X_train, y_train)

        train_metrics = compute_metrics(model, X_train, y_train, prefix="train")
        test_metrics = compute_metrics(model, X_test, y_test, prefix="test")

        # --- parameters ---
        mlflow.log_params({f"model__{k}": v for k, v in BASELINE_PARAMS.items()})
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "stage": "baseline",
                "target": target,
                "n_features": X_train.shape[1],
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "cv_folds": cv_folds,
                "random_state": RANDOM_STATE,
            }
        )

        # --- metrics ---
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(
            {
                "cv_roc_auc_mean": float(np.mean(cv_scores)),
                "cv_roc_auc_std": float(np.std(cv_scores)),
            }
        )

        # --- model ---
        # astype(float64) keeps the logged signature all-double, so the serving
        # stage does not reject integer-looking payloads at inference time.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5).astype("float64"),
        )

        # --- artifacts ---
        artifacts = save_evaluation_artifacts(model, X_test, y_test, REPORTS_DIR, MODEL_TAG)

        # Coefficients are the interpretable payoff of choosing a linear baseline.
        coef_path = REPORTS_DIR / f"{MODEL_TAG}_coefficients.csv"
        coefficients = (
            X_train.columns.to_frame(index=False, name="feature")
            .assign(coefficient=model.coef_[0])
            .reindex(columns=["feature", "coefficient"])
            .sort_values("coefficient", key=abs, ascending=False)
        )
        coefficients.to_csv(coef_path, index=False)
        artifacts.append(coef_path)

        log_artifacts(artifacts)

        mlflow.set_tags({"stage": "baseline", "dataset": "heart_failure_clinical_records"})

        # --- local copies, so the results are usable without the MLflow UI ---
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{MODEL_TAG}.joblib"
        joblib.dump(model, model_path)

        summary = {
            "run_id": run.info.run_id,
            "model_type": "LogisticRegression",
            "stage": "baseline",
            "params": BASELINE_PARAMS,
            "cv_roc_auc_mean": float(np.mean(cv_scores)),
            "cv_roc_auc_std": float(np.std(cv_scores)),
            **train_metrics,
            **test_metrics,
        }
        save_metrics_json(summary, MODELS_DIR / f"{MODEL_TAG}_metrics.json")

        print_metrics(train_metrics, "Train metrics (in-sample, for reference only)")
        print_metrics(test_metrics, "Test metrics (held-out)")
        print(f"\n  cv_roc_auc                   {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"\n[saved] model   : {model_path}")
        print(f"[saved] metrics : {MODELS_DIR / f'{MODEL_TAG}_metrics.json'}")
        print(f"[saved] reports : {REPORTS_DIR}")
        print(f"[mlflow] run id : {run.info.run_id}")

    mlflow.sklearn.autolog(disable=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline heart failure classifier.")
    parser.add_argument("--run-name", default=MODEL_TAG, help="MLflow run name.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds on the training split.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cv_folds < 2:
        print("[error] --cv-folds must be at least 2", file=sys.stderr)
        return 2
    try:
        train_baseline(cv_folds=args.cv_folds, run_name=args.run_name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
