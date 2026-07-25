"""Hyperparameter tuning with RandomizedSearchCV, tracked with MLflow.

Searches a Random Forest (default) over a randomised hyperparameter space using
stratified cross-validation on the *training* split only, then evaluates the
single best estimator once on the held-out test set. The result is compared
against the baseline produced by ``modelling.py`` — same data, same split, same
metrics — so the comparison is fair.

Run from the project root or from training/ — both work:

    python training/modelling_tuning.py
    cd training && python modelling_tuning.py

Optional overrides::

    python training/modelling_tuning.py --model gradient_boosting --n-iter 40
    python training/modelling_tuning.py --model logistic_regression --scoring f1

MLflow logs to ``mlruns/`` at the project root (override with MLFLOW_TRACKING_URI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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

BASELINE_METRICS_FILE = MODELS_DIR / "baseline_logistic_regression_metrics.json"

# Stable entry point for the serving stage, kept pointing at the best run so far.
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_METRICS_FILE = MODELS_DIR / "best_model_metrics.json"
PROMOTION_METRIC = "test_roc_auc"


def get_search_space(name: str, random_state: int = RANDOM_STATE):
    """Return the (estimator, parameter distributions) pair for a model family.

    Distributions are sampled by RandomizedSearchCV, which explores a wide space
    at a fixed budget instead of exhaustively walking a grid.
    """
    if name == "random_forest":
        estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        space = {
            "n_estimators": randint(100, 800),
            "max_depth": [None, 3, 5, 8, 12, 20],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=random_state)
        space = {
            "n_estimators": randint(50, 500),
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 6),
            "min_samples_leaf": randint(1, 10),
            "subsample": uniform(0.6, 0.4),  # 0.6 .. 1.0
        }
    elif name == "logistic_regression":
        # liblinear supports both l1 and l2 and is stable on a dataset this small.
        estimator = LogisticRegression(solver="liblinear", max_iter=2000, random_state=random_state)
        space = {
            "C": loguniform(1e-3, 1e2),
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
        }
    else:
        raise ValueError(f"Unknown model '{name}'.")

    return estimator, space


def extract_importances(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Return a feature-importance table for tree models or a linear model."""
    if hasattr(model, "feature_importances_"):
        values, column = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        values, column = model.coef_[0], "coefficient"
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, column: values})
        .sort_values(column, key=abs, ascending=False)
        .reset_index(drop=True)
    )


def load_baseline_metrics() -> dict | None:
    """Read the baseline summary written by modelling.py, if it has been run."""
    if not BASELINE_METRICS_FILE.is_file():
        print(f"[warn] baseline metrics not found at {BASELINE_METRICS_FILE}")
        print("[warn] run `python training/modelling.py` first to enable the comparison")
        return None
    try:
        return json.loads(BASELINE_METRICS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[warn] {BASELINE_METRICS_FILE} is not valid JSON, skipping the comparison")
        return None


def promote_best_model(model, summary: dict, metric: str = PROMOTION_METRIC, force: bool = False) -> bool:
    """Update ``models/best_model.joblib`` only if this run is actually better.

    Without this guard the last script to finish would win, so tuning a weaker
    model family would silently demote the model the serving stage picks up.
    """
    candidate = summary.get(metric)
    if candidate is None:
        print(f"[promote] '{metric}' unavailable for this run, leaving best_model.joblib untouched")
        return False

    incumbent = None
    if BEST_MODEL_METRICS_FILE.is_file():
        try:
            incumbent = json.loads(BEST_MODEL_METRICS_FILE.read_text(encoding="utf-8")).get(metric)
        except json.JSONDecodeError:
            print(f"[warn] {BEST_MODEL_METRICS_FILE} is not valid JSON, treating it as absent")

    if force or incumbent is None or candidate > incumbent:
        joblib.dump(model, BEST_MODEL_PATH)
        save_metrics_json(summary, BEST_MODEL_METRICS_FILE)
        previous = "none" if incumbent is None else f"{incumbent:.4f}"
        print(f"[promote] new best model ({metric}: {previous} -> {candidate:.4f})")
        return True

    print(
        f"[promote] kept the existing best model "
        f"({metric}: incumbent {incumbent:.4f} >= candidate {candidate:.4f})"
    )
    return False


def build_comparison(baseline: dict | None, tuned: dict) -> pd.DataFrame | None:
    """Build a baseline-vs-tuned table over the shared test metrics."""
    if baseline is None:
        return None

    shared = sorted(
        k for k in tuned if k.startswith("test_") and k in baseline
    )
    if not shared:
        return None

    rows = [
        {
            "metric": key.replace("test_", ""),
            "baseline": round(float(baseline[key]), 4),
            "tuned": round(float(tuned[key]), 4),
            "delta": round(float(tuned[key]) - float(baseline[key]), 4),
        }
        for key in shared
    ]
    return pd.DataFrame(rows)


def run_tuning(
    model_name: str = "random_forest",
    n_iter: int = 50,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    run_name: str | None = None,
    force_promote: bool = False,
) -> dict:
    """Run the randomised search, log everything to MLflow and save artifacts."""
    model_tag = f"tuned_{model_name}"
    run_name = run_name or model_tag

    X_train, X_test, y_train, y_test, target = load_processed_data()
    feature_names = list(X_train.columns)

    setup_mlflow(EXPERIMENT_NAME)

    estimator, space = get_search_space(model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,  # fixed seed -> the same candidates every run
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score=0.0,  # a bad candidate scores 0 instead of killing the search
    )

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[tuning] {model_name}: {n_iter} candidates x {cv_folds} folds, scoring={scoring}")
        # The search only ever sees X_train — the test set stays untouched.
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        train_metrics = compute_metrics(best_model, X_train, y_train, prefix="train")
        test_metrics = compute_metrics(best_model, X_test, y_test, prefix="test")

        # --- parameters ---
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_params(
            {
                "model_type": type(best_model).__name__,
                "stage": "tuning",
                "search": "RandomizedSearchCV",
                "n_iter": n_iter,
                "cv_folds": cv_folds,
                "scoring": scoring,
                "target": target,
                "n_features": X_train.shape[1],
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "random_state": RANDOM_STATE,
            }
        )

        # --- metrics ---
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(
            {
                f"cv_best_{scoring}": float(search.best_score_),
                f"cv_best_{scoring}_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
            }
        )

        # --- model ---
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_train.head(5).astype("float64"),
        )

        # --- artifacts ---
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        artifacts = save_evaluation_artifacts(best_model, X_test, y_test, REPORTS_DIR, model_tag)

        # Full search history: every candidate, its params and its CV score.
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv_path = REPORTS_DIR / f"{model_tag}_cv_results.csv"
        cv_results.to_csv(cv_path, index=False)
        artifacts.append(cv_path)

        best_params_path = REPORTS_DIR / f"{model_tag}_best_params.json"
        best_params_path.write_text(
            json.dumps({k: _jsonable(v) for k, v in search.best_params_.items()}, indent=2),
            encoding="utf-8",
        )
        artifacts.append(best_params_path)

        importances = extract_importances(best_model, feature_names)
        if importances is not None:
            imp_path = REPORTS_DIR / f"{model_tag}_feature_importance.csv"
            importances.to_csv(imp_path, index=False)
            artifacts.append(imp_path)

        # --- baseline vs tuned ---
        summary = {
            "run_id": run.info.run_id,
            "model_type": type(best_model).__name__,
            "stage": "tuning",
            "search": "RandomizedSearchCV",
            "n_iter": n_iter,
            "scoring": scoring,
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
            f"cv_best_{scoring}": float(search.best_score_),
            **train_metrics,
            **test_metrics,
        }

        comparison = build_comparison(load_baseline_metrics(), summary)
        if comparison is not None:
            comp_path = REPORTS_DIR / "baseline_vs_tuned.csv"
            comparison.to_csv(comp_path, index=False)
            artifacts.append(comp_path)
            for _, row in comparison.iterrows():
                mlflow.log_metric(f"delta_test_{row['metric']}", float(row["delta"]))

        log_artifacts(artifacts)
        mlflow.set_tags({"stage": "tuning", "dataset": "heart_failure_clinical_records"})

        # --- local copies ---
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_tag}.joblib"
        joblib.dump(best_model, model_path)
        save_metrics_json(summary, MODELS_DIR / f"{model_tag}_metrics.json")

        promoted = promote_best_model(best_model, summary, force=force_promote)
        mlflow.set_tag("promoted_to_best", str(promoted).lower())

        # --- console report ---
        print(f"\n[tuning] best CV {scoring}: {search.best_score_:.4f}")
        print("[tuning] best params:")
        for key, value in sorted(search.best_params_.items()):
            print(f"    {key:22s} {value}")

        print_metrics(train_metrics, "Train metrics (in-sample, for reference only)")
        print_metrics(test_metrics, "Test metrics (held-out)")

        if comparison is not None:
            print("\n--- Baseline vs tuned (test set) ---")
            print(comparison.to_string(index=False))

        print(f"\n[saved] model      : {model_path}")
        print(f"[saved] best model : {BEST_MODEL_PATH}")
        print(f"[saved] metrics    : {MODELS_DIR / f'{model_tag}_metrics.json'}")
        print(f"[saved] reports    : {REPORTS_DIR}")
        print(f"[mlflow] run id    : {run.info.run_id}")

    return summary


def _jsonable(value):
    """Convert numpy scalars so json.dumps accepts them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the heart failure classifier.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Model family to tune.",
    )
    parser.add_argument("--n-iter", type=int, default=50, help="Number of sampled candidates.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--scoring", default="roc_auc", help="Metric optimised during the search.")
    parser.add_argument("--run-name", default=None, help="MLflow run name.")
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help=f"Overwrite models/best_model.joblib even if this run has a lower {PROMOTION_METRIC}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_iter < 1:
        print("[error] --n-iter must be at least 1", file=sys.stderr)
        return 2
    if args.cv_folds < 2:
        print("[error] --cv-folds must be at least 2", file=sys.stderr)
        return 2
    try:
        run_tuning(
            model_name=args.model,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            scoring=args.scoring,
            run_name=args.run_name,
            force_promote=args.force_promote,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
