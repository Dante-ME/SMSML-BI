"""Shared helpers for the training stage.

Both ``modelling.py`` (baseline) and ``modelling_tuning.py`` (hyperparameter
tuning) import from this module so that data loading, metric computation and
MLflow setup are identical between the two runs — that is what makes the
baseline-vs-tuned comparison fair.

Nothing here trains a model; it only prepares inputs and summarises outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Non-interactive backend: these scripts run headless (CI, terminal).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

# Same seed as the experiment stage, so the split produced there is reproduced
# here bit for bit if we ever need to rebuild it.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# parents[1] == project root, regardless of the current working directory, so
# the scripts behave the same whether launched from the root or from training/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DIR = PROJECT_ROOT / "training"
MODELS_DIR = TRAINING_DIR / "models"
REPORTS_DIR = TRAINING_DIR / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# One experiment holding both runs, so they can be compared side by side in the UI.
EXPERIMENT_NAME = "heart-failure-classification"

# Fallback target names, used only when metadata.json is unavailable.
TARGET_NAME_CANDIDATES = ("death_event", "target", "label", "class", "outcome", "y")


# --------------------------------------------------------------------------- #
# MLflow setup
# --------------------------------------------------------------------------- #


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Point MLflow at the local store under ``mlruns/`` at the project root.

    A SQLite backend is used rather than the plain-directory file store: MLflow
    3.x put the file store into maintenance mode and refuses it by default.
    Everything still lives inside ``mlruns/`` — ``mlflow.db`` holds the runs and
    ``mlruns/artifacts/`` holds the models and plots.

    Using absolute URIs means runs land in the same place no matter which
    directory the script was started from. Set ``MLFLOW_TRACKING_URI`` to
    override, e.g. to log against a remote server such as DagsHub.
    """
    import os

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    artifact_uri = None

    if not tracking_uri:
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_dir = MLRUNS_DIR / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
        artifact_uri = artifact_dir.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    # Set the artifact location at creation time; it cannot be changed later.
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)

    print(f"[mlflow] tracking uri : {tracking_uri}")
    print(f"[mlflow] experiment   : {experiment_name}")
    return tracking_uri


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _read_metadata() -> dict:
    """Read the metadata written by the experiment stage, if it exists."""
    path = PROCESSED_DIR / "metadata.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] {path} is not valid JSON, ignoring it")
    return {}


def _resolve_target(df: pd.DataFrame, metadata: dict) -> str:
    """Return the target column: metadata first, then a name match, then last column."""
    target = metadata.get("target")
    if target and target in df.columns:
        return target
    if target:
        print(f"[warn] metadata target '{target}' not present in the data, detecting instead")

    for candidate in TARGET_NAME_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def _missing_data_error() -> FileNotFoundError:
    return FileNotFoundError(
        f"No processed dataset found in {PROCESSED_DIR}.\n"
        f"Expected either 'train.csv' + 'test.csv', or a single preprocessed CSV.\n"
        f"Run the experiment stage first:\n"
        f"    python experiment/automate.py"
    )


def load_processed_data(
    processed_dir: Path = PROCESSED_DIR,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """Load the already-preprocessed dataset produced by ``experiment/automate.py``.

    Preference order:
      1. ``train.csv`` + ``test.csv`` — the split written by the experiment stage;
      2. any other preprocessed CSV in the folder, split here with the same seed;
      3. otherwise raise a message telling the user how to generate the data.

    The features are already imputed, encoded and scaled, so nothing is fitted
    on the data here — that keeps the test set untouched by training.
    """
    if not processed_dir.is_dir():
        raise _missing_data_error()

    metadata = _read_metadata()
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"

    if train_path.is_file() and test_path.is_file():
        train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
        if train_df.empty or test_df.empty:
            raise ValueError(f"train.csv or test.csv is empty in {processed_dir}")
        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("train.csv and test.csv do not share the same columns.")

        target = _resolve_target(train_df, metadata)
        print(f"[data] using the split from the experiment stage ({train_path.name} / {test_path.name})")
    else:
        # Fall back to any single preprocessed CSV and rebuild the split here.
        candidates = sorted(
            p for p in processed_dir.glob("*.csv") if p.name not in {"train.csv", "test.csv"}
        )
        if not candidates:
            raise _missing_data_error()

        full_path = candidates[0]
        full_df = pd.read_csv(full_path)
        if full_df.empty:
            raise ValueError(f"Processed dataset is empty: {full_path}")

        target = _resolve_target(full_df, metadata)
        stratify = full_df[target] if full_df[target].value_counts().min() >= 2 else None
        train_df, test_df = train_test_split(
            full_df, test_size=TEST_SIZE, random_state=random_state, stratify=stratify
        )
        print(f"[data] train.csv/test.csv not found, re-split {full_path.name} with seed {random_state}")

    if target not in train_df.columns:
        raise ValueError(
            f"Target column '{target}' is missing. Available columns: {list(train_df.columns)}"
        )

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError(
            "The processed dataset still contains missing values. "
            "Re-run `python experiment/automate.py` to regenerate it."
        )

    print(f"[data] target={target!r} | train={X_train.shape} test={X_test.shape}")
    print(f"[data] class balance (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test, target


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def predict_scores(model, X: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class scores for ROC-AUC, or None if the model has none."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_metrics(model, X: pd.DataFrame, y: pd.Series, prefix: str = "test") -> dict[str, float]:
    """Compute the classification metrics for one split.

    ``zero_division=0`` keeps the run alive if a class is never predicted, which
    can happen with an unlucky hyperparameter combination during tuning.
    """
    y_pred = model.predict(X)
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, y_pred),
        f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y, y_pred, zero_division=0),
    }

    # ROC-AUC and average precision need scores, and at least two classes present.
    y_score = predict_scores(model, X)
    if y_score is not None and y.nunique() > 1:
        metrics[f"{prefix}_roc_auc"] = roc_auc_score(y, y_score)
        metrics[f"{prefix}_average_precision"] = average_precision_score(y, y_score)

    return {k: float(v) for k, v in metrics.items()}


def print_metrics(metrics: dict[str, float], title: str) -> None:
    print(f"\n--- {title} ---")
    for name, value in metrics.items():
        print(f"  {name:28s} {value:.4f}")


def save_evaluation_artifacts(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    tag: str,
) -> list[Path]:
    """Write the confusion matrix, ROC curve and classification report to disk.

    Returns the list of created files so the caller can log them to MLflow.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm_path = output_dir / f"{tag}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix — {tag}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    created.append(cm_path)

    # ROC curve (only meaningful when the model produces scores)
    y_score = predict_scores(model, X_test)
    if y_score is not None and y_test.nunique() > 1:
        roc_path = output_dir / f"{tag}_roc_curve.png"
        fig, ax = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(y_test, y_score, name=tag, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
        ax.set_title(f"ROC curve — {tag}")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)
        created.append(roc_path)

    # Text classification report
    report_path = output_dir / f"{tag}_classification_report.txt"
    report_path.write_text(
        classification_report(y_test, y_pred, digits=4, zero_division=0), encoding="utf-8"
    )
    created.append(report_path)

    return created


def log_artifacts(paths: list[Path], artifact_path: str = "evaluation") -> None:
    """Log a list of local files to the active MLflow run."""
    for path in paths:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def save_metrics_json(metrics: dict, path: Path) -> Path:
    """Persist metrics next to the model so results are readable without MLflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
