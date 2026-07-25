"""Model loading and prediction logic for the serving API.

Kept separate from ``app.py`` so the inference path can be tested without
starting a web server.

Important: the promoted model was trained on the **preprocessed** feature space
produced by ``experiment/automate.py`` — continuous columns are standard-scaled.
Raw clinical values (``platelets=265000``) must therefore go through the fitted
``preprocessor.joblib`` before reaching the model, otherwise the predictions are
meaningless. That is what :func:`predict` does by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "training" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

# The promoted model comes first; the others are fallbacks so the API still
# starts if only part of the training stage has been run.
MODEL_CANDIDATES = (
    MODELS_DIR / "best_model.joblib",
    MODELS_DIR / "tuned_random_forest.joblib",
    MODELS_DIR / "baseline_logistic_regression.joblib",
)

# Raw clinical fields accepted by /predict, in a fixed order.
FEATURE_NAMES = (
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
)

CLASS_LABELS = {0: "survived", 1: "death_event"}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model could be loaded."""


class InferenceError(RuntimeError):
    """Raised when the model is loaded but the prediction itself failed."""


@dataclass
class ModelBundle:
    """Everything the API needs to answer a prediction request."""

    model: object
    model_path: Path
    model_type: str
    preprocessor: object | None = None
    preprocessor_path: Path | None = None
    target: str = "death_event"
    metrics: dict = field(default_factory=dict)

    @property
    def expects_preprocessing(self) -> bool:
        return self.preprocessor is not None

    def describe(self) -> dict:
        """Small, JSON-safe summary used by ``/`` and ``/health``."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path.relative_to(PROJECT_ROOT)),
            "preprocessor_loaded": self.expects_preprocessing,
            "target": self.target,
            "n_features": len(FEATURE_NAMES),
            "features": list(FEATURE_NAMES),
            "test_roc_auc": self.metrics.get("test_roc_auc"),
            "test_f1": self.metrics.get("test_f1"),
            "trained_run_id": self.metrics.get("run_id"),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _find_model() -> Path:
    """Return the first available model file, preferring the promoted one."""
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path

    searched = "\n".join(f"  - {p}" for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No trained model found. Looked for:\n"
        f"{searched}\n"
        "Run the training stage first:\n"
        "    python training/modelling.py\n"
        "    python training/modelling_tuning.py"
    )


def _load_metrics(model_path: Path) -> dict:
    """Load the metrics JSON that sits next to the model, if present."""
    stem = "best_model" if model_path.stem == "best_model" else model_path.stem
    metrics_path = model_path.with_name(f"{stem}_metrics.json")
    if metrics_path.is_file():
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def load_bundle() -> ModelBundle:
    """Load the promoted model and the fitted preprocessor.

    Raises ``FileNotFoundError`` with actionable instructions when an artifact
    is missing — the API turns that into a clear startup failure rather than
    serving wrong numbers.
    """
    model_path = _find_model()
    model = joblib.load(model_path)

    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(
            f"The fitted preprocessor is missing: {PREPROCESSOR_PATH}\n"
            "The model expects scaled features, so serving raw values without it "
            "would produce wrong predictions.\n"
            "Regenerate it with:\n"
            "    python experiment/automate.py"
        )
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    target = "death_event"
    if METADATA_PATH.is_file():
        try:
            target = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("target", target)
        except json.JSONDecodeError:
            pass

    return ModelBundle(
        model=model,
        model_path=model_path,
        model_type=type(model).__name__,
        preprocessor=preprocessor,
        preprocessor_path=PREPROCESSOR_PATH,
        target=target,
        metrics=_load_metrics(model_path),
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def _to_frame(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the exact column order the transformers expect."""
    frame = pd.DataFrame(records)
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise InferenceError(f"Missing required feature(s): {missing}")
    return frame[list(FEATURE_NAMES)].astype("float64")


def predict(bundle: ModelBundle | None, records: list[dict], preprocessed: bool = False) -> list[dict]:
    """Run predictions for a batch of records.

    Args:
        bundle: the loaded model bundle.
        records: raw clinical readings, one dict per patient.
        preprocessed: set when the caller is sending rows already scaled by
            ``experiment/automate.py`` (e.g. straight from
            ``data/processed/test.csv``), which skips the preprocessor.
    """
    if bundle is None:
        raise ModelNotLoadedError("The model is not loaded; the service is not ready.")
    if not records:
        raise InferenceError("No instances were provided.")

    frame = _to_frame(records)

    try:
        if preprocessed:
            # Already-scaled rows: reorder to the model's own column order.
            features = frame[list(bundle.model.feature_names_in_)]
        else:
            features = bundle.preprocessor.transform(frame)
            features = pd.DataFrame(
                features, columns=bundle.preprocessor.get_feature_names_out(), index=frame.index
            )

        labels = bundle.model.predict(features)

        probabilities = None
        if hasattr(bundle.model, "predict_proba"):
            probabilities = bundle.model.predict_proba(features)
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as HTTP 500
        raise InferenceError(f"Prediction failed: {exc}") from exc

    results = []
    for index, label in enumerate(labels):
        label = int(label)
        item = {
            "prediction": label,
            "label": CLASS_LABELS.get(label, str(label)),
        }
        if probabilities is not None:
            classes = [int(c) for c in bundle.model.classes_]
            row = probabilities[index]
            item["probability"] = float(row[classes.index(1)]) if 1 in classes else None
            item["probabilities"] = {str(c): float(p) for c, p in zip(classes, row)}
        else:
            # Some estimators (e.g. a plain SVC) cannot produce probabilities.
            item["probability"] = None
            item["probabilities"] = None
        results.append(item)

    return results


def example_record() -> dict:
    """A realistic raw request body, used in the OpenAPI docs and the README."""
    return {
        "age": 75.0,
        "anaemia": 0,
        "creatinine_phosphokinase": 582.0,
        "diabetes": 0,
        "ejection_fraction": 20.0,
        "high_blood_pressure": 1,
        "platelets": 265000.0,
        "serum_creatinine": 1.9,
        "serum_sodium": 130.0,
        "sex": 1,
        "smoking": 0,
        "time": 4.0,
    }
