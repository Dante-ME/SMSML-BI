"""FastAPI serving layer for the heart failure classifier.

Endpoints
---------
GET  /                     service description and where to find everything else
GET  /health               readiness probe (used by Prometheus alerting too)
POST /predict              run the model on raw clinical readings
POST /predict/preprocessed run the model on rows already scaled by automate.py
GET  /metrics              Prometheus exposition endpoint

Start it with::

    uvicorn deployment.app:app --host 0.0.0.0 --port 8000
    # or simply
    python deployment/app.py
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Lets the module be imported as `deployment.app` or run as `python deployment/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import prometheus_exporter as metrics
from inference import (
    InferenceError,
    ModelBundle,
    ModelNotLoadedError,
    example_record,
    load_bundle,
    predict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heart-failure-api")

MAX_BATCH_SIZE = 1000

# Populated during startup; stays None if the artifacts are missing so that the
# service can still report *why* it is unhealthy instead of refusing to boot.
bundle: ModelBundle | None = None


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #


class PatientRecord(BaseModel):
    """One patient's raw clinical readings.

    Field constraints are deliberately wide: they reject impossible values
    (negative age, a 0/1 flag set to 7) without rejecting genuine outliers such
    as a creatinine phosphokinase of 7861, which occurs in the training data.
    """

    age: float = Field(..., ge=0, le=120, description="Age in years.")
    anaemia: int = Field(..., ge=0, le=1, description="Decrease of red blood cells (0/1).")
    creatinine_phosphokinase: float = Field(..., ge=0, description="CPK enzyme in the blood (mcg/L).")
    diabetes: int = Field(..., ge=0, le=1, description="Whether the patient has diabetes (0/1).")
    ejection_fraction: float = Field(..., ge=0, le=100, description="Blood leaving the heart per contraction (%).")
    high_blood_pressure: int = Field(..., ge=0, le=1, description="Whether the patient is hypertensive (0/1).")
    platelets: float = Field(..., ge=0, description="Platelets in the blood (kiloplatelets/mL).")
    serum_creatinine: float = Field(..., ge=0, description="Serum creatinine in the blood (mg/dL).")
    serum_sodium: float = Field(..., ge=0, description="Serum sodium in the blood (mEq/L).")
    sex: int = Field(..., ge=0, le=1, description="Biological sex (0 = female, 1 = male).")
    smoking: int = Field(..., ge=0, le=1, description="Whether the patient smokes (0/1).")
    time: float = Field(..., ge=0, description="Follow-up period in days.")

    model_config = {"json_schema_extra": {"examples": [example_record()]}}


class ProcessedRecord(BaseModel):
    """One row already scaled by ``experiment/automate.py``.

    Deliberately unbounded: standard-scaled values are centred on zero and are
    routinely negative, so the medical ranges on :class:`PatientRecord` cannot
    apply here. That difference is exactly why this is a separate schema.
    """

    age: float
    anaemia: float
    creatinine_phosphokinase: float
    diabetes: float
    ejection_fraction: float
    high_blood_pressure: float
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: float
    smoking: float
    time: float


class PredictionRequest(BaseModel):
    """A batch of one or more raw patient records."""

    instances: list[PatientRecord] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="One entry per patient.",
    )

    model_config = {"json_schema_extra": {"examples": [{"instances": [example_record()]}]}}


class ProcessedPredictionRequest(BaseModel):
    """A batch of one or more already-preprocessed rows."""

    instances: list[ProcessedRecord] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="One entry per row, taken from data/processed/.",
    )


class PredictionItem(BaseModel):
    prediction: int = Field(..., description="0 = survived, 1 = death event.")
    label: str = Field(..., description="Human-readable class name.")
    probability: float | None = Field(None, description="Probability of class 1, when available.")
    probabilities: dict[str, float] | None = Field(None, description="Probability per class.")


class PredictionResponse(BaseModel):
    model_type: str
    preprocessed_input: bool
    n_instances: int
    predictions: list[PredictionItem]
    latency_ms: float


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once, at startup, rather than per request."""
    global bundle

    metrics.start_uptime_tracking()
    try:
        bundle = load_bundle()
        metrics.set_model_info(bundle.describe())
        logger.info("Loaded %s from %s", bundle.model_type, bundle.model_path)
        logger.info("Preprocessor loaded from %s", bundle.preprocessor_path)
    except FileNotFoundError as exc:
        # Start anyway: /health explains what is missing, which is far easier to
        # debug than a container that exits immediately.
        bundle = None
        metrics.set_model_unavailable()
        logger.error("Model could not be loaded:\n%s", exc)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Heart Failure Prediction API",
    description=(
        "Serves the promoted heart failure classifier from `training/models/best_model.joblib`. "
        "Raw clinical values are scaled with the fitted preprocessor before inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_http_requests(request: Request, call_next):
    """Count every HTTP request, labelled by route template rather than raw path."""
    response = await call_next(request)
    # route.path keeps the cardinality low (no per-request unique paths).
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)
    metrics.observe_http(request.method, endpoint, response.status_code)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 422 with a readable body, and count it as a prediction failure."""
    if request.url.path.startswith("/predict"):
        metrics.observe_failure(0.0, reason="validation_error")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
            "errors": [
                {"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
                for err in exc.errors()
            ],
        },
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/", tags=["general"])
def root() -> dict:
    """Service description and a map of the available endpoints."""
    return {
        "service": "Heart Failure Prediction API",
        "version": app.version,
        "status": "ready" if bundle is not None else "model_unavailable",
        "model": bundle.describe() if bundle is not None else None,
        "endpoints": {
            "GET /": "This description.",
            "GET /health": "Readiness probe.",
            "POST /predict": "Run the model on raw clinical readings.",
            "POST /predict/preprocessed": "Run the model on already-scaled rows.",
            "GET /metrics": "Prometheus metrics.",
            "GET /docs": "Interactive OpenAPI documentation.",
        },
    }


@app.get("/health", tags=["general"])
def health() -> JSONResponse:
    """Readiness probe.

    Returns 200 when the model can serve traffic and 503 when it cannot, so
    Prometheus (and any orchestrator) can alert on the status code alone.
    """
    ready = bundle is not None
    body = {
        "status": "healthy" if ready else "unhealthy",
        "model_loaded": ready,
        "uptime_seconds": round(metrics.uptime_seconds(), 3),
    }
    if ready:
        body["model_type"] = bundle.model_type
        body["preprocessor_loaded"] = bundle.expects_preprocessing
    else:
        body["detail"] = (
            "The model artifacts are missing. Run `python experiment/automate.py` "
            "and `python training/modelling_tuning.py`, then restart the API."
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body,
    )


RESPONSES = {
    422: {"description": "The request body failed validation."},
    500: {"description": "The model is loaded but inference failed."},
    503: {"description": "The model is not loaded."},
}


def _run_prediction(records: list[dict], preprocessed: bool) -> PredictionResponse:
    """Shared inference path for both predict routes, including metrics."""
    started = time.perf_counter()

    try:
        results = predict(bundle, records, preprocessed=preprocessed)
    except ModelNotLoadedError as exc:
        metrics.observe_failure(time.perf_counter() - started, reason="model_not_loaded")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except InferenceError as exc:
        metrics.observe_failure(time.perf_counter() - started, reason="inference_error")
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    latency = time.perf_counter() - started
    metrics.observe_prediction(latency, len(records), [r["label"] for r in results])

    return PredictionResponse(
        model_type=bundle.model_type,
        preprocessed_input=preprocessed,
        n_instances=len(results),
        predictions=[PredictionItem(**r) for r in results],
        latency_ms=round(latency * 1000, 3),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"], responses=RESPONSES)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Predict the death event from **raw clinical readings**.

    Values are scaled with the fitted preprocessor before inference, so send
    real measurements (`platelets: 265000`), not standardised ones.
    """
    return _run_prediction([item.model_dump() for item in request.instances], preprocessed=False)


@app.post(
    "/predict/preprocessed",
    response_model=PredictionResponse,
    tags=["inference"],
    responses=RESPONSES,
)
def predict_preprocessed_endpoint(request: ProcessedPredictionRequest) -> PredictionResponse:
    """Predict from rows that are **already scaled** by `experiment/automate.py`.

    Convenience route for replaying `data/processed/test.csv` straight through
    the model; it skips the preprocessor. Prefer `/predict` for real clients.
    """
    return _run_prediction([item.model_dump() for item in request.instances], preprocessed=True)


@app.get("/metrics", tags=["monitoring"], include_in_schema=True)
def metrics_endpoint() -> Response:
    """Prometheus exposition endpoint, scraped by the Prometheus server."""
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    # reload=False keeps the single-process behaviour the metrics assume.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
