"""Prometheus metric definitions for the serving API.

All metrics live here so ``app.py`` only has to call the small helpers below.
The names follow the Prometheus convention: ``_total`` for counters and a base
unit suffix (``_seconds``) for timings.

Exposed on ``GET /metrics`` in the standard text exposition format, which is
what a Prometheus server scrapes.
"""

from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# Common prefix so every series from this service groups together in Grafana.
NAMESPACE = "heart_failure"

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

# How many prediction requests arrived, split by outcome.
PREDICTION_REQUESTS = Counter(
    f"{NAMESPACE}_prediction_requests_total",
    "Total number of prediction requests received.",
    ["status"],  # success | failure
)

# How many prediction requests failed, split by cause.
PREDICTION_FAILURES = Counter(
    f"{NAMESPACE}_prediction_failures_total",
    "Total number of failed prediction requests.",
    ["reason"],  # validation_error | model_not_loaded | inference_error
)

# End-to-end latency of a prediction request. The buckets are tuned for a small
# scikit-learn model served locally: most calls land in the low milliseconds.
PREDICTION_LATENCY = Histogram(
    f"{NAMESPACE}_prediction_latency_seconds",
    "Latency of prediction requests in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Number of patient records per request, useful for spotting batch behaviour.
PREDICTION_BATCH_SIZE = Histogram(
    f"{NAMESPACE}_prediction_batch_size",
    "Number of instances submitted per prediction request.",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# Distribution of what the model actually predicts — a cheap drift signal.
PREDICTIONS_BY_CLASS = Counter(
    f"{NAMESPACE}_predictions_by_class_total",
    "Total predictions produced, split by predicted class.",
    ["predicted_class"],  # survived | death_event
)

# Every HTTP request, so Grafana can chart traffic and error rates per endpoint.
HTTP_REQUESTS = Counter(
    f"{NAMESPACE}_http_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "endpoint", "http_status"],
)

# Seconds since the process started serving.
UPTIME = Gauge(
    f"{NAMESPACE}_app_uptime_seconds",
    "Seconds elapsed since the application started.",
)

# 1 when the model is loaded and the service can answer predictions, else 0.
MODEL_LOADED = Gauge(
    f"{NAMESPACE}_model_loaded",
    "Whether the model is loaded and ready to serve (1) or not (0).",
)

# Static labels describing which model is being served.
MODEL_INFO = Info(
    f"{NAMESPACE}_model",
    "Metadata about the currently served model.",
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_START_TIME = time.time()


def start_uptime_tracking() -> None:
    """Anchor the uptime gauge to now and keep it current automatically.

    ``set_function`` re-evaluates on every scrape, so the gauge stays accurate
    without a background thread.
    """
    global _START_TIME
    _START_TIME = time.time()
    UPTIME.set_function(lambda: time.time() - _START_TIME)


def uptime_seconds() -> float:
    """Current uptime, also reported by ``/health``."""
    return time.time() - _START_TIME


def set_model_info(description: dict) -> None:
    """Publish model metadata and flag the service as ready."""
    # Info labels must be strings.
    MODEL_INFO.info({k: str(v) for k, v in description.items() if v is not None})
    MODEL_LOADED.set(1)


def set_model_unavailable() -> None:
    """Flag the service as unable to serve predictions."""
    MODEL_LOADED.set(0)


def observe_prediction(latency: float, batch_size: int, labels: list[str]) -> None:
    """Record a successful prediction request."""
    PREDICTION_REQUESTS.labels(status="success").inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_BATCH_SIZE.observe(batch_size)
    for label in labels:
        PREDICTIONS_BY_CLASS.labels(predicted_class=label).inc()


def observe_failure(latency: float, reason: str) -> None:
    """Record a failed prediction request."""
    PREDICTION_REQUESTS.labels(status="failure").inc()
    PREDICTION_FAILURES.labels(reason=reason).inc()
    PREDICTION_LATENCY.observe(latency)


def observe_http(method: str, endpoint: str, status_code: int) -> None:
    """Record any HTTP request handled by the API."""
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, http_status=str(status_code)).inc()


def render() -> tuple[bytes, str]:
    """Return the metrics payload and its content type for ``GET /metrics``."""
    return generate_latest(), CONTENT_TYPE_LATEST
