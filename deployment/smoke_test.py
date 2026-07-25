"""End-to-end smoke test for the serving API.

Runs the whole app in-process with FastAPI's TestClient — no server, no ports,
no Docker — and checks every endpoint plus the error paths.

    python deployment/smoke_test.py

Exits 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app  # noqa: E402
from inference import PROJECT_ROOT, example_record  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}{f' -- {detail}' if detail else ''}")


def main() -> int:
    # TestClient's context manager runs the startup/shutdown lifespan, which is
    # what loads the model.
    with TestClient(app) as client:
        print("\n[1] GET /")
        response = client.get("/")
        body = response.json()
        check("returns 200", response.status_code == 200, str(response.status_code))
        check("reports ready", body.get("status") == "ready", str(body.get("status")))
        check("describes the model", bool(body.get("model")), "model block missing")

        print("\n[2] GET /health")
        response = client.get("/health")
        body = response.json()
        check("returns 200", response.status_code == 200, str(response.status_code))
        check("status is healthy", body.get("status") == "healthy", str(body.get("status")))
        check("model is loaded", body.get("model_loaded") is True)
        check("preprocessor is loaded", body.get("preprocessor_loaded") is True)
        check("reports uptime", isinstance(body.get("uptime_seconds"), (int, float)))

        print("\n[3] POST /predict — raw clinical values")
        response = client.post("/predict", json={"instances": [example_record()]})
        body = response.json()
        check("returns 200", response.status_code == 200, response.text[:200])
        prediction = body["predictions"][0] if response.status_code == 200 else {}
        check("prediction is 0 or 1", prediction.get("prediction") in (0, 1))
        check("returns a label", isinstance(prediction.get("label"), str))
        probability = prediction.get("probability")
        check(
            "probability in [0, 1]",
            isinstance(probability, float) and 0.0 <= probability <= 1.0,
            str(probability),
        )
        check("reports latency", isinstance(body.get("latency_ms"), (int, float)))

        print("\n[4] POST /predict — batch of raw records")
        raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "heart_failure_clinical_records_dataset.csv")
        raw.columns = [c.lower() for c in raw.columns]
        batch = raw.drop(columns=["death_event"]).head(5).to_dict(orient="records")
        response = client.post("/predict", json={"instances": batch})
        body = response.json()
        check("returns 200", response.status_code == 200, response.text[:200])
        check("returns one result per instance", body.get("n_instances") == 5, str(body.get("n_instances")))

        print("\n[5] POST /predict/preprocessed - processed dataset schema")
        processed = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
        rows = processed.drop(columns=["death_event"]).head(5).to_dict(orient="records")
        response = client.post("/predict/preprocessed", json={"instances": rows})
        check("accepts processed rows", response.status_code == 200, response.text[:200])

        print("\n[6] Raw and preprocessed paths agree")
        # The same five patients, sent raw and sent already-scaled, must produce
        # identical predictions — this proves the API applies the preprocessor.
        train_index = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
        del train_index  # only read to confirm the file is present
        raw_first = raw.drop(columns=["death_event"]).head(3).to_dict(orient="records")
        raw_response = client.post("/predict", json={"instances": raw_first}).json()

        import joblib

        preprocessor = joblib.load(PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib")
        scaled = preprocessor.transform(
            pd.DataFrame(raw_first)[list(preprocessor.feature_names_in_)]
        )
        scaled_frame = pd.DataFrame(scaled, columns=preprocessor.get_feature_names_out())
        scaled_response = client.post(
            "/predict/preprocessed", json={"instances": scaled_frame.to_dict(orient="records")}
        ).json()

        raw_labels = [p["prediction"] for p in raw_response["predictions"]]
        scaled_labels = [p["prediction"] for p in scaled_response["predictions"]]
        check(
            "raw input is scaled before inference",
            raw_labels == scaled_labels,
            f"raw={raw_labels} scaled={scaled_labels}",
        )

        print("\n[7] POST /predict — invalid input returns 422")
        bad = example_record() | {"sex": 7}  # 0/1 flag out of range
        response = client.post("/predict", json={"instances": [bad]})
        check("out-of-range flag rejected", response.status_code == 422, str(response.status_code))
        check("error body names the field", "sex" in response.text, response.text[:200])

        missing = {k: v for k, v in example_record().items() if k != "age"}
        response = client.post("/predict", json={"instances": [missing]})
        check("missing field rejected", response.status_code == 422, str(response.status_code))

        response = client.post("/predict", json={"instances": []})
        check("empty batch rejected", response.status_code == 422, str(response.status_code))

        response = client.post("/predict", json={"nope": 1})
        check("malformed body rejected", response.status_code == 422, str(response.status_code))

        print("\n[8] GET /metrics")
        response = client.get("/metrics")
        text = response.text
        check("returns 200", response.status_code == 200)
        check(
            "uses the Prometheus content type",
            "text/plain" in response.headers.get("content-type", ""),
            response.headers.get("content-type", ""),
        )
        for metric in [
            "heart_failure_prediction_requests_total",
            "heart_failure_prediction_failures_total",
            "heart_failure_prediction_latency_seconds_bucket",
            "heart_failure_app_uptime_seconds",
            "heart_failure_model_loaded",
            "heart_failure_predictions_by_class_total",
            "heart_failure_http_requests_total",
            "heart_failure_model_info",
        ]:
            check(f"exposes {metric}", metric in text)

        # Every metric line must parse as valid exposition format.
        from prometheus_client.parser import text_string_to_metric_families

        try:
            families = list(text_string_to_metric_families(text))
            check("parses as Prometheus exposition format", len(families) > 0, f"{len(families)} families")
        except Exception as exc:  # noqa: BLE001
            check("parses as Prometheus exposition format", False, str(exc))

        # The counters must reflect the calls made above.
        success = [
            f for f in text.splitlines()
            if f.startswith('heart_failure_prediction_requests_total{status="success"}')
        ]
        failure = [
            f for f in text.splitlines()
            if f.startswith('heart_failure_prediction_requests_total{status="failure"}')
        ]
        check("counted successful predictions", bool(success) and float(success[0].split()[-1]) >= 4, str(success))
        check("counted failed predictions", bool(failure) and float(failure[0].split()[-1]) >= 4, str(failure))

    print("\n" + "=" * 60)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("  failing checks:")
        for name in FAILED:
            print(f"    - {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
