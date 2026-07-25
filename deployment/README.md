# Deployment and Monitoring

Local serving of the promoted heart failure classifier with **FastAPI**, plus
monitoring with **Prometheus** and **Grafana**.

No Docker is required — everything runs as a local process.

## Prerequisites

The API loads two artifacts produced by the earlier stages:

| Artifact | Produced by |
|---|---|
| `training/models/best_model.joblib` | `python training/modelling_tuning.py` |
| `data/processed/preprocessor.joblib` | `python experiment/automate.py` |

Both are excluded by `.gitignore`, so after a fresh clone regenerate them:

```bash
python experiment/automate.py
python training/modelling.py
python training/modelling_tuning.py
```

Then install the serving dependencies:

```bash
pip install -r deployment/requirements.txt
```

> **Why the preprocessor matters.** The model was trained on standard-scaled
> features. Sending `platelets: 265000` straight to it would produce nonsense,
> so `/predict` runs the fitted preprocessor over the raw values first. If the
> preprocessor is missing the API starts but reports itself unhealthy rather
> than serving wrong numbers.

---

## Files

| File | Role |
|---|---|
| `app.py` | FastAPI application: routes, Pydantic schemas, metrics middleware. |
| `inference.py` | Model + preprocessor loading and the prediction logic. |
| `prometheus_exporter.py` | All Prometheus metric definitions and helpers. |
| `prometheus.yml` | Prometheus scrape configuration for the API. |
| `alert_rules.yml` | Alert rules loaded by `prometheus.yml`. |
| `requirements.txt` | Pinned dependencies for this stage. |
| `smoke_test.py` | End-to-end check of every endpoint, run in-process. |

---

## 1. Start the API

```bash
# from the project root
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# or simply
python deployment/app.py

# during development, with auto-reload
uvicorn deployment.app:app --reload --port 8000
```

On a healthy start you will see:

```
Loaded RandomForestClassifier from .../training/models/best_model.joblib
Preprocessor loaded from .../data/processed/preprocessor.joblib
```

Interactive API docs: <http://localhost:8000/docs>

### Verify it works

```bash
python deployment/smoke_test.py
```

This runs the app in-process (no server needed) and checks all endpoints, the
validation errors and the metrics output — 35 checks, exit code 0 when healthy.

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service description, model metadata, endpoint map. |
| `GET` | `/health` | Readiness probe. `200` when ready, `503` when the model is missing. |
| `POST` | `/predict` | Predict from **raw clinical readings**. |
| `POST` | `/predict/preprocessed` | Predict from rows **already scaled** by `automate.py`. |
| `GET` | `/metrics` | Prometheus exposition endpoint. |
| `GET` | `/docs` | Swagger UI. |

### Why two predict routes

The two accept genuinely different value ranges, so they cannot share one
schema. `/predict` enforces medical ranges (`0 <= age <= 120`, flags strictly
`0` or `1`) — those bounds are the useful part of validation, but scaled
features are centred on zero and routinely negative, so they would fail every
time. `/predict/preprocessed` therefore takes unbounded floats and is there for
replaying `data/processed/test.csv` straight through the model. Real clients
should use `/predict`.

---

## 3. Calling `/predict`

### Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
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
        "time": 4.0
      }
    ]
  }'
```

PowerShell:

```powershell
$body = @{ instances = @(@{
    age = 75.0; anaemia = 0; creatinine_phosphokinase = 582.0; diabetes = 0
    ejection_fraction = 20.0; high_blood_pressure = 1; platelets = 265000.0
    serum_creatinine = 1.9; serum_sodium = 130.0; sex = 1; smoking = 0; time = 4.0
}) } | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

### Response

```json
{
  "model_type": "RandomForestClassifier",
  "preprocessed_input": false,
  "n_instances": 1,
  "predictions": [
    {
      "prediction": 1,
      "label": "death_event",
      "probability": 0.9519806132362693,
      "probabilities": { "0": 0.04801938676373066, "1": 0.9519806132362693 }
    }
  ],
  "latency_ms": 73.822
}
```

`instances` accepts up to 1000 records per request, so batching works the same way.

### Error responses

| Status | When |
|---|---|
| `422` | Validation failed — missing field, wrong type, value out of range, empty batch. |
| `500` | The model is loaded but inference failed. |
| `503` | The model is not loaded (see `/health` for the reason). |

Example — a `0/1` flag set to `7`:

```json
{
  "detail": "Request validation failed.",
  "errors": [
    { "field": "body.instances.0.sex", "message": "Input should be less than or equal to 1" }
  ]
}
```

---

## 4. Metrics

`GET /metrics` returns the standard Prometheus text format.

| Metric | Type | Meaning |
|---|---|---|
| `heart_failure_prediction_requests_total{status}` | Counter | Prediction requests, split `success` / `failure`. |
| `heart_failure_prediction_failures_total{reason}` | Counter | Failures by cause: `validation_error`, `inference_error`, `model_not_loaded`. |
| `heart_failure_prediction_latency_seconds` | Histogram | Prediction latency, with `_bucket` / `_sum` / `_count`. |
| `heart_failure_prediction_batch_size` | Histogram | Records submitted per request. |
| `heart_failure_predictions_by_class_total{predicted_class}` | Counter | What the model predicts — a cheap drift signal. |
| `heart_failure_http_requests_total{method,endpoint,http_status}` | Counter | All HTTP traffic. |
| `heart_failure_app_uptime_seconds` | Gauge | Seconds since startup. |
| `heart_failure_model_loaded` | Gauge | `1` ready, `0` not. |
| `heart_failure_model_info` | Info | Model type, path, test ROC-AUC, training run id. |

---

## 5. Start Prometheus

Download it from <https://prometheus.io/download/> and unpack it. Then, with the
API already running:

```bash
cd deployment
prometheus --config.file=prometheus.yml
```

On Windows, from the folder where you unpacked it:

```powershell
.\prometheus.exe --config.file="D:\path\to\Submission\deployment\prometheus.yml"
```

Prometheus starts on <http://localhost:9090>.

**Check the scrape is working:** open **Status → Targets**. `heart-failure-api`
must be **UP**. Then go to **Graph** and run:

```promql
heart_failure_app_uptime_seconds
```

If the target is DOWN, confirm the API answers `curl http://localhost:8000/metrics`
and that nothing else occupies port 8000.

Validate the config files before starting (ships with Prometheus):

```bash
promtool check config prometheus.yml
promtool check rules alert_rules.yml
```

---

## 6. Connect Grafana

Download from <https://grafana.com/grafana/download> and start it. Grafana
listens on <http://localhost:3000> — the default login is `admin` / `admin`.

### Add Prometheus as a data source

1. **Connections → Data sources → Add new data source**.
2. Choose **Prometheus**.
3. Set **Prometheus server URL** to `http://localhost:9090`.
4. Click **Save & test**. You should see *Successfully queried the Prometheus API*.

---

## 7. Create a dashboard

**Dashboards → New → New dashboard → Add visualization**, pick the Prometheus
data source, then paste one query per panel.

| Panel | Type | Query |
|---|---|---|
| Total predictions | Stat | `sum(heart_failure_prediction_requests_total)` |
| Request rate | Time series | `sum(rate(heart_failure_prediction_requests_total[1m])) by (status)` |
| p95 latency | Time series | `histogram_quantile(0.95, sum(rate(heart_failure_prediction_latency_seconds_bucket[5m])) by (le))` |
| Average latency | Stat | `rate(heart_failure_prediction_latency_seconds_sum[5m]) / rate(heart_failure_prediction_latency_seconds_count[5m])` |
| Failures by reason | Time series | `sum(rate(heart_failure_prediction_failures_total[5m])) by (reason)` |
| Failure ratio | Gauge | `sum(rate(heart_failure_prediction_requests_total{status="failure"}[5m])) / clamp_min(sum(rate(heart_failure_prediction_requests_total[5m])), 0.001)` |
| Uptime | Stat | `heart_failure_app_uptime_seconds` (unit: seconds) |
| Model ready | Stat | `heart_failure_model_loaded` |
| Predicted class mix | Pie / Time series | `sum(heart_failure_predictions_by_class_total) by (predicted_class)` |
| Traffic by endpoint | Time series | `sum(rate(heart_failure_http_requests_total[1m])) by (endpoint, http_status)` |

Set the dashboard refresh to **5s** (top right) to match the scrape interval,
then **Save dashboard**.

To generate traffic while you watch the panels:

```bash
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"instances":[{"age":75,"anaemia":0,"creatinine_phosphokinase":582,"diabetes":0,"ejection_fraction":20,"high_blood_pressure":1,"platelets":265000,"serum_creatinine":1.9,"serum_sodium":130,"sex":1,"smoking":0,"time":4}]}' > /dev/null
done
```

---

## 8. Configure an alert rule

Six rules already ship in `alert_rules.yml`, loaded automatically through the
`rule_files` entry in `prometheus.yml`:

| Alert | Fires when | Severity |
|---|---|---|
| `APIDown` | The target has been unscrapeable for 30s. | critical |
| `ModelNotLoaded` | The API is up but has no model for 1m. | critical |
| `HighPredictionFailureRate` | Over 10% of predictions fail for 2m. | warning |
| `PredictionErrors` | Any inference error in the last 5m. | warning |
| `HighPredictionLatency` | p95 latency above 500ms for 2m. | warning |
| `ServiceRecentlyRestarted` | Uptime is under 60s. | info |

View them at <http://localhost:9090/alerts>. States go
**Inactive → Pending** (the `for:` window) **→ Firing**.

### Add your own rule

Append to the `rules:` list in `alert_rules.yml`:

```yaml
      - alert: LowPredictionTraffic
        expr: sum(rate(heart_failure_prediction_requests_total[5m])) < 0.01
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Almost no prediction traffic"
          description: "Fewer than 0.01 requests/sec over the last 5 minutes."
```

Then check and reload — no restart needed:

```bash
promtool check rules alert_rules.yml
curl -X POST http://localhost:9090/-/reload   # needs --web.enable-lifecycle
```

### Test that alerting works

Stop the API and wait ~30 seconds. `APIDown` moves to **Pending** and then
**Firing** on <http://localhost:9090/alerts>. Restart the API and it clears.

Routing alerts to email or Slack needs
[Alertmanager](https://prometheus.io/docs/alerting/alertmanager/); uncomment the
`alerting:` block at the bottom of `prometheus.yml` once it is running. Rules
evaluate and display in the Prometheus UI without it, which is enough for a
local setup.

### Grafana-native alerts

Alternatively, define alerts in Grafana: open a panel → **Alert → New alert rule**,
set the query and threshold, choose an evaluation interval, and pick a contact
point. Prometheus rules are used here because they live in version control
alongside the code.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health` returns 503 | Model or preprocessor missing. Run `python experiment/automate.py` then `python training/modelling_tuning.py` and restart. |
| Predictions look implausible | Raw values were posted to `/predict/preprocessed`. Use `/predict` for real readings. |
| `422` on every request | The body must be `{"instances": [ ... ]}`, not a bare record object. |
| Prometheus target DOWN | The API is not running, or the port in `prometheus.yml` does not match. |
| Grafana panels empty | Check the data source URL is `http://localhost:9090` and the dashboard time range covers the traffic. |
| Unpickling warnings on load | The installed scikit-learn differs from the training one. Install `deployment/requirements.txt`, which pins the same 1.7.2. |
