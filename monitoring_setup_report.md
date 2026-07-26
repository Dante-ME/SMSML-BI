# Monitoring Setup Report

Local observability stack for the Heart Failure Prediction API: **FastAPI → Prometheus → Grafana**.

| | |
|---|---|
| **Date** | 2026-07-26 |
| **Host OS** | Microsoft Windows 11 Pro (10.0.26200), AMD64 |
| **Result** | Stack running, all targets `UP`, dashboard and alerts provisioned from the repository |
| **Administrator privileges** | **Not required** — see [Why no admin was needed](#why-no-administrator-privileges-were-needed) |

---

## 1. Installed versions

| Component | Version | Source |
|---|---|---|
| Prometheus | **3.13.1** (`windows-amd64`) | [prometheus-3.13.1.windows-amd64.zip](https://github.com/prometheus/prometheus/releases/download/v3.13.1/prometheus-3.13.1.windows-amd64.zip) |
| promtool | 3.13.1 | bundled with Prometheus |
| Grafana OSS | **13.1.1** (`windows-amd64`) | [grafana-13.1.1.windows-amd64.zip](https://dl.grafana.com/oss/release/grafana-13.1.1.windows-amd64.zip) |
| Python | 3.10.5 | existing `.venv` in the repository |
| FastAPI / uvicorn / prometheus-client | 0.140.0 / 0.51.0 / 0.26.0 | already installed — nothing had to be added |

Both versions are the latest stable releases (`prerelease: false`, `draft: false`), resolved
at install time from the GitHub releases API rather than hardcoded.

## 2. Installation locations

Nothing was installed into the repository. The two binaries are large third-party
artifacts and live outside version control:

```
C:\Users\asus\monitoring\
├── prometheus-3.13.1.windows-amd64\    prometheus.exe, promtool.exe
├── grafana-13.1.1\                     bin\grafana.exe, conf\, public\
├── data\
│   ├── prometheus\                     TSDB (15-day retention)
│   └── grafana\                        grafana.db (SQLite: users, dashboards, alert state)
└── logs\
    ├── prometheus.out.log / .err.log
    ├── grafana.out.log / .err.log
    ├── api.out.log / .err.log
    └── grafana\grafana.log
```

The install root is configurable — `start_monitoring.ps1 -MonitoringHome <path>` defaults to
`$env:USERPROFILE\monitoring`.

## 3. URLs and credentials

| Service | URL | Credentials |
|---|---|---|
| FastAPI (OpenAPI docs) | http://localhost:8000/docs | none |
| Metrics endpoint | http://localhost:8000/metrics | none |
| Health probe | http://localhost:8000/health | none |
| Prometheus | http://localhost:9090 | none (bound to `127.0.0.1` only) |
| Prometheus targets | http://localhost:9090/targets | |
| Prometheus alerts | http://localhost:9090/alerts | |
| Grafana | http://localhost:3000 | **admin / admin** (Grafana default) |
| The dashboard | http://localhost:3000/d/heart-failure-api | |
| Grafana alert rules | http://localhost:3000/alerting/list | |

> Grafana prompts for a new password on the first UI login. That prompt can be skipped for a
> local install; change it if this host is ever reachable from outside.

## 4. Running the stack

```powershell
# start everything (idempotent — skips anything already listening)
powershell -ExecutionPolicy Bypass -File deployment\start_monitoring.ps1

# stop everything (collected data is kept)
powershell -ExecutionPolicy Bypass -File deployment\stop_monitoring.ps1
```

`start_monitoring.ps1` starts the FastAPI service, Prometheus and Grafana, then polls each
health endpoint until it answers. Grafana re-reads the provisioning files in the repository on
every start, so the data source, the dashboard and the Grafana alert rules are rebuilt from
version control rather than clicked together by hand.

## 5. Configuration

### Reused unchanged

| File | Role |
|---|---|
| `deployment/prometheus.yml` | scrape config — **not modified** |
| `deployment/alert_rules.yml` | 6 Prometheus alert rules — **not modified** |
| `deployment/prometheus_exporter.py` | metric definitions — **not modified** |
| `deployment/app.py` | FastAPI service — **not modified** |

Both configuration files were validated with `promtool check config` and passed on the first
attempt — **no fixes were required**. Two things worth recording because they are easy to get
wrong and this project got them right:

- `rule_files: alert_rules.yml` is relative. Prometheus resolves rule paths relative to the
  **config file's directory**, not the working directory, so passing an absolute
  `--config.file` from anywhere works — the "start from `deployment/`" note in the file header
  is one valid way to run it, not a requirement.
- `scrape_interval: 5s` is below the default `scrape_timeout` of 10s. Prometheus clamps the
  timeout to the interval when the timeout is not set explicitly, so this is safe as written.
  The Grafana data source sets `timeInterval: 5s` to match, which is what keeps
  `$__rate_interval` from producing gaps in the graphs.

### Added (new files, all monitoring-specific)

| File | Role |
|---|---|
| `deployment/start_monitoring.ps1` | one-command start for API + Prometheus + Grafana |
| `deployment/stop_monitoring.ps1` | matching stop |
| `deployment/alert_rules_test.yml` | `promtool` unit tests for the existing alert rules |
| `deployment/grafana/provisioning/datasources/prometheus.yml` | Prometheus as the default data source (`uid: prometheus-hf`) |
| `deployment/grafana/provisioning/dashboards/dashboards.yml` | dashboard provider pointing at the folder below |
| `deployment/grafana/provisioning/alerting/heart_failure_alerts.yml` | 2 Grafana-managed alert rules |
| `deployment/grafana/dashboards/heart_failure_api.json` | the dashboard (13 panels) |
| `deployment/grafana/provisioning/plugins/.gitkeep` | placeholder; Grafana logs an ERROR for a missing provisioning subfolder |

No existing project file was modified.

### Runtime flags

Prometheus:

```
--config.file="…\deployment\prometheus.yml"
--storage.tsdb.path="C:\Users\asus\monitoring\data\prometheus"
--storage.tsdb.retention.time=15d
--web.listen-address=127.0.0.1:9090     # loopback only
--web.enable-lifecycle                  # POST /-/reload without a restart
```

Grafana (environment, set by the launcher):

```
GF_PATHS_PROVISIONING = …\deployment\grafana\provisioning
DASHBOARDS_PATH       = …\deployment\grafana\dashboards   # interpolated inside dashboards.yml
GF_PATHS_DATA         = C:\Users\asus\monitoring\data\grafana
GF_PATHS_LOGS         = C:\Users\asus\monitoring\logs\grafana
GF_SERVER_HTTP_PORT   = 3000
```

`dashboards.yml` refers to `$DASHBOARDS_PATH` instead of an absolute path so the file stays
portable — Grafana interpolates environment variables in provisioning files at load time.

## 6. The dashboard

**Heart Failure API — Serving Overview** (`uid: heart-failure-api`, folder *Heart Failure API*,
5s refresh, 30m window). Panels are grouped by the question they answer:

| Row | Panels |
|---|---|
| Status | API scrape status · Model loaded · Uptime · Predictions served · Failure ratio (5m) · p95 latency (5m) |
| Traffic & latency | Prediction request rate by outcome · Prediction latency percentiles (p50/p95/p99) |
| Behaviour | HTTP request rate by endpoint · Predicted class mix · Prediction failures by reason |
| Detail | HTTP responses by status code · Request batch size percentiles |

Two panels are worth calling out for MLOps purposes:

- **Predicted class mix** — the ratio of `survived` to `death_event` predictions over time is a
  cheap, free drift signal. The input distribution moving usually shows up here before anything
  else does.
- **Failure ratio (5m)** and **p95 latency (5m)** use the same expressions and the same
  thresholds as the `HighPredictionFailureRate` and `HighPredictionLatency` alert rules, so the
  dashboard and the alerts can never disagree about what "bad" means.

Green/red are used only for **state** (up/down, loaded/not loaded, success/failure) and never as
series colours; everything else uses Grafana's classic categorical palette, which is validated
for both light and dark themes.

## 7. Alerting

Two independent layers, deliberately not copies of each other:

**Prometheus-evaluated** (`deployment/alert_rules.yml`, visible at http://localhost:9090/alerts):
`APIDown`, `ModelNotLoaded`, `HighPredictionFailureRate`, `PredictionErrors`,
`HighPredictionLatency`, `ServiceRecentlyRestarted`. All six load and evaluate. Alertmanager is
not running, which is fine — rules still evaluate and show their state in the Prometheus UI.

**Grafana-managed** (`deployment/grafana/provisioning/alerting/heart_failure_alerts.yml`,
visible at http://localhost:3000/alerting/list): `Heart Failure API unreachable` (critical) and
`Prediction p95 latency above 500ms` (warning). These exist because Grafana alerts can be routed
to a contact point — Slack, email, webhook — without deploying an Alertmanager. Only the two
signals someone would genuinely want a notification for are duplicated.

`deployment/alert_rules_test.yml` unit-tests the Prometheus rules offline against synthetic
series:

```powershell
cd deployment
C:\Users\asus\monitoring\prometheus-3.13.1.windows-amd64\promtool.exe test rules alert_rules_test.yml
```

The most valuable case in that file is the first one: a **healthy** service must produce zero
alerts. An alert that fires during normal operation gets muted, and a muted alert is not an
alert.

## 8. Validation results

Every check below was executed against the running stack, not assumed.

### Prometheus targets are UP

```
$ curl http://localhost:9090/api/v1/targets
heart-failure-api    up    http://localhost:8000/metrics
prometheus           up    http://localhost:9090/metrics
```

No scrape errors, and no configuration fixes were needed to get there.

### Metrics are being collected

- 19 `heart_failure_*` metric families stored in the TSDB, 854 active series in the head block.
- `heart_failure_model_loaded = 1`, so the API is not just reachable but able to serve.
- Load was generated against the live HTTP service (not `TestClient`) to give the panels real
  data: **401 successful predictions** with batch sizes from 1 to 25, plus **45 deliberate
  malformed requests** so the failure and 4xx panels are exercised rather than permanently
  empty.

### Grafana can query Prometheus

The data source resolves and returns data through Grafana's own proxy — the exact path a panel
takes at render time:

```
GET /api/datasources/proxy/uid/prometheus-hf/api/v1/query?query=…
```

### Dashboards display live data

All **16 queries across the 13 panels** were replayed through that proxy. Every one returned a
non-empty result — **0 panels with no data**:

| Panel | Series | Sample value |
|---|---|---|
| API scrape status | 1 | `1` (UP) |
| Model loaded | 1 | `1` (READY) |
| Uptime | 1 | `613.6` s |
| Predictions served | 1 | `46` |
| Failure ratio (5m) | 1 | `0.075` |
| p95 latency (5m) | 1 | `0.249` s |
| Prediction request rate by outcome | 2 | `0.818` req/s |
| Prediction latency percentiles | 1 + 1 + 1 | p50 `0.171` / p95 `0.249` / p99 `0.442` s |
| HTTP request rate by endpoint | 3 | `0.182` req/s |
| Predicted class mix | 2 | `1.486` |
| Prediction failures by reason | 1 | `3.99` |
| HTTP responses by status code | 2 | `1.203` req/s |
| Request batch size percentiles | 1 + 1 | p50 `2.2` / p95 `17.1` instances |

### Alerts are working

Three independent levels of evidence:

**1. All rules load and evaluate.** Six Prometheus rules (`inactive`, correct for a healthy
service) and two Grafana rules (`health=ok`, meaning the rule and its data source resolve
without error).

**2. Offline unit tests pass.** `promtool test rules alert_rules_test.yml` → `SUCCESS`,
including the assertion that a healthy service fires **nothing**.

**3. A live outage was staged.** The FastAPI process was killed and the stack was observed
end to end:

```
--- outage ---
prometheus target : down
                    dial tcp [::1]:8000: connectex: No connection could be made…
APIDown           : firing   severity=critical  env=local  service=heart-failure-api
                    "Prometheus has failed to scrape localhost:8000 for over 30s."
Grafana rule      : "Heart Failure API unreachable" = firing

--- after restarting the API with start_monitoring.ps1 ---
prometheus target : up
prometheus alerts : none active
Grafana rule      : "Heart Failure API unreachable" = inactive
```

Both layers detected the same outage independently, the annotation template rendered the
instance label correctly, and both cleared on their own once scraping resumed. The restart was
done with `start_monitoring.ps1`, so the launcher's recovery path is verified too.

A second alert confirmed itself without being staged: the 45 deliberate malformed requests put
the real failure ratio at **10.37%**, which pushed `HighPredictionFailureRate` into `pending` —
the >10% threshold behaving exactly as written, on real traffic.

## 9. Commands executed

Detection and pre-flight:

```powershell
(Get-CimInstance Win32_OperatingSystem).Caption          # Windows 11 Pro
$env:PROCESSOR_ARCHITECTURE                              # AMD64
.venv\Scripts\python.exe -c "import fastapi, uvicorn, prometheus_client, sklearn, pandas, joblib"
```

Start the API and confirm the metrics endpoint:

```powershell
.venv\Scripts\python.exe deployment\app.py               # background
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest  http://localhost:8000/metrics
```

Install Prometheus:

```powershell
Invoke-RestMethod https://api.github.com/repos/prometheus/prometheus/releases/latest
(New-Object Net.WebClient).DownloadFile($promUrl, "$dl\prometheus.zip")
[System.IO.Compression.ZipFile]::ExtractToDirectory("$dl\prometheus.zip", "C:\Users\asus\monitoring")
promtool.exe check config deployment\prometheus.yml
Start-Process prometheus.exe -ArgumentList $promArgs -WindowStyle Hidden
```

Install Grafana:

```powershell
Invoke-RestMethod https://api.github.com/repos/grafana/grafana/releases/latest
(New-Object Net.WebClient).DownloadFile($grafanaUrl, "$dl\grafana.zip")
[System.IO.Compression.ZipFile]::ExtractToDirectory("$dl\grafana.zip", "C:\Users\asus\monitoring")
Start-Process bin\grafana.exe -ArgumentList server, --homepath, "$grafanaHome" -WindowStyle Hidden
```

Validation:

```powershell
Invoke-RestMethod http://localhost:9090/api/v1/targets
Invoke-RestMethod http://localhost:9090/api/v1/rules
Invoke-RestMethod http://localhost:3000/api/datasources                  -Headers $basicAuth
Invoke-RestMethod http://localhost:3000/api/search?type=dash-db          -Headers $basicAuth
Invoke-RestMethod http://localhost:3000/api/v1/provisioning/alert-rules  -Headers $basicAuth
Invoke-RestMethod "http://localhost:3000/api/datasources/proxy/uid/prometheus-hf/api/v1/query?query=…"
promtool.exe test rules alert_rules_test.yml
```

### Problem hit and fixed

Prometheus refused to start with `Error parsing command line arguments: unexpected Machine`.
The repository path contains a space (`…\Sistem Machine Learning\…`) and PowerShell's
`Start-Process -ArgumentList` joins array elements with spaces **without quoting them**, so
`--config.file=…\Sistem Machine Learning\…` arrived as three separate arguments. Fixed by
quoting each path inside the argument string:

```powershell
"--config.file=`"$Deployment\prometheus.yml`""
```

`start_monitoring.ps1` does this for every path it passes, with a comment explaining why, so the
stack survives being cloned into any directory with a space in its name.

## Why no administrator privileges were needed

Both tools were installed from their official portable ZIPs into the user profile and are run as
ordinary foreground processes. Nothing was written to `Program Files`, no Windows service was
registered, no ports below 1024 were bound, and neither Chocolatey nor winget (both of which
would have triggered a UAC prompt for a machine-wide install) was used.

The trade-off: the stack does **not** survive a reboot. See the manual steps below if you want
it to.

## 10. Remaining manual steps

Nothing is required for the stack to work. These are optional:

1. **Change the Grafana password.** `admin / admin` is fine for loopback-only local use; change
   it at http://localhost:3000/profile/password if this host is ever exposed.
2. **Run at startup.** The services are plain processes and stop when the machine reboots.
   To make them persistent, either register them as Windows services with
   [NSSM](https://nssm.cc/) (**requires administrator**) or add a Task Scheduler task that runs
   `deployment\start_monitoring.ps1` at logon (does not require administrator).
3. **Route alerts somewhere.** Both alert layers currently only display state in their own UI.
   For real notifications, add a contact point under
   http://localhost:3000/alerting/notifications, or uncomment the `alerting:` block at the
   bottom of `deployment/prometheus.yml` and run an Alertmanager on port 9093.
4. **Add the rule tests to CI.** `.github/workflows/` already exists. A step running
   `promtool test rules deployment/alert_rules_test.yml` would catch a broken alert rule at push
   time. Not added here, to avoid touching existing CI configuration.
5. **Ignore the monitoring data directory** if you ever move `C:\Users\asus\monitoring` inside
   the repository — the TSDB grows continuously and must not be committed.
