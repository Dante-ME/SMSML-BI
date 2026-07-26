<#
.SYNOPSIS
    Start the full local monitoring stack: FastAPI + Prometheus + Grafana.

.DESCRIPTION
    Everything this script needs is already in the repository:
      deployment/prometheus.yml            scrape + rule configuration
      deployment/alert_rules.yml           Prometheus alert rules
      deployment/grafana/provisioning/     Grafana data source, dashboards, alerts
      deployment/grafana/dashboards/       dashboard JSON

    Only the two binaries live outside it, under $MonitoringHome, because they
    are large third-party downloads and do not belong in version control.

    Each service is started only if its port is free, so re-running the script
    is safe.

.PARAMETER MonitoringHome
    Where prometheus-*/ and grafana-*/ were extracted.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deployment\start_monitoring.ps1
#>

[CmdletBinding()]
param(
    [string]$MonitoringHome = "$env:USERPROFILE\monitoring"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Deployment = Join-Path $RepoRoot "deployment"
$LogDir = Join-Path $MonitoringHome "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Test-PortBusy([int]$Port) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Resolve-One([string]$Pattern, [string]$What) {
    $hit = Get-ChildItem -Path $MonitoringHome -Directory -Filter $Pattern -ErrorAction SilentlyContinue |
           Sort-Object Name -Descending | Select-Object -First 1
    if (-not $hit) { throw "$What not found under $MonitoringHome (looked for '$Pattern')." }
    return $hit.FullName
}

# --------------------------------------------------------------------------- #
# 1. FastAPI
# --------------------------------------------------------------------------- #
if (Test-PortBusy 8000) {
    Write-Host "[skip] something is already listening on :8000"
} else {
    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { $python = "python" }
    Write-Host "[start] FastAPI  -> http://localhost:8000"
    Start-Process -FilePath $python -ArgumentList "`"$Deployment\app.py`"" `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput "$LogDir\api.out.log" -RedirectStandardError "$LogDir\api.err.log" `
        -WindowStyle Hidden | Out-Null
}

# --------------------------------------------------------------------------- #
# 2. Prometheus
# --------------------------------------------------------------------------- #
if (Test-PortBusy 9090) {
    Write-Host "[skip] something is already listening on :9090"
} else {
    $promHome = Resolve-One "prometheus-*" "Prometheus"
    Write-Host "[start] Prometheus -> http://localhost:9090"
    # Paths are quoted individually: the repository path contains spaces, and
    # Start-Process joins -ArgumentList entries without quoting them.
    $promArgs = @(
        "--config.file=`"$Deployment\prometheus.yml`""
        "--storage.tsdb.path=`"$MonitoringHome\data\prometheus`""
        "--storage.tsdb.retention.time=15d"
        "--web.listen-address=127.0.0.1:9090"
        "--web.enable-lifecycle"
    )
    Start-Process -FilePath "$promHome\prometheus.exe" -ArgumentList $promArgs `
        -WorkingDirectory $promHome `
        -RedirectStandardOutput "$LogDir\prometheus.out.log" -RedirectStandardError "$LogDir\prometheus.err.log" `
        -WindowStyle Hidden | Out-Null
}

# --------------------------------------------------------------------------- #
# 3. Grafana
# --------------------------------------------------------------------------- #
if (Test-PortBusy 3000) {
    Write-Host "[skip] something is already listening on :3000"
} else {
    $grafanaHome = Resolve-One "grafana-*" "Grafana"
    Write-Host "[start] Grafana    -> http://localhost:3000"
    # Provisioning is read from the repository, so the data source, the
    # dashboard and the alert rules are rebuilt on every start.
    $env:GF_PATHS_PROVISIONING = "$Deployment\grafana\provisioning"
    $env:DASHBOARDS_PATH       = "$Deployment\grafana\dashboards"   # used inside dashboards.yml
    $env:GF_PATHS_DATA         = "$MonitoringHome\data\grafana"
    $env:GF_PATHS_LOGS         = "$LogDir\grafana"
    $env:GF_SERVER_HTTP_PORT   = "3000"
    New-Item -ItemType Directory -Force -Path $env:GF_PATHS_DATA, $env:GF_PATHS_LOGS | Out-Null
    Start-Process -FilePath "$grafanaHome\bin\grafana.exe" -ArgumentList @("server", "--homepath", "`"$grafanaHome`"") `
        -WorkingDirectory $grafanaHome `
        -RedirectStandardOutput "$LogDir\grafana.out.log" -RedirectStandardError "$LogDir\grafana.err.log" `
        -WindowStyle Hidden | Out-Null
}

# --------------------------------------------------------------------------- #
# 4. Wait until each one answers
# --------------------------------------------------------------------------- #
$checks = @(
    @{ Name = "FastAPI";    Url = "http://localhost:8000/health" }
    @{ Name = "Prometheus"; Url = "http://localhost:9090/-/ready" }
    @{ Name = "Grafana";    Url = "http://localhost:3000/api/health" }   # slowest: ~30s on a cold start
)

Write-Host ""
foreach ($c in $checks) {
    $up = $false
    foreach ($attempt in 1..40) {
        try { Invoke-WebRequest $c.Url -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break }
        catch { Start-Sleep -Seconds 2 }
    }
    if ($up) { Write-Host ("  OK    {0}" -f $c.Name) }
    else     { Write-Warning ("{0} did not answer {1}; check {2}" -f $c.Name, $c.Url, $LogDir) }
}

Write-Host ""
Write-Host "  API         http://localhost:8000/docs"
Write-Host "  Metrics     http://localhost:8000/metrics"
Write-Host "  Prometheus  http://localhost:9090/targets"
Write-Host "  Alerts      http://localhost:9090/alerts"
Write-Host "  Grafana     http://localhost:3000/d/heart-failure-api  (admin / admin)"
