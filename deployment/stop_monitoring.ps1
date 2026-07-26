<#
.SYNOPSIS
    Stop the local monitoring stack started by start_monitoring.ps1.

.DESCRIPTION
    Stops Prometheus and Grafana by process name, and the FastAPI service by
    whichever process owns port 8000 (the process is plain python.exe, so
    killing it by name would take down unrelated interpreters).

    Collected data is kept: the TSDB and the Grafana database live under
    $MonitoringHome\data and survive a restart.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deployment\stop_monitoring.ps1
#>

[CmdletBinding()]
param(
    [switch]$KeepApi
)

foreach ($name in "prometheus", "grafana") {
    $procs = Get-Process $name -ErrorAction SilentlyContinue
    if ($procs) { $procs | Stop-Process -Force; Write-Host "[stop] $name" }
    else        { Write-Host "[skip] $name is not running" }
}

if (-not $KeepApi) {
    $owner = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique
    if ($owner) {
        $owner | ForEach-Object { Stop-Process -Id $_ -Force }
        Write-Host "[stop] FastAPI (pid $owner)"
    } else {
        Write-Host "[skip] nothing is listening on :8000"
    }
}
