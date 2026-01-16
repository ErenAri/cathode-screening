# Tune Cloud Run scaling parameters for the backend service.
# Usage: .\scripts\gcp_scale_tune.ps1 -ProjectId YOUR_PROJECT_ID -ServiceName cathode-backend -Concurrency 4 -MinInstances 1 -MaxInstances 20

param (
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$Region = "us-central1",
    [string]$ServiceName = "cathode-backend",

    [int]$Concurrency = 0,
    [int]$MinInstances = 0,
    [int]$MaxInstances = 0,
    [string]$Memory = "",
    [string]$Cpu = "",
    [int]$TimeoutSeconds = 0
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================================`n"
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not installed. Please install it first."
}

Write-Step "Configuring GCP Project"
gcloud config set project $ProjectId | Out-Null

$UpdateArgs = @(
    "run", "services", "update", $ServiceName,
    "--platform", "managed",
    "--region", $Region
)

if ($Concurrency -gt 0) {
    $UpdateArgs += @("--concurrency", $Concurrency)
}
if ($MinInstances -gt 0) {
    $UpdateArgs += @("--min-instances", $MinInstances)
}
if ($MaxInstances -gt 0) {
    $UpdateArgs += @("--max-instances", $MaxInstances)
}
if (-not [string]::IsNullOrWhiteSpace($Memory)) {
    $UpdateArgs += @("--memory", $Memory)
}
if (-not [string]::IsNullOrWhiteSpace($Cpu)) {
    $UpdateArgs += @("--cpu", $Cpu)
}
if ($TimeoutSeconds -gt 0) {
    $UpdateArgs += @("--timeout", "${TimeoutSeconds}s")
}

Write-Step "Updating Cloud Run service"
gcloud @UpdateArgs

Write-Host "Scaling update complete." -ForegroundColor Green
