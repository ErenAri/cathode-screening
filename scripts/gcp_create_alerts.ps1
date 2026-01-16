# Create or update GCP Monitoring alert policies for the Cloud Run backend.
# Usage: .\scripts\gcp_create_alerts.ps1 -ProjectId YOUR_PROJECT_ID -ServiceName cathode-backend

param (
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$ServiceName = "cathode-backend",
    [string]$ConfigDir = "configs\\gcp",
    [switch]$UpdateExisting
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

$PolicyFiles = @(
    "alert_policy_error_rate.json",
    "alert_policy_latency.json"
)

foreach ($policyFile in $PolicyFiles) {
    $policyPath = Join-Path $ConfigDir $policyFile
    if (-not (Test-Path $policyPath)) {
        Write-Error "Missing policy file: $policyPath"
    }

    $raw = Get-Content $policyPath -Raw
    $rendered = $raw.Replace("__SERVICE_NAME__", $ServiceName)
    $policy = $rendered | ConvertFrom-Json
    $displayName = $policy.displayName

    $tmpFile = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $tmpFile -Value $rendered -Encoding UTF8

    Write-Step "Applying policy: $displayName"
    $existingPolicies = gcloud monitoring policies list --project $ProjectId --format json | ConvertFrom-Json
    $existingMatch = $existingPolicies | Where-Object { $_.displayName -eq $displayName } | Select-Object -First 1
    $existingName = $existingMatch.name
    if (-not [string]::IsNullOrWhiteSpace($existingName)) {
        if ($UpdateExisting) {
            gcloud monitoring policies update $existingName --project $ProjectId --policy-from-file $tmpFile
        } else {
            Write-Host "Policy already exists. Use -UpdateExisting to update." -ForegroundColor Yellow
        }
    } else {
        gcloud monitoring policies create --project $ProjectId --policy-from-file $tmpFile
    }

    Remove-Item $tmpFile -Force
}

Write-Host "Alert policy setup complete." -ForegroundColor Green
