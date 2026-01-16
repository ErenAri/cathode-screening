# CathodeScreen Deployment Script for Google Cloud Platform
# Usage: .\deploy_gcp.ps1 -ProjectId YOUR_PROJECT_ID [-ArtifactsGcsUri gs://YOUR_BUCKET/path]

param (
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    
    [string]$Region = "us-central1",

    [string]$BackendServiceName = "cathode-backend",
    [string]$FrontendServiceName = "cathode-frontend",
    [string]$ArtifactsGcsUri = "",

    [switch]$UseSecretManager,
    [string]$ApiKeysSecret = "cathode-api-keys",
    [string]$ApiKeyHashesSecret = "",
    [string]$ManifestKeySecret = "cathode-manifest-hmac-key",
    [string]$ServiceAccount = "",

    [int]$BackendConcurrency = 0,
    [int]$BackendMaxInstances = 0,
    [int]$BackendMinInstances = 0,

    [switch]$AllowUnauthenticatedBackend
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================================`n"
}

function Test-Secret {
    param([string]$SecretName)
    if ([string]::IsNullOrWhiteSpace($SecretName)) {
        return $false
    }
    $null = gcloud secrets describe $SecretName --project $ProjectId --format "value(name)" 2>$null
    return $LASTEXITCODE -eq 0
}

# 1. Check Pre-requisites
Write-Step "Checking Prerequisites"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not installed. Please install it first."
}

Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"
Write-Host "Backend:    $BackendServiceName"
Write-Host "Frontend:   $FrontendServiceName"

# 2. Configure Project
Write-Step "Configuring GCP Project"
gcloud config set project $ProjectId

# 3. Deploy Backend
Write-Step "Building & Deploying Backend (Cloud Run)"
Write-Host "NOTE: If prompted to enable APIs (run.googleapis.com, cloudbuild.googleapis.com), please press 'y' and Enter." -ForegroundColor Yellow

# Build using cloudbuild.yaml to support custom Dockerfile
$BuildArgs = @("builds", "submit", "--config", "backend.cloudbuild.yaml", ".")
if (-not [string]::IsNullOrWhiteSpace($ArtifactsGcsUri)) {
    $BuildArgs += @("--substitutions", "_ARTIFACTS_GCS_URI=$ArtifactsGcsUri")
} elseif (-not (Test-Path "data/artifacts")) {
    Write-Host "Warning: data/artifacts not found locally and -ArtifactsGcsUri not set. Cloud Build will fail." -ForegroundColor Yellow
}

gcloud @BuildArgs

if ($LASTEXITCODE -ne 0) { Write-Error "Backend build failed." }

if ($UseSecretManager) {
    Write-Step "Checking Secret Manager"
    if (-not (Test-Secret $ApiKeysSecret) -and -not (Test-Secret $ApiKeyHashesSecret)) {
        Write-Host "Warning: API key secrets not found. Create them in Secret Manager before deploy." -ForegroundColor Yellow
    }
    if (-not (Test-Secret $ManifestKeySecret)) {
        Write-Host "Warning: Manifest key secret not found. Signature verification will fail." -ForegroundColor Yellow
    }
}

$BackendEnvVars = @(
    "PORT=8080",
    "CATHODE_ENV=production",
    "CATHODE_AUTH_ENABLED=true",
    "CATHODE_REQUIRE_MANIFEST_SIGNATURE=true",
    "CATHODE_TRUST_PROXY=true",
    "CATHODE_TRUST_PROXY_HOPS=1",
    "CATHODE_FORCE_HTTPS=true",
    "CATHODE_SECURITY_HEADERS=true",
    "CATHODE_PROMETHEUS_ENABLED=true",
    "CATHODE_LOG_REQUESTS=true"
)

$BackendArgs = @(
    "run", "deploy", $BackendServiceName,
    "--image", "gcr.io/$ProjectId/$BackendServiceName",
    "--platform", "managed",
    "--region", $Region,
    "--port", "8080",
    "--memory", "2Gi",
    "--set-env-vars", ($BackendEnvVars -join ",")
)

if ($AllowUnauthenticatedBackend) {
    $BackendArgs += "--allow-unauthenticated"
} else {
    $BackendArgs += "--no-allow-unauthenticated"
}

if (-not [string]::IsNullOrWhiteSpace($ServiceAccount)) {
    $BackendArgs += @("--service-account", $ServiceAccount)
}

if ($BackendConcurrency -gt 0) {
    $BackendArgs += @("--concurrency", $BackendConcurrency)
}
if ($BackendMaxInstances -gt 0) {
    $BackendArgs += @("--max-instances", $BackendMaxInstances)
}
if ($BackendMinInstances -gt 0) {
    $BackendArgs += @("--min-instances", $BackendMinInstances)
}

if ($UseSecretManager) {
    $Secrets = @()
    if (-not [string]::IsNullOrWhiteSpace($ApiKeysSecret)) {
        $Secrets += "CATHODE_API_KEYS=$ApiKeysSecret:latest"
    }
    if (-not [string]::IsNullOrWhiteSpace($ApiKeyHashesSecret)) {
        $Secrets += "CATHODE_API_KEY_HASHES=$ApiKeyHashesSecret:latest"
    }
    if (-not [string]::IsNullOrWhiteSpace($ManifestKeySecret)) {
        $Secrets += "CATHODE_MANIFEST_HMAC_KEY=$ManifestKeySecret:latest"
    }
    if ($Secrets.Count -gt 0) {
        $BackendArgs += @("--set-secrets", ($Secrets -join ","))
    }
}

gcloud @BackendArgs

if ($LASTEXITCODE -ne 0) { Write-Error "Backend deployment failed." }

# Get Backend URL
$BackendUrl = (gcloud run services describe $BackendServiceName --platform managed --region $Region --format 'value(status.url)')

if ([string]::IsNullOrWhiteSpace($BackendUrl)) {
    Write-Error "Failed to retrieve Backend URL. Please check the backend deployment."
}

Write-Host "Backend is live at: $BackendUrl" -ForegroundColor Green

# 4. Deploy Frontend
Write-Step "Building & Deploying Frontend (Cloud Run)"
Write-Host "Injecting Backend URL: $BackendUrl"

# Build with backend URL substitution
gcloud builds submit --config frontend.cloudbuild.yaml --substitutions="_BACKEND_URL=$BackendUrl" .

if ($LASTEXITCODE -ne 0) { Write-Error "Frontend build failed." }

$FrontendArgs = @(
    "run", "deploy", $FrontendServiceName,
    "--image", "gcr.io/$ProjectId/$FrontendServiceName",
    "--platform", "managed",
    "--region", $Region,
    "--allow-unauthenticated",
    "--port", "3000",
    "--memory", "1Gi"
)

gcloud @FrontendArgs

if ($LASTEXITCODE -ne 0) { Write-Error "Frontend deployment failed." }

$FrontendUrl = (gcloud run services describe $FrontendServiceName --platform managed --region $Region --format 'value(status.url)')

Write-Step "Deployment Complete! 🚀"
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Green
Write-Host "Backend:  $BackendUrl" -ForegroundColor Green
