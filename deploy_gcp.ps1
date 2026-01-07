# CathodeScreen Deployment Script for Google Cloud Platform
# Usage: .\deploy_gcp.ps1 -ProjectId YOUR_PROJECT_ID

param (
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================================`n"
}

# 1. Check Pre-requisites
Write-Step "Checking Prerequisites"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not installed. Please install it first."
}

Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"

# 2. Configure Project
Write-Step "Configuring GCP Project"
gcloud config set project $ProjectId

# 3. Deploy Backend
Write-Step "Building & Deploying Backend (Cloud Run)"
Write-Host "NOTE: If prompted to enable APIs (run.googleapis.com, cloudbuild.googleapis.com), please press 'y' and Enter." -ForegroundColor Yellow

# Build using cloudbuild.yaml to support custom Dockerfile
gcloud builds submit --config backend.cloudbuild.yaml .

if ($LASTEXITCODE -ne 0) { Write-Error "Backend build failed." }

gcloud run deploy cathode-backend `
    --image "gcr.io/$ProjectId/cathode-backend" `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --port 8080 `
    --memory 2Gi

if ($LASTEXITCODE -ne 0) { Write-Error "Backend deployment failed." }

# Get Backend URL
$BackendUrl = (gcloud run services describe cathode-backend --platform managed --region $Region --format 'value(status.url)')

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

gcloud run deploy cathode-frontend `
    --image "gcr.io/$ProjectId/cathode-frontend" `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --port 3000 `
    --memory 1Gi

if ($LASTEXITCODE -ne 0) { Write-Error "Frontend deployment failed." }

$FrontendUrl = (gcloud run services describe cathode-frontend --platform managed --region $Region --format 'value(status.url)')

Write-Step "Deployment Complete! 🚀"
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Green
Write-Host "Backend:  $BackendUrl" -ForegroundColor Green
