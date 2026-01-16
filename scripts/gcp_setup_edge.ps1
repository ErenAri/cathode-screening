# Create an HTTPS Load Balancer with Cloud Armor in front of Cloud Run.
# Usage: .\scripts\gcp_setup_edge.ps1 -ProjectId YOUR_PROJECT_ID -Domain api.example.com -ServiceName cathode-backend

param (
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$Domain,

    [string]$Region = "us-central1",
    [string]$ServiceName = "cathode-backend",

    [string]$NegName = "cathode-backend-neg",
    [string]$BackendName = "cathode-backend-service",
    [string]$UrlMapName = "cathode-url-map",
    [string]$ProxyName = "cathode-https-proxy",
    [string]$AddressName = "cathode-lb-ip",
    [string]$ForwardingRuleName = "cathode-https-forwarding-rule",
    [string]$CertName = "cathode-managed-cert",
    [string]$SecurityPolicyName = "cathode-edge-policy",
    [switch]$SkipIngressUpdate
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================================`n"
}

function Test-Gcloud {
    param([string]$Command)
    $null = Invoke-Expression $Command 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not installed. Please install it first."
}

Write-Step "Configuring GCP Project"
gcloud config set project $ProjectId | Out-Null

if (-not $SkipIngressUpdate) {
    Write-Step "Locking Cloud Run ingress to load balancer only"
    gcloud run services update $ServiceName --region $Region --ingress internal-and-cloud-load-balancing
}

Write-Step "Reserving global IP address"
if (-not (Test-Gcloud "gcloud compute addresses describe $AddressName --global")) {
    gcloud compute addresses create $AddressName --global
}

Write-Step "Creating managed SSL certificate"
if (-not (Test-Gcloud "gcloud compute ssl-certificates describe $CertName --global")) {
    gcloud compute ssl-certificates create $CertName --domains $Domain
}

Write-Step "Creating serverless NEG for Cloud Run"
if (-not (Test-Gcloud "gcloud compute network-endpoint-groups describe $NegName --region $Region")) {
    gcloud compute network-endpoint-groups create $NegName --region $Region --network-endpoint-type=serverless --cloud-run-service $ServiceName
}

Write-Step "Creating backend service"
if (-not (Test-Gcloud "gcloud compute backend-services describe $BackendName --global")) {
    gcloud compute backend-services create $BackendName --global --load-balancing-scheme=EXTERNAL --protocol=HTTP
}

$negSelfLink = gcloud compute network-endpoint-groups describe $NegName --region $Region --format "value(selfLink)"
$backendGroups = gcloud compute backend-services describe $BackendName --global --format "value(backends[].group)"
$backendGroupList = @()
if (-not [string]::IsNullOrWhiteSpace($backendGroups)) {
    $backendGroupList = $backendGroups -split "`n"
}
if ($backendGroupList -notcontains $negSelfLink) {
    gcloud compute backend-services add-backend $BackendName --global --network-endpoint-group $NegName --network-endpoint-group-region $Region
}

Write-Step "Creating URL map"
if (-not (Test-Gcloud "gcloud compute url-maps describe $UrlMapName")) {
    gcloud compute url-maps create $UrlMapName --default-service $BackendName
}

Write-Step "Creating HTTPS proxy"
if (-not (Test-Gcloud "gcloud compute target-https-proxies describe $ProxyName")) {
    gcloud compute target-https-proxies create $ProxyName --url-map $UrlMapName --ssl-certificates $CertName
}

Write-Step "Creating forwarding rule"
if (-not (Test-Gcloud "gcloud compute forwarding-rules describe $ForwardingRuleName --global")) {
    gcloud compute forwarding-rules create $ForwardingRuleName --global --target-https-proxy $ProxyName --ports 443 --address $AddressName
}

Write-Step "Creating Cloud Armor policy"
if (-not (Test-Gcloud "gcloud compute security-policies describe $SecurityPolicyName")) {
    gcloud compute security-policies create $SecurityPolicyName --description "Cathode API edge policy"
}

Write-Step "Attaching Cloud Armor policy"
gcloud compute backend-services update $BackendName --global --security-policy $SecurityPolicyName

$ip = gcloud compute addresses describe $AddressName --global --format "value(address)"
Write-Host "Load balancer IP: $ip" -ForegroundColor Green
Write-Host "Update DNS A record for $Domain to point at $ip. Certificate provisioning can take 15-60 minutes." -ForegroundColor Yellow
