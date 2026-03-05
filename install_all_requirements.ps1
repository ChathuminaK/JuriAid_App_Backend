<#
.SYNOPSIS
    Install requirements for all JuriAid backend microservices
.DESCRIPTION
    Creates a .venv in each service folder (if not present) and installs its requirements.txt
#>

$ErrorActionPreference = "Stop"
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

$services = @(
    @{
        Name = "Auth Service"
        Path = Join-Path $ROOT_DIR "auth_service"
        Requirements = "requirements.txt"
    },
    @{
        Name = "PastCase Retrieval"
        Path = Join-Path $ROOT_DIR "past_case_retrieval"
        Requirements = "requirements.txt"
    },
    @{
        Name = "LawStatKG"
        Path = Join-Path $ROOT_DIR "LawStatKG\backend"
        Requirements = "requirements.txt"
    },
    @{
        Name = "QuestionGen"
        Path = Join-Path $ROOT_DIR "questionGen"
        Requirements = "requirements.txt"
    },
    @{
        Name = "Orchestrator"
        Path = Join-Path $ROOT_DIR "orchestratorc"
        Requirements = "requirements.txt"
    }
)

Write-Host "Installing requirements for all JuriAid services..." -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray

foreach ($service in $services) {
    Write-Host "`n[$($service.Name)]" -ForegroundColor Yellow
    
    if (-not (Test-Path $service.Path)) {
        Write-Host "  Directory not found: $($service.Path)" -ForegroundColor Red
        continue
    }

    $reqFile = Join-Path $service.Path $service.Requirements
    if (-not (Test-Path $reqFile)) {
        Write-Host "  requirements.txt not found in $($service.Path)" -ForegroundColor Red
        continue
    }

    $venvPath = Join-Path $service.Path ".venv"
    $pipExe   = Join-Path $venvPath "Scripts\pip.exe"

    # Create venv if it doesn't exist
    if (-not (Test-Path $venvPath)) {
        Write-Host "  Creating virtual environment..." -ForegroundColor Cyan
        python -m venv $venvPath
    } else {
        Write-Host "  Virtual environment already exists." -ForegroundColor Gray
    }

    # Install requirements
    Write-Host "  Installing from requirements.txt..." -ForegroundColor Cyan
    & $pipExe install --upgrade pip -q
    & $pipExe install -r $reqFile

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Done!" -ForegroundColor Green
    } else {
        Write-Host "  Installation failed for $($service.Name)!" -ForegroundColor Red
    }
}

Write-Host "`n" + ("=" * 60) -ForegroundColor Gray
Write-Host "All services processed." -ForegroundColor Green