<#
.SYNOPSIS
    Start all JuriAid backend microservices
.DESCRIPTION
    Launches 5 microservices in separate PowerShell windows:
    - Auth Service (Port 8001)
    - PastCase Retrieval (Port 8002)
    - LawStatKG (Port 8003)
    - QuestionGen (Port 8004)
    - Orchestrator (Port 8000)
#>

$ErrorActionPreference = "Stop"

Write-Host "Starting JuriAid Backend Services..." -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray

# Get the script directory
$ROOT_DIR = $PSScriptRoot

# Service configurations
$services = @(
    @{
        Name = "Auth Service"
        Port = 8001
        Path = Join-Path $ROOT_DIR "auth_service"
        Command = "uvicorn app:app --reload --port 8001"
        Color = "Green"
    },
    @{
        Name = "PastCase Retrieval"
        Port = 8002
        Path = Join-Path $ROOT_DIR "past_case_retrieval"
        Command = "uvicorn app:app --reload --port 8002"
        Color = "Yellow"
    },
    @{
        Name = "LawStatKG"
        Port = 8003
        Path = Join-Path $ROOT_DIR "LawStatKG\backend"
        Command = "uvicorn app.api:app --reload --port 8003"
        Color = "Magenta"
    },
    @{
        Name = "QuestionGen"
        Port = 8004
        Path = Join-Path $ROOT_DIR "questionGen"
        Command = "uvicorn api:app --reload --port 8004"
        Color = "Blue"
    },
    @{
        Name = "Orchestrator"
        Port = 8000
        Path = Join-Path $ROOT_DIR "orchestratorc"
        Command = "uvicorn app:app --reload --port 8000"
        Color = "Cyan"
    }
)

# Function to check if port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        return $connection.TcpTestSucceeded
    }
    catch {
        return $false
    }
}

# Kill processes on required ports
Write-Host "`nChecking for running services..." -ForegroundColor Yellow
foreach ($service in $services) {
    if (Test-Port -Port $service.Port) {
        Write-Host "Port $($service.Port) is in use. Attempting to free it..." -ForegroundColor Yellow
        try {
            $processes = Get-NetTCPConnection -LocalPort $service.Port -ErrorAction SilentlyContinue | 
                         Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($proc in $processes) {
                try {
                    Stop-Process -Id $proc -Force -ErrorAction SilentlyContinue
                    Write-Host "  Killed process $proc on port $($service.Port)" -ForegroundColor Green
                }
                catch {
                    Write-Host "  Could not kill process $proc" -ForegroundColor Red
                }
            }
            Start-Sleep -Seconds 1
        }
        catch {
            Write-Host "  Error freeing port $($service.Port)" -ForegroundColor Red
        }
    }
}

# Start each service in a new PowerShell window
Write-Host "`nLaunching services..." -ForegroundColor Cyan
foreach ($service in $services) {
    if (-not (Test-Path $service.Path)) {
        Write-Host "Directory not found: $($service.Path)" -ForegroundColor Red
        continue
    }

    $activateScript = Join-Path $service.Path ".venv\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        $commandBlock = "Set-Location '$($service.Path)'; .\.venv\Scripts\Activate.ps1; Write-Host 'Starting $($service.Name) on port $($service.Port)...' -ForegroundColor $($service.Color); $($service.Command)"
    }
    else {
        $commandBlock = "Set-Location '$($service.Path)'; Write-Host 'Starting $($service.Name) on port $($service.Port)...' -ForegroundColor $($service.Color); $($service.Command)"
    }

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $commandBlock
    Write-Host "Started $($service.Name) on port $($service.Port)" -ForegroundColor $service.Color
    Start-Sleep -Seconds 2
}

# Wait and check if services are running
Write-Host "`nWaiting for services to initialize (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`nHealth Check:" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray

foreach ($service in $services) {
    if (Test-Port -Port $service.Port) {
        Write-Host "$($service.Name) (Port $($service.Port)): RUNNING" -ForegroundColor Green
    }
    else {
        Write-Host "$($service.Name) (Port $($service.Port)): NOT RESPONDING" -ForegroundColor Red
    }
}

# Display API endpoints
Write-Host "`nAPI Endpoints:" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray
Write-Host "Auth Service:        http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "PastCase Retrieval:  http://127.0.0.1:8002" -ForegroundColor Yellow
Write-Host "LawStatKG:           http://127.0.0.1:8003" -ForegroundColor Magenta
Write-Host "QuestionGen:         http://127.0.0.1:8004" -ForegroundColor Blue
Write-Host "Orchestrator (MAIN): http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Docs:" -ForegroundColor Cyan
Write-Host "  Orchestrator: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  Auth:         http://127.0.0.1:8001/docs" -ForegroundColor White
Write-Host "  PastCase:     http://127.0.0.1:8002/docs" -ForegroundColor White
Write-Host "  LawStatKG:    http://127.0.0.1:8003/docs" -ForegroundColor White
Write-Host "  QuestionGen:  http://127.0.0.1:8004/docs" -ForegroundColor White

Write-Host "`nAll services launched!" -ForegroundColor Green
Write-Host "Press Ctrl+C in each window to stop individual services." -ForegroundColor Yellow