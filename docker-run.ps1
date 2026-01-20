# PowerShell script to run Docker Compose with .env file
# This script loads .env file and sets environment variables

if (Test-Path .env) {
    Write-Host "Loading .env file..." -ForegroundColor Green
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "Set $key" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "Warning: .env file not found. Using system environment variables." -ForegroundColor Yellow
}

Write-Host "`nStarting Docker Compose..." -ForegroundColor Green
docker-compose up --build

