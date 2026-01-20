# Quick API Key Fix Script
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OpenAI API Key Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (Test-Path .env) {
    Write-Host "Current .env file found." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "Creating new .env file..." -ForegroundColor Yellow
}

Write-Host "Please enter your NEW OpenAI API key:" -ForegroundColor Green
Write-Host "(Get it from: https://platform.openai.com/api-keys)" -ForegroundColor Gray
Write-Host ""
$newKey = Read-Host "API Key"

if ($newKey -and $newKey.Trim() -ne "") {
    # Remove any spaces
    $newKey = $newKey.Trim()
    
    # Validate format
    if ($newKey -notmatch "^sk-") {
        Write-Host ""
        Write-Host "⚠ Warning: API key should start with 'sk-'" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y") {
            Write-Host "Cancelled." -ForegroundColor Red
            exit
        }
    }
    
    # Write to .env file
    "OPENAI_API_KEY=$newKey" | Out-File -FilePath .env -Encoding utf8 -NoNewline
    
    Write-Host ""
    Write-Host "✓ API key updated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now restarting Docker container..." -ForegroundColor Yellow
    Write-Host ""
    
    # Stop and restart container
    docker-compose down
    docker-compose up -d --build
    
    Write-Host ""
    Write-Host "✓ Docker container restarted!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now test the video upload." -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ No API key provided. Exiting." -ForegroundColor Red
    Write-Host ""
}

