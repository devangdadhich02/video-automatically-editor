# Script to update OpenAI API key in .env file
Write-Host "OpenAI API Key Updater" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host ""
Write-Host "Please enter your new OpenAI API key:" -ForegroundColor Yellow
$newApiKey = Read-Host "API Key"

if ($newApiKey) {
    $envContent = "OPENAI_API_KEY=$newApiKey"
    $envContent | Out-File -FilePath .env -Encoding utf8 -NoNewline
    Write-Host ""
    Write-Host "✓ API key updated successfully in .env file!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now restart Docker container:" -ForegroundColor Yellow
    Write-Host "  docker-compose down" -ForegroundColor Cyan
    Write-Host "  docker-compose up --build" -ForegroundColor Cyan
} else {
    Write-Host "No API key provided. Exiting." -ForegroundColor Red
}

