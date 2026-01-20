@echo off
echo Loading environment variables from .env file...
if exist .env (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" (
            if not "%%a"=="#" (
                set "%%a=%%b"
            )
        )
    )
)
echo.
echo Starting Docker Compose...
docker-compose up --build
pause

