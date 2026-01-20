@echo off
echo AI Video Editor - Starting Server...
echo.
echo Make sure you have set your OPENAI_API_KEY environment variable!
echo.
echo Setting up Python environment...
pip install -r requirements.txt
echo.
echo Starting FastAPI server with Uvicorn...
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause

