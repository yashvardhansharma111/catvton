@echo off
REM Windows batch script to start CatVTON FastAPI server
REM Make sure you're in the CatVTON directory

echo Starting CatVTON FastAPI Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo WARNING: Could not check CUDA availability
)

echo.
echo Server will start on http://0.0.0.0:8000
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python app_fastapi.py --host 0.0.0.0 --port 8000

pause

