@echo off
chcp 65001 >nul 2>&1
title QuantAI Strategy Lab

echo ============================================
echo   QuantAI Strategy Lab - Windows Launcher
echo ============================================
echo.

cd /d "%~dp0"

:: 0. Kill old server process on port 9000
echo [0/3] Stopping old server if running...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :9000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
echo.

:: 1. Check Python virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)
echo.

:: 2. Install/update dependencies
echo [2/3] Checking dependencies...
pip install -q -r requirements.txt 2>nul
echo.

:: 3. Start server
echo [3/3] Starting server on http://localhost:9000
echo.
echo ============================================
echo   Dashboard: http://localhost:9000
echo   Press Ctrl+C to stop
echo ============================================
echo.
uvicorn main:app --reload --host 127.0.0.1 --port 9000

pause
