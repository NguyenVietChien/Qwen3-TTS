@echo off
chcp 65001 >nul
title Qwen3-TTS Server
color 0B

echo.
echo  ========================================
echo       Qwen3-TTS - START
echo  ========================================
echo.

cd /d "%~dp0"

:: Check if setup was done
if not exist ".venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found.
    echo  Running setup first...
    echo.
    call setup.bat
    if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat

:: Quick sanity check
python -c "import qwen_tts; import fastapi" >nul 2>&1
if errorlevel 1 (
    echo  [!] Dependencies missing. Running setup...
    call setup.bat
    if errorlevel 1 exit /b 1
    call .venv\Scripts\activate.bat
)

:: Start frontend in background
echo  Starting frontend...
start "Qwen3-TTS Frontend" /min cmd /c "cd /d "%~dp0frontend" && call npm run dev"

:: Wait a moment then open browser
echo  Starting backend...
echo.
echo  ========================================
echo   Backend:   http://localhost:8000
echo   Frontend:  http://localhost:5173
echo  ========================================
echo.

ping 127.0.0.1 -n 3 >nul
start http://localhost:5173

echo  Press Ctrl+C to stop the server.
echo.
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
