@echo off
chcp 65001 >nul
title Qwen3-TTS Setup
color 0E

echo.
echo  ========================================
echo        Qwen3-TTS - SETUP
echo        Setting up environment...
echo  ========================================
echo.

cd /d "%~dp0"

:: ─────────────────────────────────────
::  Step 1: Check Python
:: ─────────────────────────────────────
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  Found Python %%v

:: ─────────────────────────────────────
::  Step 2: Check Node.js
:: ─────────────────────────────────────
echo [2/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found!
    echo  Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
for /f %%v in ('node --version 2^>^&1') do echo  Found Node.js %%v

:: ─────────────────────────────────────
::  Step 3: Create fresh venv
:: ─────────────────────────────────────
echo [3/5] Setting up Python virtual environment...
if exist ".venv" (
    echo  Removing old .venv...
    rmdir /s /q .venv
)
echo  Creating .venv...
python -m venv .venv
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo  .venv created and activated

:: Upgrade pip first
python -m pip install --upgrade pip -q

:: ─────────────────────────────────────
::  Step 4: Install Python dependencies
::  ORDER MATTERS:
::    1) CUDA torch first (if GPU)
::    2) Project deps (skips torch since already installed)
::    3) Server deps (fastapi etc)
:: ─────────────────────────────────────
echo [4/5] Installing Python dependencies...
echo.

:: 4a. Detect GPU and install PyTorch
echo  [4a] Detecting GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo  NVIDIA GPU detected!
    echo  Installing PyTorch with CUDA 12.4 support...
    echo  This will take a few minutes...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo  [WARN] CUDA torch install failed, trying CPU fallback...
        pip install torch torchaudio
    )
) else (
    echo  No NVIDIA GPU detected, installing CPU PyTorch...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)
echo.

:: 4b. Install project (qwen_tts) — torch is already installed so pip skips it
echo  [4b] Installing qwen_tts project...
pip install -e .
echo.

:: 4c. Install server dependencies (fastapi, uvicorn, etc)
echo  [4c] Installing server dependencies...
pip install -r server\requirements.txt
echo.

:: 4d. Verify installation
echo  [4d] Verifying installation...
python -c "import torch; cuda = torch.cuda.is_available(); print(f'  PyTorch {torch.__version__}  CUDA: {cuda}'); exec('print(f\"  GPU: {torch.cuda.get_device_name(0)}\")' if cuda else '')"
python -c "import qwen_tts; print('  qwen_tts OK')"
python -c "import fastapi; print('  fastapi OK')"
echo.

:: ─────────────────────────────────────
::  Step 5: Install frontend dependencies
:: ─────────────────────────────────────
echo [5/5] Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo.

:: ─────────────────────────────────────
::  Done!
:: ─────────────────────────────────────
echo  ========================================
echo       SETUP COMPLETE!
echo  ========================================
echo.
echo  Run start.bat to launch the application.
echo.
pause
