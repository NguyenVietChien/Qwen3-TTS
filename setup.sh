#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "      Qwen3-TTS - SETUP"
echo "      Setting up environment..."
echo "========================================"
echo ""

# Step 1: Check Python
echo "[1/5] Checking Python..."
command -v python3 &>/dev/null || { echo "  ERROR: python3 not found. Install Python 3.10+"; exit 1; }
echo "  $(python3 --version)"

# Step 2: Check Node.js
echo "[2/5] Checking Node.js..."
command -v node &>/dev/null || { echo "  ERROR: node not found. Install Node.js 18+"; exit 1; }
echo "  Node.js $(node --version)"

# Step 3: Create venv
echo "[3/5] Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    echo "  Removing old .venv..."
    rm -rf .venv
fi
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
echo "  .venv created"

# Step 4: Install Python dependencies
echo "[4/5] Installing Python dependencies..."

# 4a. Detect GPU
echo "  [4a] Detecting GPU..."
if command -v nvidia-smi &>/dev/null; then
    echo "  NVIDIA GPU detected! Installing CUDA PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "  No GPU found, installing CPU PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 4b. Install project
echo "  [4b] Installing qwen_tts project..."
pip install -e .

# 4c. Server deps
echo "  [4c] Installing server dependencies..."
pip install -r server/requirements.txt

# 4d. Verify
echo "  [4d] Verifying..."
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
python -c "import qwen_tts; print('  qwen_tts OK')"
python -c "import fastapi; print('  fastapi OK')"

# Step 5: Frontend
echo "[5/5] Installing frontend..."
cd frontend && npm install && cd ..

echo ""
echo "========================================"
echo "     SETUP COMPLETE!"
echo "========================================"
echo ""
echo "  Run ./start.sh to launch."
