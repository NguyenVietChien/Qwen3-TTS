#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "========================================"
echo "      Qwen3-TTS - START"
echo "========================================"
echo ""

# Check setup
if [ ! -f ".venv/bin/activate" ]; then
    echo "  Running setup first..."
    bash setup.sh
fi
source .venv/bin/activate

# Sanity check
python -c "import qwen_tts; import fastapi" 2>/dev/null || {
    echo "  Dependencies missing, running setup..."
    bash setup.sh
    source .venv/bin/activate
}

cleanup() {
    echo "Stopping..."
    kill $FE_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start frontend
echo "  Starting frontend..."
cd frontend && npm run dev &
FE_PID=$!
cd ..

sleep 2
echo ""
echo "========================================"
echo "  Backend:   http://localhost:8000"
echo "  Frontend:  http://localhost:5173"
echo "========================================"
echo ""

# Open browser
if command -v xdg-open &>/dev/null; then xdg-open http://localhost:5173
elif command -v open &>/dev/null; then open http://localhost:5173; fi

echo "  Press Ctrl+C to stop."
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
