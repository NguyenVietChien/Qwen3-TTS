# coding=utf-8
"""
Qwen3-TTS FastAPI Server
========================

REST API that enqueues TTS jobs onto a Celery/Redis queue.
The actual model runs in a separate Celery worker process (with GPU).

Usage (inside venv):
    python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_PATH   — HuggingFace repo ID or local path (default: Qwen/Qwen3-TTS)
    REDIS_URL    — Redis connection string (default: redis://localhost:6379/0)
    OUTPUT_DIR   — Shared directory for generated audio (default: /data/tts_outputs)
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.presets import router as presets_router
from .routes.tasks import router as tasks_router
from .routes.tts import router as tts_router
from .routes.voices import router as voices_router

logger = logging.getLogger("qwen3_tts_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-TTS")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "https://speech.flashcutai.com,http://speech.flashcutai.com",
    ).split(",")
    if origin.strip()
]

AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen3-TTS", "name": "Qwen3-TTS (Default)", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-0.6B-CustomVoice", "name": "Qwen3-TTS 0.6B CustomVoice", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-1.7B-CustomVoice", "name": "Qwen3-TTS 1.7B CustomVoice", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-0.6B-VoiceDesign", "name": "Qwen3-TTS 0.6B VoiceDesign", "type": "voice_design"},
    {"id": "Qwen/Qwen3-TTS-1.7B-VoiceDesign", "name": "Qwen3-TTS 1.7B VoiceDesign", "type": "voice_design"},
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "name": "Qwen3-TTS 12Hz 0.6B Base (Voice Clone)", "type": "base"},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "name": "Qwen3-TTS 12Hz 1.7B Base (Voice Clone)", "type": "base"},
]

app = FastAPI(
    title="Qwen3-TTS Service",
    description=(
        "Async REST API for Qwen3 Text-to-Speech.\n\n"
        "Submit a job → receive a UUID → poll for status → download audio."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts_router)
app.include_router(tasks_router)
app.include_router(voices_router)
app.include_router(presets_router)


@app.get("/health")
async def health():
    """Simple health check for load balancers / Docker healthcheck."""
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    """List available model variants."""
    return {"models": AVAILABLE_MODELS, "current": MODEL_PATH}
