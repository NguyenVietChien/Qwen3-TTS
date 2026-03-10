# coding=utf-8
"""
Qwen3-TTS FastAPI Server
========================

Wraps the Qwen3TTSModel inference APIs into a REST server for team usage.

Usage (inside venv):
    python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_PATH  — HuggingFace repo ID or local path (default: Qwen/Qwen3-TTS)
    DEVICE      — PyTorch device string (default: cuda:0)
    DTYPE       — float16 | bfloat16 | float32 (default: bfloat16)
    AUTO_LOAD   — set to "1" to load model at startup (default: "0" = lazy load)
"""

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from qwen_tts import Qwen3TTSModel

from .routes.tts import router as tts_router

logger = logging.getLogger("qwen3_tts_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-TTS")
DTYPE_STR = os.environ.get("DTYPE", "bfloat16")
AUTO_LOAD = os.environ.get("AUTO_LOAD", "0") == "1"

# Auto-detect device
_env_device = os.environ.get("DEVICE", "")
if _env_device:
    DEVICE = _env_device
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
    logger.warning("CUDA not available — running on CPU (will be slow)")
    # CPU doesn't support bfloat16 well on all platforms
    if DTYPE_STR == "bfloat16":
        DTYPE_STR = "float32"

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Available model variants for the frontend to choose from
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen3-TTS", "name": "Qwen3-TTS (Default)", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-0.6B-CustomVoice", "name": "Qwen3-TTS 0.6B CustomVoice", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-1.7B-CustomVoice", "name": "Qwen3-TTS 1.7B CustomVoice", "type": "custom_voice"},
    {"id": "Qwen/Qwen3-TTS-0.6B-VoiceDesign", "name": "Qwen3-TTS 0.6B VoiceDesign", "type": "voice_design"},
    {"id": "Qwen/Qwen3-TTS-1.7B-VoiceDesign", "name": "Qwen3-TTS 1.7B VoiceDesign", "type": "voice_design"},
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "name": "Qwen3-TTS 12Hz 0.6B Base (Voice Clone)", "type": "base"},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "name": "Qwen3-TTS 12Hz 1.7B Base (Voice Clone)", "type": "base"},
]

# Global state for model management
_current_model_path: str = MODEL_PATH


def _load_model(model_path: str) -> Qwen3TTSModel:
    """Load a Qwen3-TTS model."""
    global _current_model_path
    dtype = _DTYPE_MAP.get(DTYPE_STR, torch.bfloat16)
    logger.info("Loading Qwen3-TTS model from %s on %s (%s)…", model_path, DEVICE, DTYPE_STR)
    load_kwargs = {"dtype": dtype}
    if DEVICE == "cpu":
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = DEVICE
    tts = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
    logger.info("Model loaded successfully ✓  type=%s", getattr(tts.model, "tts_model_type", "?"))
    _current_model_path = model_path
    return tts


# ---------------------------------------------------------------------------
# Lifespan — optionally load model on startup (AUTO_LOAD=1)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if AUTO_LOAD:
        try:
            tts = _load_model(MODEL_PATH)
            tts_router._tts_model = tts  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("Auto-load failed (model may not be downloaded yet): %s", e)
            logger.info("Server started WITHOUT model. Use POST /api/models/load to load one.")
    else:
        logger.info("Server started in lazy-load mode. Use POST /api/models/load to load a model.")

    yield  # app is running

    # Cleanup
    old = getattr(tts_router, "_tts_model", None)
    if old is not None:
        logger.info("Shutting down — releasing model…")
        del old
        tts_router._tts_model = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qwen3-TTS Server",
    description="REST API for Qwen3 Text-to-Speech — custom voice, voice design, and voice cloning.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow React dev server + any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for team distribution
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts_router)


@app.get("/health")
async def health():
    """Simple health check."""
    model_loaded = getattr(tts_router, "_tts_model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded, "current_model": _current_model_path}


@app.get("/api/models")
async def list_models():
    """List available model variants and the currently loaded one."""
    model_loaded = getattr(tts_router, "_tts_model", None) is not None
    return {
        "models": AVAILABLE_MODELS,
        "current": _current_model_path if model_loaded else None,
        "model_loaded": model_loaded,
    }


@app.post("/api/models/load")
async def load_model(body: dict):
    """
    Switch to a different model at runtime.

    Body: { "model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base" }
    """
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(400, "model_path is required")

    # Release old model
    old = getattr(tts_router, "_tts_model", None)
    if old is not None:
        del old
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        tts = _load_model(model_path)
        tts_router._tts_model = tts  # type: ignore[attr-defined]
        return {
            "status": "loaded",
            "model_path": model_path,
            "model_type": getattr(tts.model, "tts_model_type", None),
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("Model load failed:\n%s", tb)
        raise HTTPException(500, f"Failed to load model '{model_path}': {type(e).__name__}: {e}")
