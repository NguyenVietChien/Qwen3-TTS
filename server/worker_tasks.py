# coding=utf-8
"""Celery task definitions — run exclusively inside the worker process."""

import logging
import os

import soundfile as sf
import torch

from .celery_app import celery_app

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/tts_outputs")
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-TTS")
DTYPE_STR = os.environ.get("DTYPE", "bfloat16")

_env_device = os.environ.get("DEVICE", "")
if _env_device:
    DEVICE = _env_device
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
    if DTYPE_STR == "bfloat16":
        DTYPE_STR = "float32"

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_tts_model = None


def _get_tts():
    global _tts_model
    if _tts_model is None:
        from qwen_tts import Qwen3TTSModel
        dtype = _DTYPE_MAP.get(DTYPE_STR, torch.bfloat16)
        logger.info("Loading model %s on %s (%s)…", MODEL_PATH, DEVICE, DTYPE_STR)
        _tts_model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, dtype=dtype, device_map=DEVICE
        )
        logger.info("Model loaded ✓  type=%s", getattr(_tts_model.model, "tts_model_type", "?"))
    return _tts_model


def _save_wav(task_id: str, wavs, sr: int) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    sf.write(out_path, wavs[0], sr, format="WAV", subtype="PCM_16")
    return out_path


def _clean_gen_params(gen_params: dict) -> dict:
    return {k: v for k, v in (gen_params or {}).items() if v is not None}


@celery_app.task(bind=True, name="tts.generate")
def task_generate(self, text: str, mode: str, language: str, gen_params: dict,
                  speaker: str = None, instruct: str = None):
    """Generate TTS audio (custom_voice or voice_design mode)."""
    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    tts = _get_tts()
    gen_kw = _clean_gen_params(gen_params)

    self.update_state(state="PROCESSING", meta={"step": "generating"})

    if mode == "custom_voice":
        wavs, sr = tts.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            **gen_kw,
        )
    elif mode == "voice_design":
        wavs, sr = tts.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
            **gen_kw,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_path = _save_wav(self.request.id, wavs, sr)
    return {"audio_path": out_path, "sample_rate": sr}


@celery_app.task(bind=True, name="tts.voice_clone")
def task_voice_clone(self, text: str, language: str, ref_audio_path: str,
                     gen_params: dict, ref_text: str = None,
                     x_vector_only: bool = False):
    """Generate TTS audio via voice cloning."""
    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    tts = _get_tts()
    gen_kw = _clean_gen_params(gen_params)

    self.update_state(state="PROCESSING", meta={"step": "generating"})

    try:
        wavs, sr = tts.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **gen_kw,
        )
    finally:
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)

    out_path = _save_wav(self.request.id, wavs, sr)
    return {"audio_path": out_path, "sample_rate": sr}
