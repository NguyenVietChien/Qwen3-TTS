# coding=utf-8
"""TTS generation routes for the Qwen3-TTS API server."""

import io
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..models import (
    GenerationParams,
    ModelInfoResponse,
    TTSMode,
    TTSRequest,
)

router = APIRouter(prefix="/api/tts", tags=["tts"])

# ---------------------------------------------------------------------------
# The model reference is injected by main.py at startup via `router.state`
# ---------------------------------------------------------------------------


def _get_tts():
    """Retrieve the loaded Qwen3TTSModel from router state."""
    tts = getattr(router, "_tts_model", None)
    if tts is None:
        raise HTTPException(503, detail="Model not loaded yet")
    return tts


def _gen_kwargs(params: Optional[GenerationParams]) -> dict:
    """Convert optional GenerationParams to a kwargs dict (skip None values)."""
    if params is None:
        return {}
    return {k: v for k, v in params.model_dump().items() if v is not None}


def _wav_to_streaming_response(wavs: list[np.ndarray], sr: int) -> StreamingResponse:
    """Convert generated waveforms to a WAV streaming response (first waveform only)."""
    wav = wavs[0]
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Return model metadata: supported languages, speakers, model type."""
    tts = _get_tts()
    return ModelInfoResponse(
        model_type=getattr(tts.model, "tts_model_type", None),
        supported_languages=tts.get_supported_languages(),
        supported_speakers=tts.get_supported_speakers(),
        status="ready",
    )


@router.post("/generate")
async def generate_tts(req: TTSRequest):
    """
    Main TTS generation endpoint.

    - `custom_voice` → uses built-in speaker IDs
    - `voice_design` → uses natural language style instruction
    """
    tts = _get_tts()
    gen_kw = _gen_kwargs(req.generation_params)

    try:
        if req.mode == TTSMode.CUSTOM_VOICE:
            if not req.speaker:
                raise HTTPException(400, "speaker is required for custom_voice mode")
            wavs, sr = tts.generate_custom_voice(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                instruct=req.instruct,
                **gen_kw,
            )

        elif req.mode == TTSMode.VOICE_DESIGN:
            if not req.instruct:
                raise HTTPException(400, "instruct is required for voice_design mode")
            wavs, sr = tts.generate_voice_design(
                text=req.text,
                instruct=req.instruct,
                language=req.language,
                **gen_kw,
            )

        else:
            raise HTTPException(
                400,
                "Use POST /api/tts/voice-clone for voice_clone mode",
            )

    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"Generation failed: {exc}")

    return _wav_to_streaming_response(wavs, sr)


@router.post("/voice-clone")
async def voice_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    ref_audio: UploadFile = File(..., description="Reference audio WAV file"),
    # Generation params as individual form fields (optional)
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
):
    """
    Voice clone endpoint.

    Accepts multipart form with a reference audio file + text parameters.
    Returns generated WAV audio.
    """
    tts = _get_tts()

    # Save uploaded audio to a temp file
    suffix = ".wav"
    content = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    gen_kw = {}
    if top_k is not None:
        gen_kw["top_k"] = top_k
    if top_p is not None:
        gen_kw["top_p"] = top_p
    if temperature is not None:
        gen_kw["temperature"] = temperature
    if max_new_tokens is not None:
        gen_kw["max_new_tokens"] = max_new_tokens

    try:
        wavs, sr = tts.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=tmp_path,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **gen_kw,
        )
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"Voice clone failed: {exc}")
    finally:
        import os
        os.unlink(tmp_path)

    return _wav_to_streaming_response(wavs, sr)
