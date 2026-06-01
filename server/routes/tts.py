# coding=utf-8
"""TTS generation routes — enqueue Celery tasks, return UUID immediately."""

import os
import re
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..models import TaskResponse, TaskStatus, TTSMode, TTSRequest
from ..worker_tasks import task_generate, task_voice_clone

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/tts_outputs")
VOICES_DIR = os.environ.get("VOICES_DIR", "/data/voices")

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")

router = APIRouter(prefix="/api/tts", tags=["tts"])


def _find_voice_paths(voice_name: str) -> List[str]:
    """Resolve a saved voice name to a list of audio file paths (supports multi-sample voices)."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name).strip("_")
    voice_dir = os.path.join(VOICES_DIR, safe)
    if os.path.isdir(voice_dir):
        paths = sorted(
            os.path.join(voice_dir, f)
            for f in os.listdir(voice_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS
        )
        if paths:
            return paths
    raise HTTPException(404, f"Voice '{voice_name}' not found. Upload it via POST /api/voices first.")


@router.post("/generate", response_model=TaskResponse)
async def generate_tts(req: TTSRequest):
    """
    Submit a TTS generation task. Returns task_id immediately.

    Poll GET /api/tasks/{task_id} for status, then GET /api/tasks/{task_id}/audio for the result.
    """
    if req.mode == TTSMode.VOICE_CLONE:
        raise HTTPException(400, "Use POST /api/tts/voice-clone for voice_clone mode")
    if req.mode == TTSMode.CUSTOM_VOICE and not req.speaker:
        raise HTTPException(400, "speaker is required for custom_voice mode")
    if req.mode == TTSMode.VOICE_DESIGN and not req.instruct:
        raise HTTPException(400, "instruct is required for voice_design mode")

    gen_params = req.generation_params.model_dump() if req.generation_params else {}

    task = task_generate.delay(
        text=req.text,
        mode=req.mode.value,
        language=req.language,
        gen_params=gen_params,
        speaker=req.speaker,
        instruct=req.instruct,
    )
    return TaskResponse(task_id=task.id, status=TaskStatus.PENDING)


@router.post("/voice-clone", response_model=TaskResponse)
async def voice_clone(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form("Auto", description="Language name, e.g. 'Chinese', 'English', 'Auto'"),
    ref_text: Optional[str] = Form(None, description="Transcript of reference audio (for ICL mode)"),
    x_vector_only: bool = Form(False, description="Use x-vector only mode — no ref_text needed"),
    # Voice source: saved voice name OR one/multiple uploaded files
    voice_name: Optional[str] = Form(None, description="Name of a saved voice (see GET /api/voices)"),
    ref_audio: Optional[List[UploadFile]] = File(None, description="One or more reference audio files"),
    # --- Generation params (all optional) ---
    do_sample: Optional[bool] = Form(None),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    subtalker_dosample: Optional[bool] = Form(None),
    subtalker_top_k: Optional[int] = Form(None),
    subtalker_top_p: Optional[float] = Form(None),
    subtalker_temperature: Optional[float] = Form(None),
):
    """
    Submit a voice-clone TTS task. Returns task_id immediately.

    Provide EITHER:
    - voice_name: use all samples saved under that voice name
    - ref_audio: upload one or more audio files directly

    Poll GET /api/tasks/{task_id} for status.
    """
    delete_after = False

    if voice_name:
        ref_audio_paths = _find_voice_paths(voice_name)
    elif ref_audio:
        upload_dir = os.path.join(OUTPUT_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        ref_audio_paths = []
        for upload in ref_audio:
            ext = os.path.splitext(upload.filename or "")[-1].lower() or ".wav"
            path = os.path.join(upload_dir, f"{uuid.uuid4()}{ext}")
            content = await upload.read()
            with open(path, "wb") as f:
                f.write(content)
            ref_audio_paths.append(path)
        delete_after = True
    else:
        raise HTTPException(400, "Provide either voice_name or ref_audio")

    gen_params = {
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "subtalker_dosample": subtalker_dosample,
        "subtalker_top_k": subtalker_top_k,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_temperature": subtalker_temperature,
    }

    task = task_voice_clone.delay(
        text=text,
        language=language,
        ref_audio_paths=ref_audio_paths,
        gen_params=gen_params,
        ref_text=ref_text,
        x_vector_only=x_vector_only,
        delete_after=delete_after,
    )
    return TaskResponse(task_id=task.id, status=TaskStatus.PENDING)
