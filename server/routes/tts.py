# coding=utf-8
"""TTS generation routes — enqueue Celery tasks, return UUID immediately."""

import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..models import TaskResponse, TaskStatus, TTSMode, TTSRequest
from ..worker_tasks import task_generate, task_voice_clone

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/tts_outputs")

router = APIRouter(prefix="/api/tts", tags=["tts"])


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
    ref_text: Optional[str] = Form(None, description="Transcript of the reference audio (improves accuracy)"),
    x_vector_only: bool = Form(False, description="Use x-vector only mode (no in-context learning)"),
    ref_audio: UploadFile = File(..., description="Reference WAV audio file for voice cloning"),
    # --- Generation params (all optional) ---
    do_sample: Optional[bool] = Form(None),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    subtalker_dosample: Optional[bool] = Form(None, description="Sub-talker do_sample (12Hz tokenizer v2)"),
    subtalker_top_k: Optional[int] = Form(None),
    subtalker_top_p: Optional[float] = Form(None),
    subtalker_temperature: Optional[float] = Form(None),
):
    """
    Submit a voice-clone TTS task. Returns task_id immediately.

    Accepts multipart/form-data. Poll GET /api/tasks/{task_id} for status.
    """
    upload_dir = os.path.join(OUTPUT_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    upload_path = os.path.join(upload_dir, f"{uuid.uuid4()}.wav")
    content = await ref_audio.read()
    with open(upload_path, "wb") as f:
        f.write(content)

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
        ref_audio_path=upload_path,
        gen_params=gen_params,
        ref_text=ref_text,
        x_vector_only=x_vector_only,
    )
    return TaskResponse(task_id=task.id, status=TaskStatus.PENDING)
