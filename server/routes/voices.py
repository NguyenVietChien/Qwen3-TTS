# coding=utf-8
"""Voice library endpoints — upload once, reuse by name."""

import os
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/voices", tags=["voices"])

VOICES_DIR = os.environ.get("VOICES_DIR", "/data/voices")


class VoiceInfo(BaseModel):
    name: str
    filename: str
    size_kb: float


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_")


@router.get("", response_model=List[VoiceInfo])
async def list_voices():
    """List all saved voice samples."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    voices = []
    for fname in sorted(os.listdir(VOICES_DIR)):
        if fname.endswith((".wav", ".mp3", ".flac", ".m4a")):
            path = os.path.join(VOICES_DIR, fname)
            name = os.path.splitext(fname)[0]
            voices.append(VoiceInfo(
                name=name,
                filename=fname,
                size_kb=round(os.path.getsize(path) / 1024, 1),
            ))
    return voices


@router.post("", response_model=VoiceInfo)
async def upload_voice(
    name: str = Form(..., description="Voice sample name (used as ID)"),
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/FLAC)"),
):
    """Upload a voice sample and save it by name."""
    os.makedirs(VOICES_DIR, exist_ok=True)

    safe = _safe_name(name)
    if not safe:
        raise HTTPException(400, "Invalid voice name")

    ext = os.path.splitext(audio.filename or "")[-1].lower() or ".wav"
    filename = f"{safe}{ext}"
    out_path = os.path.join(VOICES_DIR, filename)

    content = await audio.read()
    with open(out_path, "wb") as f:
        f.write(content)

    return VoiceInfo(
        name=safe,
        filename=filename,
        size_kb=round(len(content) / 1024, 1),
    )


@router.delete("/{name}")
async def delete_voice(name: str):
    """Delete a saved voice sample."""
    safe = _safe_name(name)
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        path = os.path.join(VOICES_DIR, f"{safe}{ext}")
        if os.path.exists(path):
            os.unlink(path)
            return {"deleted": safe}
    raise HTTPException(404, f"Voice '{name}' not found")


@router.get("/{name}/audio")
async def get_voice_audio(name: str):
    """Download a saved voice sample."""
    safe = _safe_name(name)
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        path = os.path.join(VOICES_DIR, f"{safe}{ext}")
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(404, f"Voice '{name}' not found")
