# coding=utf-8
"""Voice library — upload multiple samples per voice, reuse by name."""

import os
import re
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/voices", tags=["voices"])

VOICES_DIR = os.environ.get("VOICES_DIR", "/data/voices")

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_")


def _voice_dir(name: str) -> str:
    return os.path.join(VOICES_DIR, _safe_name(name))


def _list_audio_files(voice_dir: str) -> List[str]:
    if not os.path.isdir(voice_dir):
        return []
    return sorted(
        f for f in os.listdir(voice_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    )


class VoiceFile(BaseModel):
    filename: str
    size_kb: float


class VoiceInfo(BaseModel):
    name: str
    file_count: int
    files: List[VoiceFile]


@router.get("", response_model=List[VoiceInfo])
async def list_voices():
    """List all saved voices and their sample files."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    voices = []
    for name in sorted(os.listdir(VOICES_DIR)):
        voice_dir = os.path.join(VOICES_DIR, name)
        if not os.path.isdir(voice_dir):
            continue
        files = _list_audio_files(voice_dir)
        voices.append(VoiceInfo(
            name=name,
            file_count=len(files),
            files=[
                VoiceFile(
                    filename=f,
                    size_kb=round(os.path.getsize(os.path.join(voice_dir, f)) / 1024, 1)
                )
                for f in files
            ]
        ))
    return voices


@router.post("", response_model=VoiceInfo)
async def upload_voice(
    name: str = Form(..., description="Voice name (used as ID)"),
    files: List[UploadFile] = File(..., description="One or more audio files (WAV/MP3/FLAC)"),
):
    """
    Upload one or more audio samples for a voice.
    Multiple uploads with the same name ADD to the existing voice.
    """
    safe = _safe_name(name)
    if not safe:
        raise HTTPException(400, "Invalid voice name")

    voice_dir = _voice_dir(safe)
    os.makedirs(voice_dir, exist_ok=True)

    existing = _list_audio_files(voice_dir)
    idx_start = len(existing)

    saved = []
    for i, upload in enumerate(files):
        ext = os.path.splitext(upload.filename or "")[-1].lower() or ".wav"
        filename = f"{safe}_{idx_start + i:03d}{ext}"
        out_path = os.path.join(voice_dir, filename)
        content = await upload.read()
        with open(out_path, "wb") as f:
            f.write(content)
        saved.append(VoiceFile(filename=filename, size_kb=round(len(content) / 1024, 1)))

    all_files = [
        VoiceFile(
            filename=f,
            size_kb=round(os.path.getsize(os.path.join(voice_dir, f)) / 1024, 1)
        )
        for f in _list_audio_files(voice_dir)
    ]

    return VoiceInfo(name=safe, file_count=len(all_files), files=all_files)


@router.delete("/{name}")
async def delete_voice(name: str):
    """Delete a voice and all its samples."""
    import shutil
    voice_dir = _voice_dir(name)
    if not os.path.isdir(voice_dir):
        raise HTTPException(404, f"Voice '{name}' not found")
    shutil.rmtree(voice_dir)
    return {"deleted": name}


@router.delete("/{name}/{filename}")
async def delete_voice_file(name: str, filename: str):
    """Delete a single sample file from a voice."""
    safe = _safe_name(name)
    path = os.path.join(VOICES_DIR, safe, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File '{filename}' not found in voice '{name}'")
    os.unlink(path)
    return {"deleted": filename}


@router.get("/{name}/{filename}")
async def get_voice_file(name: str, filename: str):
    """Download a specific sample file."""
    safe = _safe_name(name)
    path = os.path.join(VOICES_DIR, safe, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)
