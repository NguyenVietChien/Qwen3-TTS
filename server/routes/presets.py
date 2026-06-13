# coding=utf-8
"""Preset library — save/reuse TTS generation configs (mode, speaker, params)."""

import json
import os
import re
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import require_api_key
from ..models import GenerationParams, TTSMode

router = APIRouter(prefix="/api/presets", tags=["presets"], dependencies=[Depends(require_api_key)])

PRESETS_DIR = os.environ.get("PRESETS_DIR", "/data/presets")

# Guards read-modify-write sequences against concurrent requests (e.g. multiple n8n calls).
_presets_lock = threading.Lock()


def _atomic_write(path: str, content: str) -> None:
    tmp_path = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_")


def _preset_path(name: str) -> str:
    return os.path.join(PRESETS_DIR, f"{_safe_name(name)}.json")


class Preset(BaseModel):
    """A reusable TTS generation config."""
    name: str = Field(..., min_length=1, description="Preset name (used as ID)")
    mode: TTSMode = TTSMode.CUSTOM_VOICE
    language: str = "Auto"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    voice_name: Optional[str] = Field(None, description="Voice library name (voice_clone mode)")
    generation_params: Optional[GenerationParams] = None


@router.get("", response_model=list[Preset])
async def list_presets():
    """List all saved presets."""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    presets = []
    with _presets_lock:
        for fname in sorted(os.listdir(PRESETS_DIR)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(PRESETS_DIR, fname), encoding="utf-8") as f:
                presets.append(Preset.model_validate(json.load(f)))
    return presets


@router.get("/{name}", response_model=Preset)
async def get_preset(name: str):
    """Get a single preset by name."""
    path = _preset_path(name)
    with _presets_lock:
        if not os.path.exists(path):
            raise HTTPException(404, f"Preset '{name}' not found")
        with open(path, encoding="utf-8") as f:
            return Preset.model_validate(json.load(f))


@router.put("/{name}", response_model=Preset)
async def save_preset(name: str, preset: Preset):
    """Create or overwrite a preset."""
    safe = _safe_name(name)
    if not safe:
        raise HTTPException(400, "Invalid preset name")

    preset.name = safe
    with _presets_lock:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        _atomic_write(_preset_path(safe), preset.model_dump_json(indent=2))
    return preset


@router.delete("/{name}")
async def delete_preset(name: str):
    """Delete a preset."""
    path = _preset_path(name)
    with _presets_lock:
        if not os.path.exists(path):
            raise HTTPException(404, f"Preset '{name}' not found")
        os.unlink(path)
    return {"deleted": _safe_name(name)}
