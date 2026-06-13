# coding=utf-8
"""Pydantic request/response schemas for the Qwen3-TTS API server."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TTSMode(str, Enum):
    """Which generation API to use."""
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class GenerationParams(BaseModel):
    """Optional generation hyper-parameters (all have sensible defaults)."""
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    # Sub-talker params (12 Hz tokenizer v2)
    subtalker_dosample: Optional[bool] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None


class TTSRequest(BaseModel):
    """
    Main TTS generation request.

    - custom_voice mode: requires `speaker`
    - voice_design mode: requires `instruct`
    - voice_clone mode: requires ref audio (sent separately via multipart)
    """
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field("Auto", description="Language name, e.g. 'Chinese', 'English', 'Auto'")
    mode: TTSMode = Field(TTSMode.CUSTOM_VOICE, description="Generation mode")
    speaker: Optional[str] = Field(None, description="Speaker name (custom_voice mode)")
    instruct: Optional[str] = Field(None, description="Voice style instruction (voice_design / custom_voice)")
    generation_params: Optional[GenerationParams] = None
    chunk_gap_ms: int = Field(0, ge=0, description="Silence (ms) inserted between auto-split long-text chunks")


class VoiceCloneRequest(BaseModel):
    """Voice clone request body (ref audio is sent as multipart file)."""
    text: str = Field(..., min_length=1)
    language: str = Field("Auto")
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio")
    x_vector_only: bool = Field(False, description="Use x-vector only (no ICL)")
    generation_params: Optional[GenerationParams] = None
    chunk_gap_ms: int = Field(0, ge=0, description="Silence (ms) inserted between auto-split long-text chunks")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    """Model metadata returned by GET /api/tts/info."""
    model_type: Optional[str] = None
    supported_languages: Optional[List[str]] = None
    supported_speakers: Optional[List[str]] = None
    status: str = "ready"


# ---------------------------------------------------------------------------
# Task response
# ---------------------------------------------------------------------------

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    audio_url: Optional[str] = None
    error: Optional[str] = None
