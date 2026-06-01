# coding=utf-8
"""Task polling endpoints."""

import os

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..celery_app import celery_app
from ..models import TaskResponse, TaskStatus

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/tts_outputs")


def _map_state(state: str) -> TaskStatus:
    if state in ("PENDING", "RECEIVED"):
        return TaskStatus.PENDING
    if state in ("STARTED", "PROCESSING", "RETRY"):
        return TaskStatus.PROCESSING
    if state == "SUCCESS":
        return TaskStatus.DONE
    if state in ("FAILURE", "REVOKED"):
        return TaskStatus.ERROR
    return TaskStatus.PENDING


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Poll task status. audio_url is populated when status == done."""
    result = AsyncResult(task_id, app=celery_app)
    status = _map_state(result.state)

    audio_url = None
    error = None

    if result.state == "SUCCESS":
        if (result.result or {}).get("audio_path"):
            audio_url = f"/api/tasks/{task_id}/audio"
    elif result.state == "FAILURE":
        error = str(result.result)

    return TaskResponse(task_id=task_id, status=status, audio_url=audio_url, error=error)


@router.get("/{task_id}/audio")
async def get_task_audio(task_id: str):
    """Download the generated WAV file for a completed task."""
    result = AsyncResult(task_id, app=celery_app)
    if result.state != "SUCCESS":
        raise HTTPException(404, "Audio not ready yet")

    audio_path = (result.result or {}).get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(404, "Audio file not found on server")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{task_id}.wav")
