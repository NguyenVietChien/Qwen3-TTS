# coding=utf-8
"""Celery task definitions — run exclusively inside the worker process."""

import logging
import os

import soundfile as sf
import torch

from .celery_app import celery_app
from .chunking import merge_wavs, resolve_chunk_profile, split_text, validate_chunks

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/tts_outputs")
MODEL_PATH_CUSTOM = os.environ.get("MODEL_PATH_CUSTOM", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
MODEL_PATH_BASE = os.environ.get("MODEL_PATH_BASE", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
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

_model_custom = None  # custom_voice + voice_design
_model_base = None    # voice_clone


def _load_model(model_path: str):
    from qwen_tts import Qwen3TTSModel
    dtype = _DTYPE_MAP.get(DTYPE_STR, torch.bfloat16)
    logger.info("Loading model %s on %s (%s)…", model_path, DEVICE, DTYPE_STR)
    model = Qwen3TTSModel.from_pretrained(model_path, dtype=dtype, device_map=DEVICE)
    logger.info("Model loaded ✓  type=%s", getattr(model.model, "tts_model_type", "?"))
    return model


def _get_custom_model():
    global _model_custom
    if _model_custom is None:
        _model_custom = _load_model(MODEL_PATH_CUSTOM)
    return _model_custom


def _get_base_model():
    global _model_base
    if _model_base is None:
        _model_base = _load_model(MODEL_PATH_BASE)
    return _model_base


VOICES_DIR = os.environ.get("VOICES_DIR", "/data/voices")


@celery_app.task(bind=True, name="tts.preprocess_voice")
def task_preprocess_voice(self, voice_name: str, ref_audio_paths: list,
                          ref_text: str = None, x_vector_only: bool = True):
    """Pre-compute VoiceClonePromptItem tensors and save to disk. Audio files deleted after."""
    import re
    import torch

    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    tts = _get_base_model()

    self.update_state(state="PROCESSING", meta={"step": "extracting_voice"})

    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name).strip("_")
    voice_dir = os.path.join(VOICES_DIR, safe)
    os.makedirs(voice_dir, exist_ok=True)

    try:
        ref_audio = ref_audio_paths if len(ref_audio_paths) > 1 else ref_audio_paths[0]
        items = tts.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

        # Move tensors to CPU before saving
        cpu_items = []
        for item in items:
            cpu_items.append({
                "ref_code": item.ref_code.cpu() if item.ref_code is not None else None,
                "ref_spk_embedding": item.ref_spk_embedding.cpu(),
                "x_vector_only_mode": item.x_vector_only_mode,
                "icl_mode": item.icl_mode,
                "ref_text": item.ref_text,
            })

        prompt_path = os.path.join(voice_dir, "prompt.pt")
        torch.save(cpu_items, prompt_path)
        logger.info("Voice '%s' preprocessed → %s (%d samples)", safe, prompt_path, len(cpu_items))

    finally:
        # Delete uploaded audio files after extraction
        for path in ref_audio_paths:
            if path and os.path.exists(path):
                os.unlink(path)

    return {"voice_name": safe, "samples": len(cpu_items), "prompt_path": prompt_path}


def _load_voice_prompt(voice_name: str):
    """Load pre-computed VoiceClonePromptItem tensors from disk."""
    import re
    import torch
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name).strip("_")
    prompt_path = os.path.join(VOICES_DIR, safe, "prompt.pt")
    if not os.path.exists(prompt_path):
        return None

    cpu_items = torch.load(prompt_path, map_location="cpu", weights_only=False)
    device = torch.device(DEVICE)

    items = []
    for d in cpu_items:
        items.append(VoiceClonePromptItem(
            ref_code=d["ref_code"].to(device) if d["ref_code"] is not None else None,
            ref_spk_embedding=d["ref_spk_embedding"].to(device),
            x_vector_only_mode=d["x_vector_only_mode"],
            icl_mode=d["icl_mode"],
            ref_text=d["ref_text"],
        ))
    return items


def _save_wav(task_id: str, wavs, sr: int) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    sf.write(out_path, wavs[0], sr, format="WAV", subtype="PCM_16")
    return out_path


def _clean_gen_params(gen_params: dict) -> dict:
    return {k: v for k, v in (gen_params or {}).items() if v is not None}


def _generate_chunked(self, text: str, language: str, chunk_gap_ms: int, generate_fn):
    """Split `text` if needed, run `generate_fn(chunk_text) -> (wavs, sr)` per
    chunk, and merge the results into a single (wavs, sr) pair.

    `generate_fn` should return the model's (wavs, sr) for one chunk of text;
    only wavs[0] from each chunk is kept and merged.
    """
    profile = resolve_chunk_profile("qwen3", text, language)
    chunks = split_text(text, profile) or [text]
    validate_chunks(chunks, profile)

    wavs_list = []
    sr = None
    for i, chunk_text in enumerate(chunks):
        self.update_state(
            state="PROCESSING",
            meta={"step": "generating", "chunk": i + 1, "total_chunks": len(chunks)},
        )
        wavs, sr = generate_fn(chunk_text)
        wavs_list.append(wavs[0])

    merged = merge_wavs(wavs_list, sr, gap_ms=chunk_gap_ms)
    return [merged], sr


@celery_app.task(bind=True, name="tts.generate")
def task_generate(self, text: str, mode: str, language: str, gen_params: dict,
                  speaker: str = None, instruct: str = None, chunk_gap_ms: int = 0):
    """Generate TTS audio (custom_voice or voice_design mode)."""
    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    tts = _get_custom_model()
    gen_kw = _clean_gen_params(gen_params)

    if mode == "custom_voice":
        def generate_fn(chunk_text):
            return tts.generate_custom_voice(
                text=chunk_text, speaker=speaker, language=language, instruct=instruct, **gen_kw,
            )
    elif mode == "voice_design":
        def generate_fn(chunk_text):
            return tts.generate_voice_design(
                text=chunk_text, instruct=instruct, language=language, **gen_kw,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    wavs, sr = _generate_chunked(self, text, language, chunk_gap_ms, generate_fn)
    out_path = _save_wav(self.request.id, wavs, sr)
    return {"audio_path": out_path, "sample_rate": sr}


@celery_app.task(bind=True, name="tts.voice_clone")
def task_voice_clone(self, text: str, language: str, gen_params: dict,
                     voice_name: str = None, ref_audio_paths: list = None,
                     ref_text: str = None, x_vector_only: bool = False,
                     delete_after: bool = False, chunk_gap_ms: int = 0):
    """Generate TTS audio via voice cloning.

    Priority:
    1. voice_name with prompt.pt → use pre-computed tensors (fastest)
    2. voice_name without prompt.pt → load audio files from voice dir
    3. ref_audio_paths → use uploaded audio files
    """
    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    tts = _get_base_model()
    gen_kw = _clean_gen_params(gen_params)

    try:
        # 1. Pre-computed prompt — fastest, no audio re-processing
        prompt_items = _load_voice_prompt(voice_name) if voice_name else None
        if prompt_items:
            def generate_fn(chunk_text):
                return tts.generate_voice_clone(
                    text=chunk_text, language=language, voice_clone_prompt=prompt_items, **gen_kw,
                )
        else:
            # 2. Fall back to audio files
            paths = ref_audio_paths or []
            if not paths and voice_name:
                # Load audio from saved voice dir
                import re
                safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name).strip("_")
                voice_dir = os.path.join(VOICES_DIR, safe)
                paths = sorted(
                    os.path.join(voice_dir, f)
                    for f in os.listdir(voice_dir)
                    if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".m4a")
                ) if os.path.isdir(voice_dir) else []

            if not paths:
                raise ValueError("No reference audio available")

            ref_audio = paths if len(paths) > 1 else paths[0]

            def generate_fn(chunk_text):
                return tts.generate_voice_clone(
                    text=chunk_text, language=language, ref_audio=ref_audio,
                    ref_text=ref_text, x_vector_only_mode=x_vector_only, **gen_kw,
                )

        wavs, sr = _generate_chunked(self, text, language, chunk_gap_ms, generate_fn)

    finally:
        if delete_after and ref_audio_paths:
            for path in ref_audio_paths:
                if path and os.path.exists(path):
                    os.unlink(path)

    out_path = _save_wav(self.request.id, wavs, sr)
    return {"audio_path": out_path, "sample_rate": sr}
