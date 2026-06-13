# coding=utf-8
"""Provider-agnostic long-text chunking + audio merging.

Splits long text into TTS-friendly chunks at natural boundaries
(paragraph > sentence > clause > whitespace), and merges the resulting
WAV arrays back into a single track with optional silence gaps.

Designed to be dropped into any TTS provider/worker with no framework
dependencies — only `numpy` (already a torch dependency) is required,
and only for `merge_wavs`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Chunk profiles — target/max/min character counts per script.
# Providers can register their own profile in CHUNK_PROFILES; falls back
# to DEFAULT_PROFILE if the provider or script isn't found.
# ---------------------------------------------------------------------------

DEFAULT_PROFILE: Dict[str, Dict[str, int]] = {
    "default": {"target_chars": 600, "max_chars": 750, "min_chars": 200},
    "latin": {"target_chars": 600, "max_chars": 750, "min_chars": 200},
    "cjk": {"target_chars": 260, "max_chars": 320, "min_chars": 100},
    "thai": {"target_chars": 500, "max_chars": 700, "min_chars": 180},
    "arabic": {"target_chars": 600, "max_chars": 720, "min_chars": 200},
    "hebrew": {"target_chars": 600, "max_chars": 720, "min_chars": 200},
    "devanagari": {"target_chars": 560, "max_chars": 700, "min_chars": 200},
}

CHUNK_PROFILES: Dict[str, Dict[str, Dict[str, int]]] = {
    "qwen3": {
        "default": {"target_chars": 340, "max_chars": 420, "min_chars": 120},
        "latin": {"target_chars": 380, "max_chars": 450, "min_chars": 140},
        "cjk": {"target_chars": 220, "max_chars": 260, "min_chars": 90},
        "thai": {"target_chars": 260, "max_chars": 340, "min_chars": 100},
        "arabic": {"target_chars": 320, "max_chars": 400, "min_chars": 120},
        "hebrew": {"target_chars": 320, "max_chars": 400, "min_chars": 120},
        "devanagari": {"target_chars": 300, "max_chars": 380, "min_chars": 120},
    },
}

MAX_TOTAL_CHARS = 60_000
MAX_CHUNKS = 80


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def infer_script(text: str, language_code: Optional[str] = None) -> str:
    """Best-effort script classification, used to pick a chunk profile."""
    normalized_language = str(language_code or "").strip().lower().replace("_", "-")
    if normalized_language:
        if normalized_language.startswith(("ja", "zh", "ko")):
            return "cjk"
        if normalized_language.startswith("th"):
            return "thai"
        if normalized_language.startswith(("ar", "fa", "ur")):
            return "arabic"
        if normalized_language.startswith(("he", "yi")):
            return "hebrew"
        if normalized_language.startswith(("hi", "ne", "mr", "sa")):
            return "devanagari"

    counts: Dict[str, int] = {
        "latin": 0, "cjk": 0, "thai": 0, "arabic": 0, "hebrew": 0, "devanagari": 0,
    }
    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF:
            counts["cjk"] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            counts["thai"] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
            counts["arabic"] += 1
        elif 0x0590 <= code <= 0x05FF:
            counts["hebrew"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif char.isalpha():
            counts["latin"] += 1

    if not any(counts.values()):
        return "default"
    return max(counts.items(), key=lambda item: item[1])[0]


def resolve_chunk_profile(
    provider: str,
    text: str = "",
    language_code: Optional[str] = None,
) -> Dict[str, int]:
    """Pick target/max/min chars for `provider` based on the text's script."""
    provider_profiles = CHUNK_PROFILES.get(provider, DEFAULT_PROFILE)
    script = infer_script(text, language_code)
    profile = provider_profiles.get(script) or provider_profiles.get("default") or DEFAULT_PROFILE["default"]
    return dict(profile)


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

_SENTENCE_END_CHARS = ".?!。！？"
_CLAUSE_END_CHARS = ",;:、，；："


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _trim_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _boundary_candidates(text: str, min_pos: int, max_pos: int) -> List[Tuple[int, int]]:
    """Return (priority, cut_position) candidates within [min_pos, max_pos).

    Lower priority is preferred: 0 = paragraph break, 1 = sentence end,
    2 = clause break, 3 = fallback whitespace.
    """
    candidates: List[Tuple[int, int]] = []
    for idx in range(min_pos, max_pos):
        char = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        cut = idx + 1
        if char == "\n" and nxt == "\n":
            candidates.append((0, _skip_whitespace(text, cut)))
            continue
        if char in _SENTENCE_END_CHARS and (not nxt or nxt.isspace()):
            candidates.append((1, cut))
            continue
        if char in _CLAUSE_END_CHARS and (not nxt or nxt.isspace()):
            candidates.append((2, cut))

    if not candidates:
        for idx in range(max_pos, min_pos, -1):
            if text[idx - 1].isspace():
                candidates.append((3, idx))
                break

    return candidates


def split_text(text: str, profile: Dict[str, int]) -> List[str]:
    """Split `text` into chunks within [min_chars, max_chars], preferring
    cuts near `target_chars` at paragraph/sentence/clause/whitespace
    boundaries (in that priority order).

    Raises ValueError if a chunk would exceed max_chars with no usable
    boundary (e.g. one giant token with no whitespace).
    """
    text = text.strip()
    if not text:
        return []

    target_chars = int(profile.get("target_chars") or DEFAULT_PROFILE["default"]["target_chars"])
    max_chars = int(profile.get("max_chars") or DEFAULT_PROFILE["default"]["max_chars"])
    min_chars = int(profile.get("min_chars") or DEFAULT_PROFILE["default"]["min_chars"])

    chunks: List[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_chars:
            cut = len(text)
        else:
            min_pos = min(start + min_chars, len(text))
            max_pos = min(start + max_chars, len(text))
            target_pos = min(start + target_chars, max_pos)
            candidates = _boundary_candidates(text, min_pos, max_pos)
            if candidates:
                _, cut = sorted(candidates, key=lambda item: (abs(item[1] - target_pos), item[0]))[0]
            else:
                window = text[start:max_pos]
                if window and window.isascii() and not any(c.isspace() for c in window):
                    raise ValueError(
                        f"Chunk contains a token too large ({len(window)} chars, max {max_chars})"
                    )
                cut = max_pos

        chunk_start = _skip_whitespace(text, start)
        chunk_end = _trim_end(text, chunk_start, cut)
        if chunk_end > chunk_start:
            chunks.append(text[chunk_start:chunk_end])
        start = _skip_whitespace(text, cut)

    return chunks


def validate_chunks(chunks: List[str], profile: Dict[str, int]) -> None:
    """Raise ValueError if the split result exceeds sane limits."""
    max_chars = int(profile.get("max_chars") or DEFAULT_PROFILE["default"]["max_chars"])
    total_chars = sum(len(c) for c in chunks)
    if total_chars > MAX_TOTAL_CHARS:
        raise ValueError(f"Text too large ({total_chars} chars, max {MAX_TOTAL_CHARS})")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"Too many chunks ({len(chunks)}, max {MAX_CHUNKS})")
    for chunk in chunks:
        if len(chunk) > max_chars:
            raise ValueError(f"Chunk too large ({len(chunk)} chars, max {max_chars})")


# ---------------------------------------------------------------------------
# Audio merging
# ---------------------------------------------------------------------------

def merge_wavs(
    wavs: List[np.ndarray],
    sample_rate: int,
    gap_ms: int = 0,
) -> np.ndarray:
    """Concatenate mono/multi-channel WAV arrays with optional silence gaps.

    All arrays must share `sample_rate` and dtype/shape conventions
    (caller is expected to pass arrays straight from the same model).
    """
    if not wavs:
        return np.zeros(0, dtype=np.float32)
    if len(wavs) == 1:
        return wavs[0]

    gap_samples = int(sample_rate * gap_ms / 1000)
    parts: List[np.ndarray] = []
    silence = None
    if gap_samples > 0:
        shape = (gap_samples,) + wavs[0].shape[1:]
        silence = np.zeros(shape, dtype=wavs[0].dtype)

    for i, wav in enumerate(wavs):
        parts.append(wav)
        if silence is not None and i < len(wavs) - 1:
            parts.append(silence)

    return np.concatenate(parts, axis=0)
