# coding=utf-8
"""
Quick voice clone test — downloads 1.7B Base model + generates a WAV file.

Usage:
    .venv\Scripts\activate
    python test_voice_clone.py
"""
import os
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUT_DIR = "test_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Auto-detect device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
print(f"Device: {device}  |  dtype: {dtype}")

# Load model (downloads on first run ~3.4GB)
print(f"Loading model: {MODEL_PATH} ...")
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
print(f"Model loaded in {time.time()-t0:.1f}s")

# ── Reference voice ──
# Use local sample_voice.MP3 from the project root
ref_audio = os.path.join(os.path.dirname(__file__), "sample_voice.MP3")
if not os.path.exists(ref_audio):
    print(f"ERROR: Reference audio not found: {ref_audio}")
    exit(1)
print(f"Reference audio: {ref_audio}")

# Reference text — what the person says in sample_voice.MP3
# (Leave empty string if you don't know the transcript — model will still work)
ref_text = ""

# ── Text to synthesize ──
syn_text = "Xin chào mọi người, đây là bài test voice clone. Tôi muốn kiểm tra xem giọng nói được tổng hợp có giống giọng gốc hay không."
syn_lang = "Auto"

print(f"Generating voice clone...")
print(f"  Text: {syn_text}")

t0 = time.time()
wavs, sr = tts.generate_voice_clone(
    text=syn_text,
    language=syn_lang,
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=True,  # True = only uses voice characteristics (no transcript needed)
    max_new_tokens=2048,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.05,
)

if device.startswith("cuda"):
    torch.cuda.synchronize()
elapsed = time.time() - t0

out_file = os.path.join(OUT_DIR, "voice_clone_test.wav")
sf.write(out_file, wavs[0], sr)
print(f"\nDone! Generated in {elapsed:.1f}s")
print(f"Output: {os.path.abspath(out_file)}")
print(f"Sample rate: {sr}  |  Duration: {len(wavs[0])/sr:.1f}s")
