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

print("=" * 50)
if device.startswith("cuda"):
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  🚀 GPU MODE: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"  dtype: {dtype}")
else:
    print(f"  🐢 CPU MODE (no NVIDIA GPU detected)")
    print(f"  dtype: {dtype}")
    print(f"  ⚠ Warning: CPU mode is 5-10x slower than GPU!")
print("=" * 50)

# Load model (downloads on first run ~3.4GB)
print(f"Loading model: {MODEL_PATH} ...")
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
print(f"Model loaded in {time.time()-t0:.1f}s")

# ── Reference voice ──
# Use local sample_voice.MP3 from the project root
ref_audio = os.path.join(os.path.dirname(__file__), "samples", "sample_voice.MP3")
if not os.path.exists(ref_audio):
    print(f"ERROR: Reference audio not found: {ref_audio}")
    exit(1)
print(f"Reference audio: {ref_audio}")

# Reference text — what the person says in sample_voice.MP3
# (Leave empty string if you don't know the transcript — model will still work)
ref_text = ""

# ── Text to synthesize (read from input.txt) ──
input_file = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file):
    print(f"ERROR: input.txt not found. Create it and paste your text inside.")
    exit(1)
with open(input_file, "r", encoding="utf-8") as f:
    syn_text = f.read().strip()
if not syn_text:
    print("ERROR: input.txt is empty. Paste some text to synthesize.")
    exit(1)
syn_lang = "Auto"

print(f"Generating voice clone...")
print(f"  Text: {syn_text[:100]}{'...' if len(syn_text) > 100 else ''}")

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

# Build output filename: Language_First10Words_HH_MM_dd_mm_yyyy.wav
from datetime import datetime
import re
lang_tag = syn_lang if syn_lang != "Auto" else "Auto"
words = re.split(r'\s+', syn_text.strip())
first_10 = " ".join(words[:10])
# Sanitize for filename
first_10_safe = re.sub(r'[<>:"/\\|?*]', '', first_10).strip()
first_10_safe = first_10_safe[:80]  # cap length
timestamp = datetime.now().strftime("%H_%M_%d_%m_%Y")
out_name = f"{lang_tag}_{first_10_safe}_{timestamp}.wav"
out_file = os.path.join(OUT_DIR, out_name)

sf.write(out_file, wavs[0], sr)
print(f"\nDone! Generated in {elapsed:.1f}s")
print(f"Output: {os.path.abspath(out_file)}")
print(f"Sample rate: {sr}  |  Duration: {len(wavs[0])/sr:.1f}s")
