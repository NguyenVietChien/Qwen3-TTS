# Qwen3-TTS Docker Service

Async TTS service với task polling, Redis + Celery, GPU support.

## Quick Start (GPU Server)

```bash
# 1. Clone và setup
git clone <repo>
cd Qwen3-TTS
cp .env.example .env
# Chỉnh MODEL_PATH trong .env nếu cần

# 2. Build và start với GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# 3. Kiểm tra
curl http://localhost:8000/health
```

## API Usage

### Submit TTS job (custom_voice / voice_design)
```bash
curl -X POST http://localhost:8000/api/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "mode": "custom_voice",
    "speaker": "Chelsie",
    "language": "English"
  }'
# → {"task_id": "uuid-...", "status": "pending"}
```

### Submit voice clone job
```bash
curl -X POST http://localhost:8000/api/tts/voice-clone \
  -F "text=Hello test" \
  -F "language=English" \
  -F "ref_text=Sample transcript" \
  -F "ref_audio=@/path/to/audio.wav"
# → {"task_id": "uuid-...", "status": "pending"}
```

### Poll task status
```bash
curl http://localhost:8000/api/tasks/{task_id}
# → {"task_id": "...", "status": "done", "audio_url": "/api/tasks/{id}/audio"}
```

### Download audio
```bash
curl -o output.wav http://localhost:8000/api/tasks/{task_id}/audio
```

## Architecture

```
Client → FastAPI (port 8000) → Redis → Celery Worker (GPU)
                                          ↓
                                    /data/tts_outputs/{task_id}.wav
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3-TTS` | HuggingFace model ID |
| `DEVICE` | `cuda:0` | PyTorch device |
| `DTYPE` | `bfloat16` | Model dtype |
| `OUTPUT_DIR` | `/data/tts_outputs` | Audio output dir |
| `API_PORT` | `8000` | API server port |

## Generation Parameters

Tất cả endpoints hỗ trợ các params sau (optional):

- `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`, `max_new_tokens`
- `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature` (12Hz tokenizer v2)

## Notes

- Model load lần đầu sẽ tải từ HuggingFace (~vài GB), cache tại `/root/.cache/huggingface`
- CPU inference rất chậm (~5-10 phút/câu), GPU nhanh hơn 30-50x
- Task status: `pending` → `processing` → `done` | `error`
