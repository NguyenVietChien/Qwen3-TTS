FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps: sox for audio processing, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps: server requirements first (layer cache)
COPY server/requirements.txt /tmp/server-requirements.txt
RUN pip install --no-cache-dir -r /tmp/server-requirements.txt

# Project deps from pyproject.toml
COPY pyproject.toml MANIFEST.in ./
COPY qwen_tts/ ./qwen_tts/
RUN pip install --no-cache-dir -e .

# Application code
COPY server/ ./server/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OUTPUT_DIR=/data/tts_outputs

EXPOSE 8000

# Default: run the API server. Override CMD in docker-compose to run the worker.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
