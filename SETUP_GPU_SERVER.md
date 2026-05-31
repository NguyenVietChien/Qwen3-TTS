# GPU Server Setup Guide

## 1. Fix DNS (nếu cần)

```bash
sudo chattr -i /etc/resolv.conf
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 1.1.1.1" >> /etc/resolv.conf'
sudo bash -c 'echo "options edns0 trust-ad" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf
ping -c 3 google.com
```

## 2. Cài Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

## 3. Cài NVIDIA Driver

```bash
sudo apt install -y nvidia-driver-580 nvidia-utils-580
```

## 4. Cài NVIDIA Container Toolkit

```bash
cd /tmp
wget https://github.com/NVIDIA/libnvidia-container/releases/download/v1.17.4/libnvidia-container1_1.17.4-1_amd64.deb
wget https://github.com/NVIDIA/libnvidia-container/releases/download/v1.17.4/libnvidia-container-tools_1.17.4-1_amd64.deb
wget https://github.com/NVIDIA/nvidia-container-toolkit/releases/download/v1.17.4/nvidia-container-toolkit_1.17.4-1_amd64.deb
sudo dpkg -i libnvidia-container1_1.17.4-1_amd64.deb
sudo dpkg -i libnvidia-container-tools_1.17.4-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit_1.17.4-1_amd64.deb
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 5. Reboot

```bash
sudo reboot
```

## 6. Test GPU (sau reboot)

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 7. Deploy TTS Service

```bash
cd ~
git clone <repo-url> Qwen3-TTS
cd Qwen3-TTS
cp .env.example .env
nano .env
```

Sửa `.env`:
```
MODEL_PATH=Qwen/Qwen3-TTS
DEVICE=cuda:0
DTYPE=bfloat16
API_PORT=8000
OUTPUT_DIR=/data/tts_outputs
```

Build và start:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Theo dõi logs:
```bash
docker compose logs -f worker
```

## 8. Test API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","mode":"custom_voice","speaker":"Chelsie","language":"English"}'
```
