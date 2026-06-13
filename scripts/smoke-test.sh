#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.gpu.yml)
HEALTH_URL="${1:-${HEALTH_URL:-http://localhost/health}}"

echo "Checking compose configuration..."
docker compose "${COMPOSE_FILES[@]}" config --quiet

if [ -f frontend/package.json ]; then
  echo "Building frontend..."
  (cd frontend && npm install && npm run build)
fi

echo "Checking running containers..."
docker compose "${COMPOSE_FILES[@]}" ps

echo "Checking health endpoint: ${HEALTH_URL}"
curl --fail --show-error --silent "${HEALTH_URL}"
echo

echo "Smoke test passed."
