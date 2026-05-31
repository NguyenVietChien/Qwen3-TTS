#!/bin/bash
set -e

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Frontend built successfully at frontend/dist/"
echo "Ready to deploy with docker compose up"
