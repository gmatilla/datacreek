#!/bin/bash
# Start all services using docker compose
DIR="$(dirname "$0")/.."
cd "$DIR" || exit 1
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi
docker compose up -d
