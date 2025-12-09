#!/bin/bash
# Restart ingestion services to roll back recent updates.
# Usage: ./revert_ingestion.sh

set -e
DIR="$(dirname "$0")/.."
cd "$DIR" || exit 1
docker compose restart ingestion

echo "Ingestion service rolled back"
