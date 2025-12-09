#!/bin/bash
# Cron watchdog wrapper running every 15 minutes
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR" || exit 1
python3 scripts/watchdog.py "$@"
