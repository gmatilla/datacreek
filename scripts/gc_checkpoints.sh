#!/usr/bin/env bash
# Remove stale training checkpoints older than 30 days except those marked as best.
# Usage: gc_checkpoints.sh [checkpoints_dir]
# The script prints each deleted file for auditing.

set -euo pipefail

DIR="${1:-checkpoints}"

# Find regular files older than 30 days whose name does not start with "best"
# and delete them. The `-print` flag logs each removed file for debugging.
find "$DIR" -type f -mtime +30 ! -name 'best*' -print -delete
