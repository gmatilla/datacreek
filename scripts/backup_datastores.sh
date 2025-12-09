#!/bin/bash
# Backup LMDB and Redis state before migrations.
# Usage: ./backup_datastores.sh [backup_dir]
# Environment variables:
#   LMDB_PATH       Path to LMDB directory (default: ./data/lmdb)
#   REDIS_BACKUP_CMD  Command to dump redis database (default: redis-cli --rdb)

set -e

OUT_DIR="${1:-backups}"
LMDB_SRC="${LMDB_PATH:-./data/lmdb}"
mkdir -p "$OUT_DIR"
if [ -d "$LMDB_SRC" ]; then
  cp -a "$LMDB_SRC" "$OUT_DIR/lmdb_backup"
fi
if [ -n "$REDIS_BACKUP_CMD" ]; then
  $REDIS_BACKUP_CMD "$OUT_DIR/redis_dump.rdb"
else
  redis-cli --rdb "$OUT_DIR/redis_dump.rdb"
fi
echo "Backups saved to $OUT_DIR"
