#!/bin/bash
# Restore LMDB and Redis state from a backup directory.
# Usage: ./revert_cache.sh [backup_dir]
# Environment variables:
#   LMDB_PATH        Path to LMDB directory to restore (default: ./data/lmdb)
#   REDIS_RESTORE_CMD  Command to restore redis database (default: redis-cli --pipe)

set -e

BACKUP_DIR="${1:-backups}"
LMDB_DST="${LMDB_PATH:-./data/lmdb}"
if [ -d "$BACKUP_DIR/lmdb_backup" ]; then
  rm -rf "$LMDB_DST"
  cp -a "$BACKUP_DIR/lmdb_backup" "$LMDB_DST"
fi
if [ -f "$BACKUP_DIR/redis_dump.rdb" ]; then
  if [ -n "$REDIS_RESTORE_CMD" ]; then
    $REDIS_RESTORE_CMD "$BACKUP_DIR/redis_dump.rdb"
  else
    redis-cli --pipe < "$BACKUP_DIR/redis_dump.rdb"
  fi
fi
echo "Cache restored from $BACKUP_DIR"
