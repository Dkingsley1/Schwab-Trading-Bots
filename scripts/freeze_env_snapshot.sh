#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/exports/env_snapshots/$STAMP"
PY="$PROJECT_ROOT/.venv312/bin/python"
PIP="$PROJECT_ROOT/.venv312/bin/pip"

mkdir -p "$OUT_DIR"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=$PROJECT_ROOT"
  uname -a
} > "$OUT_DIR/system.txt"

"$PY" -V > "$OUT_DIR/python_version.txt" 2>&1 || true
"$PIP" freeze | sort > "$OUT_DIR/requirements_freeze.txt" 2>/dev/null || true

cp "$OUT_DIR/requirements_freeze.txt" "$PROJECT_ROOT/config/requirements.lock.txt" 2>/dev/null || true

echo "$OUT_DIR"
