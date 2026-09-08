#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export PROJECT_ROOT
export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

GUARD_SOURCE="$PROJECT_ROOT/scripts/ops/storage_eject_guard.swift"
GUARD_BINARY="${STORAGE_EJECT_GUARD_BINARY:-$HOME/Library/Application Support/schwab_trading_bot/bin/storage_eject_guard}"

if [[ -x "$GUARD_BINARY" && "$GUARD_BINARY" -nt "$GUARD_SOURCE" ]]; then
  exec "$GUARD_BINARY"
fi

SWIFT_CACHE_DIR="${TMPDIR:-/tmp}/schwab_trading_bot_swift_module_cache"
mkdir -p "$SWIFT_CACHE_DIR"
mkdir -p "$(dirname "$GUARD_BINARY")"
GUARD_BINARY_TMP="$GUARD_BINARY.tmp.$$"
CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" \
  /usr/bin/swiftc -O "$GUARD_SOURCE" -o "$GUARD_BINARY_TMP" && \
  chmod +x "$GUARD_BINARY_TMP" && \
  mv "$GUARD_BINARY_TMP" "$GUARD_BINARY" && \
  exec "$GUARD_BINARY"
rm -f "$GUARD_BINARY_TMP"

CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" \
  /usr/bin/swiftc -typecheck "$GUARD_SOURCE"
CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" \
  exec /usr/bin/swift "$GUARD_SOURCE"
