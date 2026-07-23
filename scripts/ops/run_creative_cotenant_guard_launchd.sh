#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
SUPPORT_NICE="${OPS_SUPPORT_JOB_NICE:-14}"
ACTION="${CREATIVE_COTENANT_GUARD_ACTION:-status}"
TIMEOUT_SECONDS="${CREATIVE_COTENANT_GUARD_TIMEOUT_SECONDS:-45}"
LIGHTWEIGHT_FLAG="${CREATIVE_COTENANT_GUARD_LIGHTWEIGHT:-1}"
LOCK_DIR="${CREATIVE_COTENANT_GUARD_LOCK_DIR:-$PROJECT_ROOT/governance/locks/creative_cotenant_guard_launchd.lockdir}"

mkdir -p "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  existing_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    existing_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  if [[ "$existing_pid" == <-> ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    echo "creative_cotenant_guard skip existing_pid=$existing_pid"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi

printf '%s\n' "$$" > "$LOCK_DIR/pid"
cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT INT TERM

guard_cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/creative_cotenant_guard.py" "$ACTION")
if [[ "$LIGHTWEIGHT_FLAG" == "1" || "$LIGHTWEIGHT_FLAG" == "true" || "$LIGHTWEIGHT_FLAG" == "yes" || "$LIGHTWEIGHT_FLAG" == "on" ]]; then
  guard_cmd+=(--lightweight)
fi

/usr/bin/nice -n "$SUPPORT_NICE" "${guard_cmd[@]}" &
child_pid="$!"
deadline=$((SECONDS + TIMEOUT_SECONDS))

while kill -0 "$child_pid" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    kill "$child_pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$child_pid" >/dev/null 2>&1 || true
    echo "creative_cotenant_guard timeout child_pid=$child_pid timeout_seconds=$TIMEOUT_SECONDS" >&2
    exit 124
  fi
  sleep 1
done

wait "$child_pid"
