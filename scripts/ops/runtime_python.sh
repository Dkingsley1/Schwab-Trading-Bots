#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

load_python314_runtime_override() {
  local override_file="$PROJECT_ROOT/config/.env.python314_runtime_override"
  [[ -f "$override_file" ]] || return 0
  [[ "${BOT_RUNTIME_SKIP_PY314_OVERRIDE:-0}" == "1" ]] && return 0

  local runtime_explicit="${BOT_PYTHON_BIN:-}${BOT_PYTHON_VERSION:-}${BOT_RUNTIME_LANE:-}${BOT_PYTHON_RUNTIME:-}"
  local training_explicit="${BOT_TRAINING_PYTHON_BIN:-}${BOT_TRAINING_PYTHON_VERSION:-}${BOT_TRAINING_RUNTIME_LANE:-}${BOT_TRAINING_PYTHON_RUNTIME:-}"
  if [[ -n "$runtime_explicit" || -n "$training_explicit" ]]; then
    return 0
  fi

  set -a
  # shellcheck disable=SC1090
  source "$override_file"
  set +a
}

load_python314_runtime_override

runtime_lane() {
  local lane="${BOT_RUNTIME_LANE:-${BOT_PYTHON_RUNTIME:-auto}}"
  lane="${lane:l}"
  if [[ -z "$lane" ]]; then
    lane="auto"
  fi
  print -r -- "$lane"
}

runtime_version() {
  if [[ -n "${BOT_PYTHON_VERSION:-}" ]]; then
    print -r -- "${BOT_PYTHON_VERSION}"
    return 0
  fi
  local lane
  lane="$(runtime_lane)"
  case "$lane" in
    production|shadow314|py314|canary314|python314|py312|python312)
      print -r -- "3.14"
      ;;
    *)
      return 1
      ;;
  esac
}

training_lane() {
  local lane="${BOT_TRAINING_RUNTIME_LANE:-${BOT_TRAINING_PYTHON_RUNTIME:-training}}"
  lane="${lane:l}"
  if [[ -z "$lane" ]]; then
    lane="training"
  fi
  print -r -- "$lane"
}

training_version() {
  if [[ -n "${BOT_TRAINING_PYTHON_VERSION:-}" ]]; then
    print -r -- "${BOT_TRAINING_PYTHON_VERSION}"
    return 0
  fi
  local lane
  lane="$(training_lane)"
  case "$lane" in
    shadow314|py314|canary314|python314)
      print -r -- "3.14"
      ;;
    *)
      print -r -- "3.14"
      ;;
  esac
}

resolve_runtime_python() {
  if [[ -n "${BOT_PYTHON_BIN:-}" ]]; then
    print -r -- "${BOT_PYTHON_BIN}"
    return 0
  fi

  local -a candidates=()
  local version=""
  if version="$(runtime_version 2>/dev/null)"; then
    :
  elif [[ "${BOT_PREFER_MLX_RUNTIME:-}" =~ ^(1|true|yes|on)$ ]]; then
    version="3.14"
  elif [[ ! "${BOT_PREFER_MLX_RUNTIME:-}" =~ ^(0|false|no|off)$ ]] \
    && [[ -x "$PROJECT_ROOT/.venv314/bin/python" ]] \
    && "$PROJECT_ROOT/.venv314/bin/python" -c 'import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' mlx >/dev/null 2>&1; then
    version="3.14"
  else
    version="3.14"
  fi

  if [[ "$version" == 3.14* ]]; then
    candidates=(
      "$PROJECT_ROOT/.venv314/bin/python"
      "$PROJECT_ROOT/.venv313/bin/python"
    )
  else
    candidates=(
      "$PROJECT_ROOT/.venv314/bin/python"
      "$PROJECT_ROOT/.venv313/bin/python"
    )
  fi

  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      print -r -- "$path"
      return 0
    fi
  done

  print -r -- "${candidates[1]:-${candidates[0]}}"
}

resolve_training_python() {
  if [[ -n "${BOT_TRAINING_PYTHON_BIN:-}" ]]; then
    print -r -- "${BOT_TRAINING_PYTHON_BIN}"
    return 0
  fi

  local version
  version="$(training_version)"
  local -a candidates=()
  if [[ "$version" == 3.14* ]]; then
    candidates=(
      "$PROJECT_ROOT/.venv314/bin/python"
      "$PROJECT_ROOT/.venv313/bin/python"
    )
  else
    candidates=(
      "$PROJECT_ROOT/.venv314/bin/python"
      "$PROJECT_ROOT/.venv313/bin/python"
    )
  fi

  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]] && "$path" -c 'import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' mlx >/dev/null 2>&1; then
      print -r -- "$path"
      return 0
    fi
  done

  resolve_runtime_python
}

if [[ "${ZSH_EVAL_CONTEXT:-}" == "toplevel" ]]; then
  case "${BOT_PYTHON_PURPOSE:-runtime}" in
    training)
      resolve_training_python
      ;;
    *)
      resolve_runtime_python
      ;;
  esac
fi
