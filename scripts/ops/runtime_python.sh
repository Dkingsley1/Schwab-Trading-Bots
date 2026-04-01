#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

runtime_lane() {
  local lane="${BOT_RUNTIME_LANE:-${BOT_PYTHON_RUNTIME:-production}}"
  lane="${lane:l}"
  if [[ -z "$lane" ]]; then
    lane="production"
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
    shadow314|py314|canary314|python314)
      print -r -- "3.14"
      ;;
    *)
      print -r -- "3.12"
      ;;
  esac
}

resolve_runtime_python() {
  if [[ -n "${BOT_PYTHON_BIN:-}" ]]; then
    print -r -- "${BOT_PYTHON_BIN}"
    return 0
  fi

  local version
  version="$(runtime_version)"
  local -a candidates=()
  if [[ "$version" == 3.14* ]]; then
    candidates=(
      "$PROJECT_ROOT/.venv314/bin/python"
      "$PROJECT_ROOT/.venv313/bin/python"
      "$PROJECT_ROOT/.venv312/bin/python"
    )
  else
    candidates=(
      "$PROJECT_ROOT/.venv312/bin/python"
      "$PROJECT_ROOT/.venv314/bin/python"
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

if [[ "${ZSH_EVAL_CONTEXT:-}" == "toplevel" ]]; then
  resolve_runtime_python
fi
