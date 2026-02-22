#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

cmd="${1:-}"
arg="${2:-}"

case "$cmd" in
  tag)
    ts="$(date -u +%Y%m%d_%H%M%S)"
    tag="release_${ts}"
    git tag -a "$tag" -m "immutable release $tag"
    echo "$tag"
    ;;
  rollback)
    if [[ -z "$arg" ]]; then
      echo "usage: release_ops.sh rollback <tag>"
      exit 2
    fi
    echo "Rollback command (manual review required): git checkout $arg"
    ;;
  *)
    echo "usage: release_ops.sh {tag|rollback <tag>}"
    exit 2
    ;;
esac
