#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/project_timeline_report.py"
GUARD_ARTIFACT="$PROJECT_ROOT/governance/health/chrome_headless_guard_latest.json"

AUTO_RENDER="${PROJECT_TIMELINE_AUTO_RENDER_PDF:-1}"
ALLOW_GUI="${PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER:-1}"

if [[ -f "$GUARD_ARTIFACT" ]]; then
  POLICY="$("$PYTHON_BIN" -c 'import json, sys; from pathlib import Path; path=Path(sys.argv[1]); payload=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}; print(str(payload.get("timeline_pdf_policy") or "allow"))' "$GUARD_ARTIFACT" 2>/dev/null || printf 'allow')"
  case "$POLICY" in
    suppress)
      AUTO_RENDER="0"
      ALLOW_GUI="0"
      ;;
    headless_only)
      ALLOW_GUI="0"
      ;;
  esac
fi

export PROJECT_TIMELINE_AUTO_RENDER_PDF="$AUTO_RENDER"
export PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER="$ALLOW_GUI"

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" project_timeline_autoupdate \
  "$PYTHON_BIN" "$RUN_SCRIPT" "$@"
