#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="$PROJECT_ROOT/config/run_profiles"
OUT_LATEST="$PROJECT_ROOT/governance/session_configs/profile_applied_latest.json"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/apply_run_profile.sh <profile> [--exports]

Examples:
  ./scripts/apply_run_profile.sh conservative_weekday --exports
  eval "$(./scripts/apply_run_profile.sh event_mode --exports)"

Profiles:
  conservative_weekday
  event_mode
  overnight_low_load
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

PROFILE="$1"
MODE="info"
if [[ "${2:-}" == "--exports" ]]; then
  MODE="exports"
fi

FILE="$PROFILE_DIR/$PROFILE.env"
if [[ ! -f "$FILE" ]]; then
  echo "Missing profile: $FILE"
  exit 1
fi

if [[ "$MODE" == "exports" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    printf 'export %s=%q\n' "$key" "$val"
  done < "$FILE"
  exit 0
fi

# info mode
python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path(r"$FILE")
rows = []
for raw in p.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    rows.append((k.strip(), v.strip()))
out = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "profile": "$PROFILE",
    "file": str(p),
    "count": len(rows),
    "env": {k: v for k, v in rows},
}
out_path = Path(r"$OUT_LATEST")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
print(f"profile_ready name=$PROFILE vars={len(rows)} file={p}")
print("To apply in shell:")
print(f'eval "$(./scripts/apply_run_profile.sh $PROFILE --exports)"')
PY
