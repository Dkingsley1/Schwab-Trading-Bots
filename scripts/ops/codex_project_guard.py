#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "codex_project_guard_latest.json"
REQUIRED_MARKERS = {
    "AGENTS.md": ["Source Of Truth", "Scope Discipline", "Current Separate Domains"],
    "README.md": ["Sortino ratio", "Sharpe ratio", "signal_generation_*.jsonl", "SOURCE_OF_TRUTH.md"],
    "docs/architecture/SOURCE_OF_TRUTH.md": ["Operator commands", "Decision and signal evidence", "Codex project guardrails"],
    "docs/architecture/ADR-0001-system-source-of-truth.md": ["System Source Of Truth", "Signal Evidence"],
}
SEPARATE_DOMAIN_PATTERNS = ["Logic Pro", "creative-audio", "96 kHz", "96khz", "96000", "sample rate", "standalone app"]
DOC_BOUNDARY_PATHS = ["README.md", "docs/architecture/SOURCE_OF_TRUTH.md", "docs/architecture/ADR-0001-system-source-of-truth.md"]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _marker_row(rel_path: str, markers: list[str]) -> dict:
    path = PROJECT_ROOT / rel_path
    text = _read(path)
    missing = [marker for marker in markers if marker not in text]
    status = "ready" if path.exists() and not missing else "blocked"
    return {
        "name": rel_path,
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "missing_markers": missing,
    }


def build_payload() -> dict:
    rows = [_marker_row(path, markers) for path, markers in REQUIRED_MARKERS.items()]
    hits = []
    for rel_path in DOC_BOUNDARY_PATHS:
        text = _read(PROJECT_ROOT / rel_path).lower()
        for pattern in SEPARATE_DOMAIN_PATTERNS:
            if pattern.lower() in text:
                hits.append({"path": rel_path, "pattern": pattern})
    rows.append({
        "name": "separate_domain_doc_boundary",
        "status": "blocked" if hits else "ready",
        "ok": not hits,
        "hits": hits,
    })
    blocked = [row for row in rows if row.get("status") == "blocked"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": not blocked,
        "overall_status": "blocked" if blocked else "ready",
        "guards": rows,
        "metrics": {"guard_count": len(rows), "blocked_guard_count": len(blocked)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard README/source-of-truth docs against AI-assisted project drift.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()
    payload = build_payload()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"codex_project_guard overall_status={payload['overall_status']} blocked={payload['metrics']['blocked_guard_count']}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
