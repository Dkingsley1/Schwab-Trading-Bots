#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "codex_project_guard_latest.json"

REQUIRED_AGENT_MARKERS = [
    "Source Of Truth",
    "Scope Discipline",
    "Current Separate Domains",
    "Regression Guardrails",
    "per-surface retry budgets",
    "codex-project-guard",
]
REQUIRED_SOURCE_TRUTH_MARKERS = [
    "Operator commands",
    "Report opening and PDF fallbacks",
    "Schwab auth handshake",
    "Sleeve performance metrics",
    "Decision and signal evidence",
    "Storage routing",
]
REQUIRED_README_MARKERS = [
    "docs/architecture/SOURCE_OF_TRUTH.md",
    "docs/architecture/ADR-0001-system-source-of-truth.md",
    "Sortino ratio",
    "Sharpe ratio",
    "signal_generation_*.jsonl",
]
SEPARATE_DOMAIN_PATTERNS = [
    "Logic Pro",
    "creative-audio",
    "96 kHz",
    "96khz",
    "96000",
    "sample rate",
    "standalone app",
]
SEPARATE_DOMAIN_PATH_HINTS = [
    "apple_silicon_profile.py",
    "test_apple_silicon_profile.py",
    "mlx_audio_runtime_audit.py",
    "test_mlx_audio_runtime_audit.py",
    "creative_cotenant_guard.py",
    "test_creative_cotenant_guard.py",
]
TRADING_SYSTEM_DOC_PATHS = {
    "README.md",
    "docs/architecture/SOURCE_OF_TRUTH.md",
    "docs/architecture/ADR-0001-system-source-of-truth.md",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _marker_check(project_root: Path, rel_path: str, markers: list[str]) -> dict[str, Any]:
    path = project_root / rel_path
    text = _read_text(path)
    missing = [marker for marker in markers if marker not in text]
    status = "ready" if path.exists() and not missing else "blocked"
    detail = "markers_present" if status == "ready" else f"missing_markers={missing or ['file_missing']}"
    return {
        "name": rel_path,
        "family": "codex_contract",
        "path": str(path),
        "exists": path.exists(),
        "status": status,
        "ok": status == "ready",
        "detail": detail,
        "missing_markers": missing,
    }


def _separate_domain_doc_check(project_root: Path, rel_paths: list[str], allow_separate_domain: bool) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    for rel_path in rel_paths:
        text = _read_text(project_root / rel_path)
        haystack = text.lower()
        for pattern in SEPARATE_DOMAIN_PATTERNS:
            if pattern.lower() in haystack:
                hits.append({"path": rel_path, "pattern": pattern})
    status = "ready" if allow_separate_domain or not hits else "blocked"
    return {
        "name": "separate_domain_doc_boundary",
        "family": "codex_contract",
        "path": ",".join(rel_paths),
        "exists": True,
        "status": status,
        "ok": status == "ready",
        "detail": "separate_domain_absent" if not hits else f"separate_domain_hits={hits}",
        "hits": hits,
    }


def _git_staged_paths(project_root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _staged_scope_check(staged_paths: list[str], allow_separate_domain: bool) -> dict[str, Any]:
    doc_hits = sorted(path for path in staged_paths if path in TRADING_SYSTEM_DOC_PATHS)
    separate_hits = sorted(
        path
        for path in staged_paths
        if any(hint in path for hint in SEPARATE_DOMAIN_PATH_HINTS)
    )
    mixed = bool(doc_hits and separate_hits)
    status = "ready" if allow_separate_domain or not mixed else "blocked"
    return {
        "name": "staged_scope_boundary",
        "family": "codex_contract",
        "path": "git_index",
        "exists": True,
        "status": status,
        "ok": status == "ready",
        "detail": (
            "staged_scope_clean"
            if status == "ready"
            else f"trading_docs={doc_hits} separate_domain_files={separate_hits}"
        ),
        "staged_path_count": len(staged_paths),
        "trading_doc_paths": doc_hits,
        "separate_domain_paths": separate_hits,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    staged_paths: list[str] | None = None,
    include_staged: bool = False,
    allow_separate_domain: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    rows = [
        _marker_check(project_root, "AGENTS.md", REQUIRED_AGENT_MARKERS),
        _marker_check(project_root, "docs/architecture/SOURCE_OF_TRUTH.md", REQUIRED_SOURCE_TRUTH_MARKERS),
        _marker_check(project_root, "README.md", REQUIRED_README_MARKERS),
        _marker_check(project_root, "docs/architecture/ADR-0001-system-source-of-truth.md", ["System Source Of Truth", "Signal Evidence"]),
        _separate_domain_doc_check(
            project_root,
            [
                "README.md",
                "docs/architecture/SOURCE_OF_TRUTH.md",
                "docs/architecture/ADR-0001-system-source-of-truth.md",
            ],
            allow_separate_domain,
        ),
    ]
    if include_staged:
        rows.append(_staged_scope_check(staged_paths if staged_paths is not None else _git_staged_paths(project_root), allow_separate_domain))

    blocked = [row for row in rows if str(row.get("status") or "") == "blocked"]
    degraded = [row for row in rows if str(row.get("status") or "") in {"degraded", "warning", "warn"}]
    overall_status = "blocked" if blocked else "degraded" if degraded else "ready"
    recommended_actions = ordered_unique(
        [
            "read `AGENTS.md` and `docs/architecture/SOURCE_OF_TRUTH.md` before continuing"
            if any(row.get("name") == "AGENTS.md" for row in blocked)
            else "",
            "keep Logic/audio/96kHz work out of the Schwab README/source-of-truth publish scope unless explicitly requested"
            if any(row.get("name") == "separate_domain_doc_boundary" for row in blocked)
            else "",
            "stage explicit, scoped paths only; split separate-domain work into a separate commit/PR"
            if any(row.get("name") == "staged_scope_boundary" for row in blocked)
            else "",
        ]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "guards": rows,
        "metrics": {
            "guard_count": len(rows),
            "blocked_guard_count": len(blocked),
            "degraded_guard_count": len(degraded),
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard Codex-authored changes against project scope and source-of-truth drift.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file")
    parser.add_argument("--staged", action="store_true", help="Check the Git index for mixed-domain staged changes.")
    parser.add_argument("--allow-separate-domain", action="store_true", help="Allow separate-domain Logic/audio content when explicitly requested.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        include_staged=bool(args.staged),
        allow_separate_domain=bool(args.allow_separate_domain),
    )
    out_file = Path(args.out_file).expanduser() if args.out_file else project_root / "governance" / "health" / "codex_project_guard_latest.json"
    write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "codex_project_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"blocked={int((payload.get('metrics') or {}).get('blocked_guard_count', 0) or 0)}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
