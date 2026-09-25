#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

from core.accountability import safe_write_json_atomic
from core.execution_lane_pipeline import execution_lane_daily_path
from core.paper_live_equivalence import compare_record_sets


DEFAULT_OUT = Path("governance/health/paper_live_equivalence_latest.json")


def _read_jsonl(path: Path, maximum_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max(maximum_rows, 1))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return list(rows)


def _candidate_roots(project_root: Path) -> list[Path]:
    roots = {
        project_root / "governance" / "execution_lanes",
        project_root / "local_fallback_storage" / "governance" / "execution_lanes",
    }
    configured = str(os.getenv("EXECUTION_LANE_ROOT") or "").strip()
    if configured:
        roots.add(Path(configured).expanduser())
    external_project = str(os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT") or "").strip()
    if external_project:
        external = Path(external_project).expanduser()
    else:
        mount = Path(
            os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")
        ).expanduser()
        project_dir = str(
            os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot")
            or "schwab_trading_bot"
        )
        external = mount / project_dir
    roots.add(external / "governance" / "execution_lanes")
    roots.add(external / "local_fallback_storage" / "governance" / "execution_lanes")
    return sorted(roots, key=str)


def _discover_paths(project_root: Path, stem: str, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    paths: set[Path] = set()
    for root in _candidate_roots(project_root):
        if root.is_dir():
            paths.update(root.glob(f"{stem}_*.jsonl"))
    return sorted(
        paths,
        key=lambda path: (path.stem.rsplit("_", 1)[-1], path.stat().st_mtime_ns),
        reverse=True,
    )


def _date_token(path: Path) -> str:
    token = path.stem.rsplit("_", 1)[-1]
    return token if len(token) == 8 and token.isdigit() else ""


def _read_paths(paths: list[Path], maximum_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining = max(int(maximum_rows), 1)
    for path in paths:
        if remaining <= 0:
            break
        current = _read_jsonl(path, remaining)
        rows.extend(current)
        remaining -= len(current)
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    paper_path: Path | None = None,
    live_path: Path | None = None,
    maximum_rows: int = 5000,
) -> dict[str, Any]:
    paper_paths = _discover_paths(project_root, "execution_intents", paper_path)
    live_paths = _discover_paths(project_root, "execution_promoted", live_path)
    live_dates = {_date_token(path) for path in live_paths if _date_token(path)}
    paper_paths.sort(
        key=lambda path: (
            _date_token(path) in live_dates,
            _date_token(path),
            path.stat().st_mtime_ns,
        ),
        reverse=True,
    )
    paper_rows = _read_paths(paper_paths, maximum_rows)
    live_rows = _read_paths(live_paths, maximum_rows)
    report = compare_record_sets(paper_rows, live_rows)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        **report,
        "paper_path": str(paper_paths[0])
        if paper_paths
        else str(Path(execution_lane_daily_path(project_root, "execution_intents"))),
        "live_shadow_path": str(live_paths[0])
        if live_paths
        else str(Path(execution_lane_daily_path(project_root, "execution_promoted"))),
        "paper_paths": [str(path) for path in paper_paths],
        "live_shadow_paths": [str(path) for path in live_paths],
        "live_orders_enabled_by_this_control": False,
        "readiness_semantics": "every live-shadow intent must match paper semantics; paper intents are not required to promote",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare mode-invariant paper and live-shadow order intents."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--paper-file", default="")
    parser.add_argument("--live-file", default="")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        root,
        paper_path=Path(args.paper_file).expanduser() if args.paper_file else None,
        live_path=Path(args.live_file).expanduser() if args.live_file else None,
        maximum_rows=args.max_rows,
    )
    out = Path(args.out_file).expanduser()
    if not out.is_absolute():
        out = root / out
    safe_write_json_atomic(
        str(out),
        payload,
        project_root=str(root),
        source="paper_live_equivalence_report",
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"paper_live_equivalence status={payload['status']} "
            f"pairs={payload['paired_count']} mismatches={payload['mismatch_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
