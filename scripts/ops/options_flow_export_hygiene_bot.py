#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.collect_options_flow_context import (
        DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
        DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
        inspect_unusual_whales_export,
        promote_unusual_whales_export,
    )
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from ..collect_options_flow_context import (
        DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
        DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
        inspect_unusual_whales_export,
        promote_unusual_whales_export,
    )
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "options_flow_export_hygiene_latest.json"


def _canonical_export_target(raw_path: str) -> str:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if not str(raw_path or "").strip():
        return ""
    if candidate.is_dir():
        return str(candidate / "latest_options_flow_export.json")
    if candidate.name == "latest_options_flow_export.json":
        return str(candidate)
    return str(candidate.parent / "latest_options_flow_export.json")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    export_path: str | None = None,
    max_age_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS,
    min_stable_seconds: int = DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS,
    apply: bool = False,
) -> dict[str, Any]:
    configured_path = str(export_path if export_path is not None else os.getenv("UNUSUAL_WHALES_EXPORT_PATH", "")).strip()
    _, inspection = inspect_unusual_whales_export(
        configured_path,
        max_age_seconds=max(int(max_age_seconds), 1),
        min_stable_seconds=max(int(min_stable_seconds), 0),
    )
    promotion = {
        "usable": bool(inspection.get("usable", False)),
        "promoted": False,
        "selected_candidate": str(inspection.get("selected_candidate") or ""),
        "promoted_path": "",
        "issues": list(inspection.get("issues") or []),
    }
    if apply and configured_path:
        promotion = promote_unusual_whales_export(
            configured_path,
            promoted_path=_canonical_export_target(configured_path),
            max_age_seconds=max(int(max_age_seconds), 1),
            min_stable_seconds=max(int(min_stable_seconds), 0),
        )

    issues = ordered_unique(
        [str(item) for item in list(inspection.get("issues") or []) + list(promotion.get("issues") or []) if str(item).strip()]
    )
    operator_followups: list[str] = []
    if not configured_path:
        operator_followups.append("set UNUSUAL_WHALES_EXPORT_PATH to a file or inbox directory before expecting export ingestion")
    if configured_path and not bool(inspection.get("usable", False)):
        operator_followups.append(
            "repair or replace the Unusual Whales export because no stable, parseable, fresh candidate is available"
        )
    if int(inspection.get("rejected_row_count", 0) or 0) > 0:
        operator_followups.append("clean malformed JSONL rows so export ingestion stops dropping records")
    if str(inspection.get("schema_status") or "") in {"legacy", "unversioned"}:
        operator_followups.append("upgrade the export drop to the canonical schema so downstream adapters stay predictable")

    overall_status = "blocked"
    if bool(inspection.get("usable", False)):
        overall_status = "ready"
        issue_set = set(issues)
        if bool(promotion.get("promoted", False)):
            issue_set -= {"schema_version_missing"}
        handoff_ready = (
            not bool(apply)
            or bool(promotion.get("promoted", False))
            or str(inspection.get("selected_candidate") or "") == _canonical_export_target(configured_path)
        )
        if issue_set or not handoff_ready:
            overall_status = "degraded"

    payload = {
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "export_path": configured_path,
        "canonical_target_path": _canonical_export_target(configured_path),
        "inspection": inspection,
        "promotion": promotion,
        "metrics": {
            "candidate_count": int(inspection.get("candidate_count", 0) or 0),
            "symbol_count": int(inspection.get("symbol_count", 0) or 0),
            "row_count": int(inspection.get("row_count", 0) or 0),
            "rejected_row_count": int(inspection.get("rejected_row_count", 0) or 0),
            "fresh": bool(inspection.get("fresh", False)),
            "usable": bool(inspection.get("usable", False)),
            "promoted": bool(promotion.get("promoted", False)),
        },
        "issues": issues,
        "operator_followups": operator_followups,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and canonicalize the Unusual Whales export handoff for options-flow ingestion.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--export-path", default=os.getenv("UNUSUAL_WHALES_EXPORT_PATH", ""))
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MAX_AGE_SECONDS))),
    )
    parser.add_argument(
        "--min-stable-seconds",
        type=int,
        default=int(os.getenv("UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS", str(DEFAULT_UNUSUAL_WHALES_EXPORT_MIN_STABLE_SECONDS))),
    )
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    payload = build_payload(
        project_root,
        export_path=str(args.export_path or ""),
        max_age_seconds=int(args.max_age_seconds),
        min_stable_seconds=int(args.min_stable_seconds),
        apply=bool(args.apply),
    )
    out_path = Path(args.out_path).expanduser()
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "options_flow_export_hygiene overall_status={status} usable={usable} promoted={promoted} issues={issues}".format(
                status=str(payload.get("overall_status") or ""),
                usable=str(bool((payload.get("metrics") or {}).get("usable", False))).lower(),
                promoted=str(bool((payload.get("metrics") or {}).get("promoted", False))).lower(),
                issues=len(list(payload.get("issues") or [])),
            )
        )
    return 0 if str(payload.get("overall_status") or "") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
