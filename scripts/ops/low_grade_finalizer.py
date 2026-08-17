#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import system_needs_intelligence
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, write_payload
else:
    from . import system_needs_intelligence
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "low_grade_finalizer_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.low_grade_finalizer_override"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _runtime_override_text(payload: dict[str, Any]) -> str:
    contract = _as_dict(payload.get("finalization_contract"))
    lines = [
        "# Auto-managed by scripts/ops/low_grade_finalizer.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
        f"LOW_GRADE_FINALIZER_ACTIVE={'1' if contract.get('active') else '0'}",
        f"LOW_GRADE_FINALIZER_EFFECTIVE_GRADE={contract.get('effective_control_posture_grade') or ''}",
        f"LOW_GRADE_FINALIZER_RAW_GRADES_PRESERVED={'1' if contract.get('raw_grades_preserved') else '0'}",
        f"LOW_GRADE_FINALIZER_LAYER_COUNT={payload.get('raw_low_grade_layer_count') or 0}",
    ]
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    audit = system_needs_intelligence._low_grade_layer_audit(project_root)
    layers = [row for row in _as_list(audit.get("layers")) if isinstance(row, dict)]
    finalized_layers = []
    for row in layers:
        finalized_layers.append(
            {
                "exact_file": row.get("exact_file", ""),
                "exact_json_path": row.get("exact_json_path", ""),
                "canonical_json_path": row.get("canonical_json_path", ""),
                "category": row.get("category", "low_grade_layer"),
                "base_grade": row.get("current_grade", ""),
                "effective_grade": row.get("current_grade", ""),
                "active_blocker_after_finalization": bool(row.get("active_blocker", False)),
                "raw_grade_preserved": True,
                "control_state": row.get("control_state", "actionable_low_grade_blocker"),
                "command": row.get("command", []),
            }
        )
    active_blockers = int(audit.get("active_blocker_count", 0) or 0)
    control_grade = str(audit.get("control_posture_grade") or "D")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": active_blockers == 0,
        "overall_status": "ready" if active_blockers == 0 else "needs_action",
        "raw_low_grade_layer_count": len(layers),
        "effective_low_grade_layer_count": int(audit.get("effective_low_grade_layer_count", len(layers)) or 0),
        "active_blocker_count_after_finalization": active_blockers,
        "contained_or_controlled_count_after_finalization": int(audit.get("contained_or_controlled_count", 0) or 0),
        "finalization_contract": {
            "active": True,
            "mode": "truthful_low_grade_classification_v2",
            "effective_control_posture_grade": control_grade,
            "target_effective_grade": "A+",
            "raw_grades_preserved": True,
            "rewrites_raw_evidence": False,
            "cosmetic_grade_uplift_allowed": False,
            "reason": (
                "Low grades are classified by current authority and control state. A repair command or containment "
                "does not upgrade the underlying evidence grade."
            ),
            "stop_condition": "system-needs low_grade_layer_audit.active_blocker_count is 0 and control_posture_grade is A+",
        },
        "finalized_layers": finalized_layers,
        "source_audit": {
            "raw_hit_count": audit.get("raw_hit_count", 0),
            "unique_low_grade_layer_count": audit.get("unique_low_grade_layer_count", 0),
            "active_blocker_count_before_finalization": active_blockers,
            "by_category": audit.get("by_category", {}),
        },
        "protected_volumes": ["/Volumes/VIDEO"],
    }


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path | None = None,
    override_path: Path | None = None,
) -> dict[str, Any]:
    out = out_path or (project_root / "governance" / "health" / "low_grade_finalizer_latest.json")
    override = override_path or (project_root / "config" / ".env.low_grade_finalizer_override")
    out = out if out.is_absolute() else project_root / out
    override = override if override.is_absolute() else project_root / override
    write_payload(out, payload)
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(_runtime_override_text(payload), encoding="utf-8")
    applied = dict(payload)
    applied["apply_result"] = {
        "applied": True,
        "out_path": str(out),
        "override_path": str(override),
    }
    write_payload(out, applied)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize low-grade audit surfaces into an effective A+ control posture without rewriting raw evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--override-file", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.project_root)
    if args.apply:
        payload = apply_payload(args.project_root, payload, out_path=args.out_file, override_path=args.override_file)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "low_grade_finalizer "
            f"status={payload.get('overall_status')} "
            f"raw_layers={payload.get('raw_low_grade_layer_count')} "
            f"effective_low={payload.get('effective_low_grade_layer_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
