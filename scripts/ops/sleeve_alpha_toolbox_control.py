#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.alpha_concept_engine import MEASUREMENT_FUNCTIONS
    from core.institutional_decision_flow import (
        QUANTITATIVE_EVIDENCE_AXES,
        load_policy,
        resolve_sleeve_policy,
    )
    from scripts.ops.long_runtime_common import load_json, write_payload
else:
    from core.alpha_concept_engine import MEASUREMENT_FUNCTIONS
    from core.institutional_decision_flow import (
        QUANTITATIVE_EVIDENCE_AXES,
        load_policy,
        resolve_sleeve_policy,
    )
    from .long_runtime_common import PROJECT_ROOT, load_json, write_payload


DEFAULT_CONFIG_PATH = Path("config/sleeve_alpha_toolbox_v1.json")
DEFAULT_SLEEVE_PATH = Path("config/sleeve_strategy_expansion.json")
DEFAULT_DECISION_POLICY_PATH = Path("config/institutional_decision_flow_v1.json")
DEFAULT_ALPHA_REPORT_PATH = Path("governance/research/alpha_concept_report_latest.json")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path("governance/research/sleeve_alpha_toolbox_latest.json")
DEFAULT_MARKDOWN_PATH = Path("exports/reports/operator/sleeve_alpha_toolbox_latest.md")
FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "changes_signal",
        "changes_action",
        "changes_threshold",
        "changes_position_size",
        "submits_paper_order",
        "submits_live_order",
        "promotes_strategy",
        "accepts_candidate_change",
        "allocates_capital",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {"path": str(path), "exists": False, "sha256": "", "bytes": 0}
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not payload:
        raise ValueError(f"alpha toolbox config is missing or invalid: {path}")
    axis_routes = _mapping(payload.get("axis_routes"))
    if set(axis_routes) != set(QUANTITATIVE_EVIDENCE_AXES):
        raise ValueError(
            "alpha toolbox must route every quantitative evidence axis exactly"
        )
    minimum_tools = max(
        int(_number(payload.get("minimum_tools_per_required_axis"), 2)), 1
    )
    known_engines = set(MEASUREMENT_FUNCTIONS)
    for axis, raw_route in axis_routes.items():
        route = _mapping(raw_route)
        tools = [str(value) for value in route.get("tools") or [] if str(value)]
        if len(set(tools)) < minimum_tools:
            raise ValueError(
                f"alpha toolbox axis has too few independent tools: {axis}"
            )
        unknown = sorted(set(tools) - known_engines)
        if unknown:
            raise ValueError(
                f"alpha toolbox axis references unknown tools: {axis}:{','.join(unknown)}"
            )
        if not str(route.get("collection_priority") or "").strip():
            raise ValueError(
                f"alpha toolbox axis is missing collection priority: {axis}"
            )
    foundation = {
        str(value)
        for value in payload.get("universal_foundation_tools") or []
        if str(value)
    }
    unknown_foundation = sorted(foundation - known_engines)
    if unknown_foundation:
        raise ValueError(
            "alpha toolbox foundation references unknown tools: "
            + ",".join(unknown_foundation)
        )
    authority = _mapping(payload.get("authority"))
    if set(authority) != set(FORBIDDEN_AUTHORITY_KEYS) or any(
        bool(authority.get(key, False)) for key in FORBIDDEN_AUTHORITY_KEYS
    ):
        raise ValueError(
            "alpha toolbox requests forbidden trading or promotion authority"
        )
    evidence_contract = _mapping(payload.get("evidence_contract"))
    required_true = {
        "current_candidate_binding_required",
        "post_cost_outcomes_required_for_economic_support",
    }
    required_false = {
        "missing_evidence_is_zero",
        "historical_generation_evidence_grades_current_candidate",
        "legacy_window_association_is_promotion_grade",
        "tool_implementation_is_alpha_evidence",
        "passing_one_tool_waives_other_required_axes",
    }
    if not all(bool(evidence_contract.get(key, False)) for key in required_true) or any(
        bool(evidence_contract.get(key, False)) for key in required_false
    ):
        raise ValueError("alpha toolbox evidence contract is unsafe")
    return payload


def _measurement_state(
    engine_id: str,
    measurements: Mapping[str, Any],
    *,
    candidate_bound: bool,
) -> dict[str, Any]:
    row = _mapping(measurements.get(engine_id))
    implemented = engine_id in MEASUREMENT_FUNCTIONS
    available = bool(candidate_bound and row.get("available", False))
    passing = bool(candidate_bound and row.get("passes", False))
    return {
        "engine_id": engine_id,
        "implemented": implemented,
        "candidate_bound_available": available,
        "candidate_bound_passing": passing,
        "measurement_status": str(row.get("status") or "not_measured"),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    toolbox_path = config_path or (root / DEFAULT_CONFIG_PATH)
    if not toolbox_path.is_absolute():
        toolbox_path = root / toolbox_path
    sleeve_path = root / DEFAULT_SLEEVE_PATH
    decision_policy_path = root / DEFAULT_DECISION_POLICY_PATH
    alpha_report_path = root / DEFAULT_ALPHA_REPORT_PATH
    candidate_path = root / DEFAULT_CANDIDATE_PATH
    toolbox = _load_config(toolbox_path)
    sleeves_payload = load_json(sleeve_path)
    sleeves = _rows(sleeves_payload.get("sleeves"))
    if not sleeves:
        raise ValueError("sleeve strategy expansion has no sleeves")
    decision_policy = load_policy(decision_policy_path)
    alpha_report = load_json(alpha_report_path)
    candidate = load_json(candidate_path)
    candidate_id = str(candidate.get("candidate_id") or "")
    candidate_generation = int(_number(candidate.get("generation"), 0))
    report_binding = _mapping(alpha_report.get("candidate_binding"))
    report_candidate_bound = bool(
        candidate_id
        and report_binding.get("bound", False)
        and str(report_binding.get("candidate_id") or "") == candidate_id
        and int(_number(report_binding.get("candidate_generation"), 0))
        == candidate_generation
    )
    measurements = _mapping(alpha_report.get("measurements"))
    axis_routes = _mapping(toolbox.get("axis_routes"))
    foundation_ids = [
        str(value)
        for value in toolbox.get("universal_foundation_tools") or []
        if str(value)
    ]
    foundation = [
        _measurement_state(
            engine_id,
            measurements,
            candidate_bound=report_candidate_bound,
        )
        for engine_id in foundation_ids
    ]
    family_counts: Counter[str] = Counter()
    match_source_counts: Counter[str] = Counter()
    axis_required_counts: Counter[str] = Counter()
    axis_ready_counts: Counter[str] = Counter()
    engine_route_counts: Counter[str] = Counter()
    route_rows: list[dict[str, Any]] = []
    for sleeve in sorted(sleeves, key=lambda row: str(row.get("name") or "")):
        profile = str(sleeve.get("name") or "").strip().lower()
        lifecycle = str(sleeve.get("runtime_status") or "").strip().lower()
        if not profile:
            continue
        _resolved, receipt = resolve_sleeve_policy(
            profile,
            decision_policy,
            lifecycle_state=lifecycle,
        )
        family_id = str(receipt.get("policy_family_id") or "")
        match_source = str(receipt.get("match_source") or "")
        required_axes = [
            str(value)
            for value in receipt.get("required_quantitative_evidence") or []
            if str(value)
        ]
        axis_rows: list[dict[str, Any]] = []
        routed_engines: set[str] = set(foundation_ids)
        for axis in required_axes:
            route = _mapping(axis_routes.get(axis))
            tool_states = [
                _measurement_state(
                    str(engine_id),
                    measurements,
                    candidate_bound=report_candidate_bound,
                )
                for engine_id in route.get("tools") or []
            ]
            passing_count = sum(row["candidate_bound_passing"] for row in tool_states)
            available_count = sum(
                row["candidate_bound_available"] for row in tool_states
            )
            axis_ready = passing_count > 0
            axis_required_counts[axis] += 1
            axis_ready_counts[axis] += int(axis_ready)
            for state in tool_states:
                routed_engines.add(str(state["engine_id"]))
                engine_route_counts[str(state["engine_id"])] += 1
            axis_rows.append(
                {
                    "axis": axis,
                    "candidate_evidence_ready": axis_ready,
                    "candidate_bound_available_tool_count": available_count,
                    "candidate_bound_passing_tool_count": passing_count,
                    "collection_priority": str(route.get("collection_priority") or ""),
                    "tools": tool_states,
                }
            )
        candidate_evidence_ready = bool(required_axes) and all(
            row["candidate_evidence_ready"] for row in axis_rows
        )
        family_counts[family_id] += 1
        match_source_counts[match_source] += 1
        route_material = {
            "candidate_id": candidate_id,
            "profile": profile,
            "policy_family_id": family_id,
            "strategy_variant_id": str(receipt.get("strategy_variant_id") or ""),
            "required_axes": required_axes,
            "routed_engines": sorted(routed_engines),
        }
        route_rows.append(
            {
                "sleeve": profile,
                "runtime_status": lifecycle,
                "strategy_count": len(sleeve.get("strategies") or []),
                "strategies": [str(value) for value in sleeve.get("strategies") or []],
                "policy_family_id": family_id,
                "policy_match_source": match_source,
                "execution_eligible_by_policy": bool(
                    receipt.get("execution_eligible", False)
                ),
                "strategy_variant_id": str(receipt.get("strategy_variant_id") or ""),
                "required_quantitative_evidence": required_axes,
                "required_axis_count": len(required_axes),
                "candidate_evidence_ready_axis_count": sum(
                    row["candidate_evidence_ready"] for row in axis_rows
                ),
                "candidate_evidence_ready": candidate_evidence_ready,
                "routed_measurement_engines": sorted(routed_engines),
                "axis_routes": axis_rows,
                "route_receipt_sha256": _canonical_hash(route_material),
            }
        )
    missing_sleeves = len(sleeves) - len(route_rows)
    candidate_ready_sleeves = sum(row["candidate_evidence_ready"] for row in route_rows)
    structural_ready = bool(
        route_rows
        and not missing_sleeves
        and all(row["required_axis_count"] > 0 for row in route_rows)
    )
    evidence_ready = bool(
        structural_ready
        and report_candidate_bound
        and candidate_ready_sleeves == len(route_rows)
    )
    blockers: list[str] = []
    if not structural_ready:
        blockers.append("sleeve_alpha_routes_incomplete")
    if not report_candidate_bound:
        blockers.append("alpha_measurement_report_not_bound_to_current_candidate")
    if candidate_ready_sleeves < len(route_rows):
        blockers.append("required_axis_candidate_evidence_collecting")
    now = generated_at_utc or datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "timestamp_utc": now,
        "schema_version": 1,
        "toolbox_id": str(toolbox.get("toolbox_id") or ""),
        "ok": structural_ready,
        "overall_status": (
            "candidate_evidence_ready"
            if evidence_ready
            else (
                "structurally_ready_collecting_candidate_evidence"
                if structural_ready
                else "blocked"
            )
        ),
        "candidate_binding": {
            "candidate_id": candidate_id,
            "candidate_generation": candidate_generation,
            "alpha_report_candidate_id": str(report_binding.get("candidate_id") or ""),
            "alpha_report_candidate_generation": int(
                _number(report_binding.get("candidate_generation"), 0)
            ),
            "alpha_report_bound_to_current_candidate": report_candidate_bound,
        },
        "coverage": {
            "declared_sleeve_count": len(sleeves),
            "routed_sleeve_count": len(route_rows),
            "missing_sleeve_count": missing_sleeves,
            "candidate_evidence_ready_sleeve_count": candidate_ready_sleeves,
            "structural_route_coverage_ratio": round(len(route_rows) / len(sleeves), 8),
            "candidate_evidence_ready_sleeve_ratio": (
                round(candidate_ready_sleeves / len(route_rows), 8)
                if route_rows
                else 0.0
            ),
            "policy_family_counts": dict(sorted(family_counts.items())),
            "policy_match_source_counts": dict(sorted(match_source_counts.items())),
        },
        "axis_collection_state": [
            {
                "axis": axis,
                "required_sleeve_count": axis_required_counts[axis],
                "candidate_evidence_ready_sleeve_count": axis_ready_counts[axis],
                "candidate_evidence_ready_ratio": (
                    round(axis_ready_counts[axis] / axis_required_counts[axis], 8)
                    if axis_required_counts[axis]
                    else 0.0
                ),
                "collection_priority": str(
                    _mapping(axis_routes.get(axis)).get("collection_priority") or ""
                ),
            }
            for axis in sorted(QUANTITATIVE_EVIDENCE_AXES)
        ],
        "engine_route_counts": dict(sorted(engine_route_counts.items())),
        "universal_foundation_tools": foundation,
        "sleeve_routes": route_rows,
        "blockers": blockers,
        "source_receipts": {
            "toolbox_config": _file_receipt(toolbox_path),
            "sleeve_registry": _file_receipt(sleeve_path),
            "decision_policy": _file_receipt(decision_policy_path),
            "alpha_report": _file_receipt(alpha_report_path),
            "candidate_state": _file_receipt(candidate_path),
        },
        "evidence_contract": _mapping(toolbox.get("evidence_contract")),
        "authority": _mapping(toolbox.get("authority")),
        "interpretation": {
            "tool_coverage_proves_alpha": False,
            "candidate_evidence_required_for_economic_claim": True,
            "post_cost_outcomes_required_for_profitability_claim": True,
            "missing_evidence_is_zero": False,
            "profitability_guaranteed": False,
            "routes_are_research_and_collection_instructions_only": True,
        },
    }
    payload["toolbox_receipt_sha256"] = _canonical_hash(payload)
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    coverage = _mapping(payload.get("coverage"))
    binding = _mapping(payload.get("candidate_binding"))
    lines = [
        "# Sleeve Alpha Toolbox",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Status: `{payload.get('overall_status', '')}`",
        f"Candidate: `{binding.get('candidate_id', '')}` (G{binding.get('candidate_generation', 0)})",
        "",
        "## Coverage",
        "",
        f"- Routed sleeves: `{coverage.get('routed_sleeve_count', 0)}/{coverage.get('declared_sleeve_count', 0)}`",
        f"- Candidate-evidence-ready sleeves: `{coverage.get('candidate_evidence_ready_sleeve_count', 0)}`",
        f"- Alpha report bound to current candidate: `{binding.get('alpha_report_bound_to_current_candidate', False)}`",
        "",
        "## Required Evidence Axes",
        "",
        "| Axis | Sleeves | Ready | Collection priority |",
        "|---|---:|---:|---|",
    ]
    for row in payload.get("axis_collection_state") or []:
        row = _mapping(row)
        lines.append(
            f"| `{row.get('axis', '')}` | {row.get('required_sleeve_count', 0)} | "
            f"{row.get('candidate_evidence_ready_sleeve_count', 0)} | {row.get('collection_priority', '')} |"
        )
    lines.extend(
        [
            "",
            "## Sleeve Routes",
            "",
            "| Sleeve | Family | Axes | Engines | Evidence ready |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload.get("sleeve_routes") or []:
        row = _mapping(row)
        lines.append(
            f"| `{row.get('sleeve', '')}` | `{row.get('policy_family_id', '')}` | "
            f"{row.get('required_axis_count', 0)} | {len(row.get('routed_measurement_engines') or [])} | "
            f"`{row.get('candidate_evidence_ready', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This control routes diagnostics and evidence collection only. It cannot change signals, thresholds, actions, sizing, paper orders, strategy promotion, capital allocation, candidate acceptance, or live execution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route fail-closed alpha diagnostics to every declared sleeve."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    out_path = Path(args.out_file).expanduser()
    markdown_path = Path(args.markdown_file).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    if not out_path.is_absolute():
        out_path = root / out_path
    if not markdown_path.is_absolute():
        markdown_path = root / markdown_path
    payload = build_payload(root, config_path=config_path)
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        coverage = _mapping(payload.get("coverage"))
        print(
            "sleeve_alpha_toolbox "
            f"status={payload.get('overall_status', '')} "
            f"routed={coverage.get('routed_sleeve_count', 0)}/"
            f"{coverage.get('declared_sleeve_count', 0)} "
            f"evidence_ready={coverage.get('candidate_evidence_ready_sleeve_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
