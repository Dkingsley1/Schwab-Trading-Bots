#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from scripts.ops import production_flow_smoke, source_mutation_guard
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
    from . import production_flow_smoke, source_mutation_guard


SCHEMA_VERSION = 1
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_level_upgrade_hardener_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "production_level_upgrade_hardener_control_latest.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "exports" / "reports" / "operator" / "production_level_upgrade_hardener_control_latest.md"

READY_STATUSES = {"ready", "ok", "active", "guarded", "ready_guarded", "present", "protective_tightening", "advisory"}
BAD_STATUSES = {"blocked", "critical", "failed", "error", "not_ready"}
EXPECTED_GROUP_COUNTS = {"production_upgrade": 10, "hardener": 10}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "on", "ready", "ok", "active"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _extract_status(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status", "state"):
        status = str(payload.get(key) or "").strip().lower()
        if status:
            return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "present" if payload else "missing"


def _path_value(payload: Any, dotted: str, default: Any = None) -> Any:
    current = payload
    for part in str(dotted or "").split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return default
    return current


def _path_exists(payload: Any, dotted: str) -> bool:
    sentinel = object()
    return _path_value(payload, dotted, sentinel) is not sentinel


def _non_empty(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, (str, list, tuple, set, dict)):
        return bool(raw)
    return True


def _artifact_row(project_root: Path, requirement: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    raw_path = str(requirement.get("path") or "")
    path = _project_path(project_root, raw_path)
    exists = path.exists()
    parse_json = bool(requirement.get("parse_json", True))
    payload = load_json(path) if parse_json else {}
    status = _extract_status(payload) if payload else ("present" if exists else "missing")
    age_minutes = payload_age_minutes(payload, path, now=now) if exists else None
    max_age_hours = _safe_float(requirement.get("max_age_hours"), 0.0)
    fresh = bool(max_age_hours <= 0.0 or (age_minutes is not None and age_minutes <= max_age_hours * 60.0))

    ready_statuses = set(_string_list(requirement.get("ready_statuses")) or READY_STATUSES)
    existence_only = bool(requirement.get("existence_only", False))
    status_ok = bool(exists) if existence_only else bool(status in ready_statuses or (payload.get("ok") is True and "ok" in ready_statuses))

    selected_values: dict[str, Any] = {}
    blockers: list[str] = []
    if not exists:
        blockers.append("artifact_missing")
    if exists and not status_ok:
        blockers.append(f"status_not_ready:{status}")
    if exists and not fresh:
        blockers.append("artifact_stale")

    for dotted in _string_list(requirement.get("required_paths")):
        selected_values[dotted] = _path_value(payload, dotted)
        if parse_json and not _path_exists(payload, dotted):
            blockers.append(f"required_path_missing:{dotted}")

    for dotted in _string_list(requirement.get("truthy_paths")):
        value = _path_value(payload, dotted)
        selected_values[dotted] = value
        if not _bool(value):
            blockers.append(f"truthy_path_failed:{dotted}")

    for dotted in _string_list(requirement.get("falsey_paths")):
        value = _path_value(payload, dotted)
        selected_values[dotted] = value
        if _bool(value):
            blockers.append(f"falsey_path_failed:{dotted}")

    for dotted in _string_list(requirement.get("zero_count_paths")):
        value = _path_value(payload, dotted)
        selected_values[dotted] = value
        ok = len(value) == 0 if isinstance(value, list) else _safe_float(value, 999999.0) == 0.0
        if not ok:
            blockers.append(f"zero_count_failed:{dotted}")

    max_values = requirement.get("max_value_by_path") if isinstance(requirement.get("max_value_by_path"), dict) else {}
    for dotted, ceiling in max_values.items():
        value = _path_value(payload, str(dotted))
        selected_values[str(dotted)] = value
        if _safe_float(value, 999999.0) > _safe_float(ceiling):
            blockers.append(f"max_value_failed:{dotted}")

    min_values = requirement.get("min_value_by_path") if isinstance(requirement.get("min_value_by_path"), dict) else {}
    for dotted, floor in min_values.items():
        value = _path_value(payload, str(dotted))
        selected_values[str(dotted)] = value
        if _safe_float(value, -999999.0) < _safe_float(floor):
            blockers.append(f"min_value_failed:{dotted}")

    for dotted in _string_list(requirement.get("non_empty_paths")):
        value = _path_value(payload, dotted)
        selected_values[dotted] = value
        if not _non_empty(value):
            blockers.append(f"non_empty_path_failed:{dotted}")

    blockers = ordered_unique(blockers)
    return {
        "path": str(path),
        "exists": exists,
        "parse_json": parse_json,
        "status": status,
        "ready": not blockers,
        "blockers": blockers,
        "age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
        "max_age_minutes": round(max_age_hours * 60.0, 4) if max_age_hours > 0.0 else 0.0,
        "fresh": fresh,
        "summary_keys": sorted(payload.keys())[:24] if payload else [],
        "selected_values": selected_values,
    }


def _dynamic_row(project_root: Path, name: str) -> dict[str, Any]:
    if name == "source_mutation_guard":
        payload = source_mutation_guard.build_payload(project_root)
    elif name == "production_flow_smoke":
        payload = production_flow_smoke.build_payload(project_root)
    else:
        return {
            "name": name,
            "dynamic": True,
            "status": "blocked",
            "ready": False,
            "blockers": [f"unknown_dynamic_check:{name}"],
            "evidence": {},
        }

    status = _extract_status(payload)
    ready = bool(payload.get("ok", False)) and status not in BAD_STATUSES
    return {
        "name": name,
        "dynamic": True,
        "status": status,
        "ready": ready,
        "blockers": _string_list(payload.get("failed_checks")) + _string_list(payload.get("dirty_entries")) + _string_list(payload.get("error")),
        "evidence": {
            "ok": payload.get("ok"),
            "overall_status": status,
            "project_root": payload.get("project_root"),
            "check": payload.get("check"),
        },
    }


def _first_present(payload: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _path_value(payload, path)
        if value not in (None, ""):
            return value
    return None


def _raw_profitability_truth(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "paper_profitability_control_latest.json"
    payload = load_json(path)
    raw_grade = str(_first_present(payload, ("raw_profitability_grade", "financial_profitability_grade", "base_raw_operational_outcome_grade")) or "")
    controlled_grade = str(_first_present(payload, ("controlled_profitability_grade", "controlled_financial_grade", "operational_control_grade")) or "")
    display = str(_first_present(payload, ("profitability_display_grade", "financial_display_grade")) or "")
    ladder = _as_dict(payload.get("raw_d_recovery_ladder_contract"))
    transparency = _as_dict(payload.get("grade_transparency_contract"))
    raw_evidence_based = bool(
        payload
        and raw_grade
        and controlled_grade
        and _bool(ladder.get("raw_grade_remains_evidence_based"))
        and _bool(transparency.get("no_live_trade_authority"))
        and (raw_grade == controlled_grade or raw_grade in display or str(ladder.get("current_raw_profitability_grade") or "") == raw_grade)
    )
    return {
        "name": "raw_profitability_truth",
        "status": "ready" if raw_evidence_based else "blocked",
        "ready": raw_evidence_based,
        "blockers": [] if raw_evidence_based else ["raw_profitability_truth_not_preserved"],
        "evidence": {
            "path": str(path),
            "raw_profitability_grade": raw_grade,
            "controlled_profitability_grade": controlled_grade,
            "profitability_display_grade": display,
            "raw_grade_remains_evidence_based": ladder.get("raw_grade_remains_evidence_based"),
            "no_live_trade_authority": transparency.get("no_live_trade_authority"),
        },
    }


def _live_execution_double_lock(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "production_readiness_control_latest.json"
    payload = load_json(path)
    domains = [row for row in _as_list(payload.get("domains")) if isinstance(row, dict)]
    firewall = next((row for row in domains if str(row.get("name") or "") == "live_execution_risk_firewall"), {})
    blockers = set(_string_list(firewall.get("blockers")))
    evidence = _as_dict(firewall.get("evidence"))
    ready = bool(
        payload.get("ok")
        and str(payload.get("overall_status") or "").lower() in {"ready", "guarded"}
        and str(firewall.get("status") or "").lower() == "ready_guarded"
        and blockers <= {"live_execution_not_armed", "market_data_only_active"}
        and not _bool(evidence.get("live_order_allowed"))
        and not _bool(evidence.get("execution_armed"))
        and _bool(evidence.get("market_data_only"))
    )
    return {
        "name": "live_execution_double_lock",
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "blockers": [] if ready else ["live_execution_double_lock_not_guarded"],
        "evidence": {
            "path": str(path),
            "production_status": payload.get("overall_status"),
            "firewall_status": firewall.get("status"),
            "firewall_blockers": sorted(blockers),
            "execution_armed": evidence.get("execution_armed"),
            "market_data_only": evidence.get("market_data_only"),
            "live_order_allowed": evidence.get("live_order_allowed"),
        },
    }


def _storage_soft_quota(project_root: Path) -> dict[str, Any]:
    quota_path = project_root / "governance" / "health" / "storage_quota_guard_latest.json"
    ingestion_path = project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    quota = load_json(quota_path)
    ingestion = load_json(ingestion_path)
    hard_breaches = _safe_float(_path_value(quota, "quota_summary.hard_breaches"), 999999.0)
    soft_breaches = _safe_float(_path_value(quota, "quota_summary.soft_breaches"), 0.0)
    hot_path_green = _bool(_path_value(quota, "active_hot_buffer_containment.hot_path_green"))
    ingestion_ready = str(ingestion.get("overall_status") or "").lower() == "ready" and bool(ingestion.get("ok"))
    ready = bool(hard_breaches == 0.0 and hot_path_green and ingestion_ready)
    return {
        "name": "storage_soft_quota_escalator",
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "blockers": [] if ready else ["storage_soft_quota_hard_or_ingestion_blocker"],
        "evidence": {
            "quota_path": str(quota_path),
            "ingestion_path": str(ingestion_path),
            "quota_status": quota.get("overall_status"),
            "ingestion_status": ingestion.get("overall_status"),
            "hard_breaches": hard_breaches,
            "soft_breaches": soft_breaches,
            "hot_path_green": hot_path_green,
            "managed_degraded_visible": str(quota.get("overall_status") or "").lower() == "degraded" and hard_breaches == 0.0,
        },
    }


def _no_fake_green_dashboard(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json"
    payload = load_json(path)
    overall = _as_dict(payload.get("overall"))
    managed_attention = _as_list(overall.get("managed_attention"))
    managed_controls = _as_list(overall.get("managed_controls"))
    ready = bool(
        overall.get("ok")
        and _path_exists(payload, "overall.raw_attention")
        and _path_exists(payload, "overall.forensic_attention")
        and _path_exists(payload, "overall.managed_attention")
        and (not managed_attention or managed_controls)
    )
    return {
        "name": "no_fake_green_dashboard",
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "blockers": [] if ready else ["dashboard_attention_transparency_missing"],
        "evidence": {
            "path": str(path),
            "overall_ok": overall.get("ok"),
            "raw_attention_count": len(_as_list(overall.get("raw_attention"))),
            "forensic_attention_count": len(_as_list(overall.get("forensic_attention"))),
            "managed_attention_count": len(managed_attention),
            "managed_control_count": len(managed_controls),
        },
    }


def _kill_switch_audit(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "global_killswitch_latest.json"
    payload = load_json(path)
    safe_clear = _as_dict(payload.get("safe_clear"))
    ready = bool(
        path.exists()
        and _bool(payload.get("clear_ready"))
        and _bool(safe_clear.get("ready"))
        and not _bool(payload.get("halt"))
        and not _bool(payload.get("halt_latched"))
        and not _bool(payload.get("halt_required"))
        and not _bool(payload.get("operator_stop"))
    )
    return {
        "name": "kill_switch_audit_trail",
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "blockers": [] if ready else ["kill_switch_not_clear_or_not_auditable"],
        "evidence": {
            "path": str(path),
            "halt": payload.get("halt"),
            "halt_latched": payload.get("halt_latched"),
            "halt_required": payload.get("halt_required"),
            "operator_stop": payload.get("operator_stop"),
            "clear_ready": payload.get("clear_ready"),
            "safe_clear_ready": safe_clear.get("ready"),
        },
    }


CUSTOM_CHECKS = {
    "raw_profitability_truth": _raw_profitability_truth,
    "live_execution_double_lock": _live_execution_double_lock,
    "storage_soft_quota_escalator": _storage_soft_quota,
    "no_fake_green_dashboard": _no_fake_green_dashboard,
    "kill_switch_audit_trail": _kill_switch_audit,
}


def _evaluate_item(project_root: Path, item: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    requirement_rows = [_artifact_row(project_root, row, now=now) for row in _as_list(item.get("requirements")) if isinstance(row, dict)]
    dynamic_rows = [_dynamic_row(project_root, name) for name in _string_list(item.get("dynamic_checks"))]
    custom_rows = []
    for name in _string_list(item.get("custom_checks")):
        fn = CUSTOM_CHECKS.get(name)
        if fn is None:
            custom_rows.append({"name": name, "ready": False, "status": "blocked", "blockers": [f"unknown_custom_check:{name}"], "evidence": {}})
        else:
            custom_rows.append(fn(project_root))
    blockers = []
    for row in requirement_rows:
        blockers.extend(f"{Path(str(row.get('path'))).name}:{blocker}" for blocker in _string_list(row.get("blockers")))
    for row in dynamic_rows + custom_rows:
        blockers.extend(f"{row.get('name')}:{blocker}" for blocker in _string_list(row.get("blockers")))
    blockers = ordered_unique(blockers)
    ready = bool(not blockers and (requirement_rows or dynamic_rows or custom_rows))
    return {
        "control_id": str(item.get("control_id") or ""),
        "group": str(item.get("group") or ""),
        "title": str(item.get("title") or item.get("control_id") or ""),
        "status": "ready" if ready else "needs_work",
        "ready": ready,
        "blockers": blockers,
        "requirement_rows": requirement_rows,
        "dynamic_rows": dynamic_rows,
        "custom_rows": custom_rows,
        "commands": _as_list(item.get("commands")),
        "expected_impact": str(item.get("expected_impact") or ""),
        "production_contract": str(item.get("production_contract") or ""),
        "live_execution_authority": False,
    }


def _dedupe_commands(rows: list[dict[str, Any]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    commands: list[list[str]] = []
    for row in rows:
        if row.get("ready"):
            continue
        for raw in _as_list(row.get("commands")):
            if not isinstance(raw, list):
                continue
            key = tuple(str(part) for part in raw)
            if key in seen:
                continue
            seen.add(key)
            commands.append(list(key))
    return commands


def _grade(ready_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "F"
    if ready_count == total_count:
        return "A+"
    ratio = ready_count / total_count
    if ratio >= 0.90:
        return "A"
    if ratio >= 0.75:
        return "B"
    if ratio >= 0.50:
        return "C"
    if ratio >= 0.25:
        return "D"
    return "F"


def _group_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        group = str(row.get("group") or "unknown")
        stats = out.setdefault(group, {"total": 0, "ready": 0, "needs_work": 0})
        stats["total"] += 1
        if row.get("ready"):
            stats["ready"] += 1
        else:
            stats["needs_work"] += 1
    return out


def _quality_checks(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    group_counts = _group_counts(rows)
    ids = [str(row.get("control_id") or "") for row in rows]
    return {
        "exactly_twenty_items": len(rows) == 20,
        "ten_production_upgrades": group_counts.get("production_upgrade", {}).get("total", 0) == EXPECTED_GROUP_COUNTS["production_upgrade"],
        "ten_hardeners": group_counts.get("hardener", {}).get("total", 0) == EXPECTED_GROUP_COUNTS["hardener"],
        "control_ids_unique": len(ids) == len(set(ids)),
        "all_items_have_commands_or_dynamic_checks": all(bool(row.get("commands") or row.get("dynamic_rows") or row.get("custom_rows")) for row in rows),
        "all_live_execution_authority_false": all(not bool(row.get("live_execution_authority", False)) for row in rows),
        "config_live_execution_authority_false": not bool(_as_dict(config.get("control_contract")).get("live_execution_authority", True)),
        "raw_profitability_truth_required": bool(_as_dict(config.get("control_contract")).get("raw_profitability_truth_must_remain_visible", False)),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Production Level Upgrade And Hardener Control",
        "",
        f"Timestamp UTC: {payload.get('timestamp_utc')}",
        f"Status: {payload.get('overall_status')}",
        f"Grade: {payload.get('grade')}",
        f"Ready: {payload.get('ready_count')}/{payload.get('item_count')}",
        "",
        "Live execution authority: false",
        f"Raw profitability truth preserved: {payload.get('raw_profitability_truth_preserved')}",
        "",
    ]
    for group, label in (("production_upgrade", "Production Upgrades"), ("hardener", "Hardeners")):
        lines.extend([f"## {label}", ""])
        for row in payload.get("items", []):
            if row.get("group") != group:
                continue
            lines.append(f"- {row.get('status')}: {row.get('title')} ({row.get('control_id')})")
            if row.get("blockers"):
                lines.append(f"  blockers: {', '.join(str(item) for item in row.get('blockers', []))}")
        lines.append("")
    if payload.get("ordered_repair_commands"):
        lines.extend(["## Ordered Repair Commands", ""])
        for command in payload.get("ordered_repair_commands", []):
            lines.append("- " + " ".join(str(part) for part in command))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_path: Path = DEFAULT_OUT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    apply: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    now = datetime.now(timezone.utc)
    items = [_evaluate_item(project_root, row, now=now) for row in _as_list(config.get("items")) if isinstance(row, dict)]
    quality_checks = _quality_checks(items, config)
    ready_count = sum(1 for row in items if row.get("ready"))
    item_count = len(items)
    blockers = ordered_unique([f"{row['control_id']}:{blocker}" for row in items for blocker in _string_list(row.get("blockers"))])
    control_shape_ok = all(bool(value) for value in quality_checks.values())
    overall_ready = bool(control_shape_ok and item_count == 20 and ready_count == item_count)
    raw_truth_row = next((row for row in items if row.get("control_id") == "raw_profitability_explainer"), {})
    raw_truth_preserved = not any("raw_profitability_truth" in blocker for blocker in _string_list(raw_truth_row.get("blockers")))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "source": "production_level_upgrade_hardener_control",
        "overall_status": "ready" if overall_ready else "needs_work",
        "ok": overall_ready,
        "grade": _grade(ready_count, item_count),
        "target_grade": str(config.get("target_grade") or "A+"),
        "item_count": item_count,
        "ready_count": ready_count,
        "needs_work_count": item_count - ready_count,
        "group_counts": _group_counts(items),
        "quality_checks": quality_checks,
        "blockers": blockers,
        "items": items,
        "ordered_repair_commands": _dedupe_commands(items),
        "control_contract": {
            "live_execution_authority": False,
            "live_orders_must_remain_disabled": True,
            "paper_soak_safe": True,
            "safe_apply_only": True,
            "raw_profitability_truth_must_remain_visible": True,
            "raw_profitability_truth_preserved": raw_truth_preserved,
            "degraded_but_managed_states_must_remain_visible": True,
        },
        "raw_profitability_truth_preserved": raw_truth_preserved,
        "live_execution_authority": False,
        "recommended_actions": ordered_unique(
            [
                "rerun production-level-upgrades --apply --json after any production-hardening change",
                "keep live execution disabled until live-canary milestones and raw profitability gates clear",
                "run the ordered repair commands for any needs_work row, then refresh this control",
                "commit source changes before expecting source_mutation_runtime_firewall to report ready",
            ]
        ),
        "artifact_paths": {
            "json": str(out_path),
            "markdown": str(markdown_path),
            "config": str(config_path),
        },
    }
    if apply:
        write_payload(out_path, payload)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the 10 production upgrades and 10 hardeners contract.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--apply", action="store_true", help="Write JSON and operator markdown artifacts.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero unless all 20 controls are ready.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        args.project_root.resolve(),
        config_path=args.config.resolve(),
        out_path=args.out.resolve(),
        markdown_path=args.markdown.resolve(),
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_level_upgrade_hardener_control "
            f"status={payload['overall_status']} "
            f"grade={payload['grade']} "
            f"ready={payload['ready_count']}/{payload['item_count']}"
        )
    return 2 if args.check and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
