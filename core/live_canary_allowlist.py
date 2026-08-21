from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (payload if isinstance(payload, dict) else {}), isinstance(payload, dict)


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _parse_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _symbols(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        symbol = str(item or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def evaluate_live_canary_allowlist(
    project_root: str | Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    readiness, _ = _load_json(root / "config" / "production_readiness_control_v1.json")
    firewall = readiness.get("live_execution_risk_firewall")
    firewall = firewall if isinstance(firewall, dict) else {}
    allowlist_path = _project_path(root, firewall.get("canary_allowlist_path"))
    plan_path = _project_path(
        root,
        firewall.get("canary_plan_path") or "config/live_canary_micro_policy_v1.json",
    )
    candidate_path = _project_path(
        root,
        firewall.get("production_candidate_state_path")
        or "governance/runtime/production_candidate_state.json",
    )
    lifecycle_path = _project_path(
        root,
        firewall.get("symbol_lifecycle_path") or "config/symbol_lifecycle_v1.json",
    )

    allowlist, allowlist_valid_json = _load_json(allowlist_path)
    plan, plan_valid_json = _load_json(plan_path)
    candidate, candidate_valid_json = _load_json(candidate_path)
    lifecycle, lifecycle_valid_json = _load_json(lifecycle_path)

    blockers: list[str] = []
    if not allowlist_path.exists():
        blockers.append("canary_allowlist_missing")
    elif not allowlist_valid_json:
        blockers.append("canary_allowlist_invalid_json")
    if not plan_valid_json:
        blockers.append("canary_plan_invalid")
    if not candidate_valid_json:
        blockers.append("production_candidate_state_invalid")
    if not lifecycle_valid_json:
        blockers.append("symbol_lifecycle_invalid")

    enabled = bool(allowlist.get("enabled", False))
    if allowlist_valid_json and not enabled:
        blockers.append("canary_allowlist_disabled")
    if allowlist_valid_json and int(allowlist.get("schema_version", 0) or 0) != 1:
        blockers.append("canary_allowlist_schema_invalid")

    current_candidate_id = str(candidate.get("candidate_id") or "").strip()
    candidate_accepted_at = _parse_timestamp(candidate.get("accepted_at_utc"))
    if candidate_valid_json and not candidate_accepted_at:
        blockers.append("production_candidate_acceptance_missing")
    allowlist_candidate_id = str(allowlist.get("candidate_id") or "").strip()
    candidate_matches = bool(current_candidate_id and allowlist_candidate_id == current_candidate_id)
    if allowlist_valid_json and not candidate_matches:
        blockers.append("canary_allowlist_candidate_mismatch")

    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued_at = _parse_timestamp(allowlist.get("issued_at_utc"))
    expires_at = _parse_timestamp(allowlist.get("expires_at_utc"))
    activation = plan.get("activation_contract") if isinstance(plan.get("activation_contract"), dict) else {}
    try:
        max_allowlist_hours = max(float(activation.get("max_allowlist_duration_hours", 4.0) or 4.0), 0.0)
    except Exception:
        max_allowlist_hours = 4.0
    if allowlist_valid_json and (
        not issued_at
        or issued_at > now_utc + timedelta(minutes=5)
    ):
        blockers.append("canary_allowlist_issued_at_invalid")
    if (
        allowlist_valid_json
        and issued_at
        and candidate_accepted_at
        and issued_at < candidate_accepted_at
    ):
        blockers.append("canary_allowlist_predates_candidate_acceptance")
    unexpired = bool(expires_at and expires_at > now_utc)
    if allowlist_valid_json and not unexpired:
        blockers.append("canary_allowlist_expired_or_invalid")
    duration_hours = (
        (expires_at - issued_at).total_seconds() / 3600.0
        if issued_at and expires_at
        else 0.0
    )
    if (
        allowlist_valid_json
        and (
            duration_hours <= 0.0
            or max_allowlist_hours <= 0.0
            or duration_hours > max_allowlist_hours
        )
    ):
        blockers.append("canary_allowlist_duration_exceeds_policy")

    try:
        stage = int(allowlist.get("stage", 0) or 0)
    except Exception:
        stage = 0
    stages = [row for row in plan.get("stages", []) if isinstance(row, dict)] if plan_valid_json else []
    stage_row = next((row for row in stages if int(row.get("stage", 0) or 0) == stage), {})
    stage_symbols = _symbols(stage_row.get("symbols"))
    allowlist_symbols = _symbols(allowlist.get("symbols"))
    if allowlist_valid_json and not stage_row:
        blockers.append("canary_allowlist_stage_invalid")
    if allowlist_valid_json and not allowlist_symbols:
        blockers.append("canary_allowlist_empty")
    out_of_plan_symbols = sorted(set(allowlist_symbols) - set(stage_symbols))
    if out_of_plan_symbols:
        blockers.append("canary_allowlist_symbol_not_in_stage")

    renamed = lifecycle.get("renamed_symbols") if isinstance(lifecycle.get("renamed_symbols"), dict) else {}
    deprecated_symbols = sorted(symbol for symbol in allowlist_symbols if symbol in {str(key).upper() for key in renamed})
    if deprecated_symbols:
        blockers.append("canary_allowlist_contains_deprecated_symbol")

    hard_limits = plan.get("hard_limits") if isinstance(plan.get("hard_limits"), dict) else {}
    return {
        "ready": not blockers,
        "blockers": blockers,
        "path": str(allowlist_path),
        "exists": allowlist_path.exists(),
        "enabled": enabled,
        "candidate_id": allowlist_candidate_id,
        "current_candidate_id": current_candidate_id,
        "candidate_accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        "candidate_matches": candidate_matches,
        "issued_at_utc": str(allowlist.get("issued_at_utc") or ""),
        "expires_at_utc": str(allowlist.get("expires_at_utc") or ""),
        "duration_hours": round(float(duration_hours), 6),
        "max_allowlist_duration_hours": float(max_allowlist_hours),
        "unexpired": unexpired,
        "stage": stage,
        "symbols": allowlist_symbols,
        "stage_symbols": stage_symbols,
        "out_of_plan_symbols": out_of_plan_symbols,
        "deprecated_symbols": deprecated_symbols,
        "plan_path": str(plan_path),
        "plan_status": str(plan.get("status") or "missing"),
        "planned_stages": stages,
        "hard_limits": hard_limits,
        "candidate_state_path": str(candidate_path),
        "symbol_lifecycle_path": str(lifecycle_path),
        "live_execution_authority": False,
    }
