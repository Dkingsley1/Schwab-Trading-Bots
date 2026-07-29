#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.dependency_activation_smoke import build_payload as build_dependency_activation_smoke
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, status_rank, write_payload
else:
    from .dependency_activation_smoke import build_payload as build_dependency_activation_smoke
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, status_rank, write_payload


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_readiness_control_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "production_readiness_control_latest.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "exports" / "reports" / "operator" / "production_readiness_control_latest.md"

READY_STATUSES = {"ready", "ok", "active", "applied", "guarded", "ready_guarded", "pending_install", "thin", "advisory"}
BLOCKING_STATUSES = {"blocked", "critical", "failed", "error"}
SECRET_REDACTION = "[REDACTED]"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _payload_count_metric(payload: dict[str, Any], key: str) -> tuple[Any, bool]:
    if key in payload:
        return payload.get(key), False
    if key == "unsafe_skipped_blob_count" and "skipped_blob_count" in payload:
        return payload.get("skipped_blob_count"), True
    return None, False


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


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
    return "ready" if payload else "missing"


def _artifact_row(project_root: Path, raw_path: str) -> dict[str, Any]:
    path = _project_path(project_root, raw_path)
    payload = load_json(path)
    exists = path.exists()
    status = _extract_status(payload) if exists else "missing"
    return {
        "path": str(path),
        "exists": exists,
        "status": status,
        "ok": status in READY_STATUSES,
        "summary_keys": sorted(payload.keys())[:20] if payload else [],
    }


def _artifact_capability_row(project_root: Path, raw: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    capability_id = str(raw.get("capability_id") or raw.get("name") or "").strip()
    raw_path = str(raw.get("artifact") or raw.get("path") or "").strip()
    path = _project_path(project_root, raw_path)
    payload = load_json(path)
    exists = path.exists()
    status = _extract_status(payload) if exists else "missing"
    ready_statuses = set(_string_list(raw.get("ready_statuses")) or ["ready", "ok", "stable"])
    max_age_hours = _safe_float(raw.get("max_age_hours"), 0.0)
    age_minutes = payload_age_minutes(payload, path, now=now) if payload else None
    fresh = bool(max_age_hours <= 0.0 or (age_minutes is not None and age_minutes <= max_age_hours * 60.0))
    truthy_keys = _string_list(raw.get("truthy_keys"))
    truthy_ok = all(bool(payload.get(key, False)) for key in truthy_keys) if truthy_keys else True
    zero_count_keys = _string_list(raw.get("zero_count_keys"))
    zero_counts_ok = all(_safe_int(payload.get(key), 1) == 0 for key in zero_count_keys) if zero_count_keys else True
    max_count_by_key = raw.get("max_count_by_key") if isinstance(raw.get("max_count_by_key"), dict) else {}
    max_counts_ok = True
    max_count_rows: list[dict[str, Any]] = []
    for key, ceiling in max_count_by_key.items():
        metric_key = str(key)
        metric_value, legacy_fallback = _payload_count_metric(payload, metric_key)
        actual = _safe_float(metric_value, 0.0)
        allowed = _safe_float(ceiling, 0.0)
        row_ok = actual <= allowed
        max_count_rows.append(
            {
                "key": metric_key,
                "value": actual,
                "ceiling": allowed,
                "ok": row_ok,
                "legacy_fallback_from_skipped_blob_count": bool(legacy_fallback),
            }
        )
        if not row_ok:
            max_counts_ok = False
    status_ok = bool(status in ready_statuses or (bool(payload.get("ok", False)) and "ok" in ready_statuses))
    ready = bool(payload and status_ok and fresh and truthy_ok and zero_counts_ok and max_counts_ok)
    blockers = ordered_unique(
        [
            f"{capability_id}_missing" if not payload else "",
            f"{capability_id}_status_not_ready" if payload and not status_ok else "",
            f"{capability_id}_stale" if payload and not fresh else "",
            f"{capability_id}_truthy_keys_not_met" if payload and not truthy_ok else "",
            f"{capability_id}_zero_count_keys_not_met" if payload and not zero_counts_ok else "",
            f"{capability_id}_max_count_keys_not_met" if payload and not max_counts_ok else "",
        ]
    )
    return {
        "capability_id": capability_id,
        "title": str(raw.get("title") or capability_id).strip(),
        "required": bool(raw.get("required", True)),
        "path": str(path),
        "exists": exists,
        "status": status,
        "ok": payload.get("ok") if payload else None,
        "ready": ready,
        "blockers": blockers,
        "age_minutes": round(float(age_minutes), 4) if age_minutes is not None else None,
        "max_age_minutes": round(float(max_age_hours) * 60.0, 4) if max_age_hours > 0.0 else 0.0,
        "fresh": fresh,
        "truthy_keys": truthy_keys,
        "zero_count_keys": zero_count_keys,
        "max_count_by_key": max_count_by_key,
        "max_count_rows": max_count_rows,
        "summary_keys": sorted(payload.keys())[:20] if payload else [],
    }


def _domain(name: str, status: str, *, evidence: dict[str, Any] | None = None, blockers: list[str] | None = None, actions: list[str] | None = None) -> dict[str, Any]:
    status = str(status or "advisory").strip().lower()
    blockers = ordered_unique(blockers or [])
    actions = ordered_unique(actions or [])
    return {
        "name": name,
        "status": status,
        "ok": status not in BLOCKING_STATUSES,
        "blockers": blockers,
        "recommended_actions": actions,
        "evidence": evidence or {},
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_jsonish(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_activation_domain(
    project_root: Path,
    config: dict[str, Any],
    *,
    batch: str,
    profile: str,
    installed_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected_batch = batch or str(config.get("default_dependency_activation_batch") or "production_core_safe")
    payload = build_dependency_activation_smoke(
        project_root,
        batch=selected_batch,
        profile=profile,
        import_smoke=False,
        require_installed=False,
        installed_versions=installed_versions,
    )
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    status = "ready" if payload.get("ok") else "blocked"
    if _safe_int(summary.get("pending_install_count"), 0):
        status = "pending_install"
    return _domain(
        "dependency_activation_smoke_runner",
        status,
        evidence={
            "selection": payload.get("selection", {}),
            "summary": summary,
            "artifact_paths": payload.get("artifact_paths", {}),
            "control_contract": payload.get("control_contract", {}),
        },
        actions=_string_list(payload.get("recommended_actions")),
    )


def _order_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("orders", "order_intents", "intents", "proposed_orders"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return [payload] if payload else []


def _order_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _validate_order_intents(rows: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    blockers: list[str] = []
    max_notional = _safe_float(policy.get("max_single_order_notional"), 0.0)
    max_quantity = _safe_float(policy.get("max_order_quantity"), 0.0)
    max_quote_age = _safe_float(policy.get("max_quote_age_seconds"), 0.0)
    max_spread = _safe_float(policy.get("max_spread_bps"), 0.0)
    max_daily_loss = _safe_float(policy.get("max_daily_loss"), 0.0)
    for index, row in enumerate(rows):
        symbol = str(_order_value(row, "symbol", "ticker") or "").strip().upper()
        side = str(_order_value(row, "side", "instruction", "action") or "").strip().upper()
        quantity = _safe_float(_order_value(row, "quantity", "qty", "shares"), 0.0)
        price = _safe_float(_order_value(row, "limit_price", "price", "estimated_price"), 0.0)
        notional = _safe_float(_order_value(row, "notional", "estimated_notional"), quantity * price)
        quote_age = _safe_float(_order_value(row, "quote_age_seconds", "market_data_age_seconds"), 0.0)
        spread_bps = _safe_float(_order_value(row, "spread_bps", "estimated_spread_bps"), 0.0)
        daily_loss = abs(_safe_float(_order_value(row, "daily_loss_if_filled", "estimated_daily_loss_after_order"), 0.0))
        strategy = str(_order_value(row, "strategy_id", "bot_id", "source") or "").strip()
        key = f"{symbol}|{side}|{quantity}|{price}|{strategy}"
        reasons: list[str] = []
        if not symbol:
            reasons.append("missing_symbol")
        if side not in {"BUY", "SELL", "BUY_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_OPEN", "SELL_TO_CLOSE"}:
            reasons.append("invalid_side")
        if quantity <= 0:
            reasons.append("quantity_must_be_positive")
        if max_quantity and quantity > max_quantity:
            reasons.append("quantity_exceeds_cap")
        if price <= 0:
            reasons.append("price_must_be_positive")
        if max_notional and notional > max_notional:
            reasons.append("notional_exceeds_cap")
        if max_quote_age and quote_age > max_quote_age:
            reasons.append("quote_is_stale")
        if max_spread and spread_bps > max_spread:
            reasons.append("spread_exceeds_cap")
        if max_daily_loss and daily_loss > max_daily_loss:
            reasons.append("daily_loss_exceeds_cap")
        if key in seen:
            reasons.append("duplicate_order_intent")
        seen.add(key)
        if reasons:
            blockers.extend(f"order_{index}:{reason}" for reason in reasons)
        checked.append(
            {
                "index": index,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "notional": round(notional, 6),
                "quote_age_seconds": quote_age,
                "spread_bps": spread_bps,
                "strategy_id": strategy,
                "status": "blocked" if reasons else "ready",
                "reasons": reasons,
            }
        )
    return checked, ordered_unique(blockers)


def _live_execution_risk_firewall(
    project_root: Path,
    policy: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    order_intents_path: Path | None = None,
) -> dict[str, Any]:
    env = env if isinstance(env, dict) else dict(os.environ)
    allow_env = str(policy.get("allow_order_execution_env") or "ALLOW_ORDER_EXECUTION")
    market_data_env = str(policy.get("market_data_only_env") or "MARKET_DATA_ONLY")
    execution_armed = _truthy(env.get(allow_env), False)
    market_data_only_default = bool(policy.get("market_data_only_default", True))
    market_data_only = _truthy(env.get(market_data_env), market_data_only_default)
    halt_flags = [_project_path(project_root, path) for path in _string_list(policy.get("halt_flags"))]
    required_safety_flags = [_project_path(project_root, path) for path in _string_list(policy.get("required_safety_flags"))]
    active_halt_flags = [str(path) for path in halt_flags if path.exists()]
    missing_safety_flags = [str(path) for path in required_safety_flags if not path.exists()]
    intent_path = order_intents_path or _project_path(project_root, policy.get("order_intent_path") or "")
    intent_payload = load_json(intent_path)
    order_rows = _order_rows_from_payload(intent_payload)
    checked_orders, order_blockers = _validate_order_intents(order_rows, policy) if intent_payload else ([], [])
    live_order_allowed = bool(
        execution_armed
        and not market_data_only
        and not active_halt_flags
        and not missing_safety_flags
        and not order_blockers
    )
    status = "ready"
    blockers: list[str] = []
    if order_blockers:
        status = "blocked"
        blockers.extend(order_blockers)
    if active_halt_flags:
        status = "ready_guarded"
        blockers.append("halt_flags_active")
    if missing_safety_flags:
        status = "ready_guarded"
        blockers.append("required_safety_flag_missing")
    if not execution_armed:
        status = "ready_guarded" if status != "blocked" else status
        blockers.append("live_execution_not_armed")
    if market_data_only:
        status = "ready_guarded" if status != "blocked" else status
        blockers.append("market_data_only_active")
    return _domain(
        "live_execution_risk_firewall",
        status,
        evidence={
            "execution_armed": execution_armed,
            "market_data_only": market_data_only,
            "market_data_only_env": market_data_env,
            "market_data_only_default": market_data_only_default,
            "live_order_allowed": live_order_allowed,
            "active_halt_flags": active_halt_flags,
            "missing_safety_flags": missing_safety_flags,
            "order_intents_path": str(intent_path),
            "order_intent_count": len(order_rows),
            "checked_orders": checked_orders[:50],
            "policy": policy,
        },
        blockers=blockers,
        actions=[
            "keep ALLOW_ORDER_EXECUTION=0 unless an operator intentionally arms a smoke-reviewed live path",
            "repair or acknowledge every order firewall blocker before live order submission",
        ],
    )


def _deterministic_replay_domain(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    artifact_rows = [_artifact_row(project_root, path) for path in _string_list(config.get("required_artifacts"))]
    blocked_artifacts = [row["path"] for row in artifact_rows if str(row.get("status")) in BLOCKING_STATUSES]
    existing_artifacts = [row for row in artifact_rows if row.get("exists")]
    fingerprints = []
    for raw_path in _string_list(config.get("fingerprint_paths")):
        path = _project_path(project_root, raw_path)
        if not path.exists() or not path.is_file():
            continue
        fingerprints.append({"path": str(path), "sha256": _hash_file(path), "bytes": path.stat().st_size})
    fingerprint_hash = _hash_jsonish(fingerprints)
    baseline_path = _project_path(project_root, config.get("baseline_path") or "")
    baseline = load_json(baseline_path)
    baseline_hash = str(baseline.get("fingerprint_hash") or "")
    mismatch = bool(baseline_hash and baseline_hash != fingerprint_hash)
    status = "blocked" if blocked_artifacts or mismatch else "ready" if existing_artifacts and fingerprints else "advisory"
    blockers = []
    if blocked_artifacts:
        blockers.append("replay_artifact_blocked")
    if mismatch:
        blockers.append("deterministic_replay_fingerprint_mismatch")
    return _domain(
        "deterministic_replay_harness",
        status,
        evidence={
            "artifact_rows": artifact_rows,
            "fingerprint_count": len(fingerprints),
            "fingerprint_hash": fingerprint_hash,
            "baseline_path": str(baseline_path),
            "baseline_present": bool(baseline),
            "baseline_hash": baseline_hash,
            "fingerprints": fingerprints,
        },
        blockers=blockers,
        actions=[
            "refresh deterministic replay baseline only after replay diffs and promotion gate evidence are reviewed"
            if mismatch or not baseline
            else "",
        ],
    )


def _redact_text(text: str, patterns: list[str]) -> str:
    redacted = str(text)
    for pattern in patterns:
        try:
            redacted = re.sub(pattern, SECRET_REDACTION, redacted)
        except re.error:
            continue
    return redacted


def _observability_redaction_domain(config: dict[str, Any]) -> dict[str, Any]:
    patterns = _string_list(config.get("redaction_patterns"))
    sample_rows = config.get("redaction_samples") if isinstance(config.get("redaction_samples"), list) else []
    checked = []
    blockers: list[str] = []
    for sample in sample_rows:
        if not isinstance(sample, dict):
            continue
        raw = str(sample.get("input") or "")
        redacted = _redact_text(raw, patterns)
        leaks = [needle for needle in _string_list(sample.get("must_not_contain")) if needle and needle in redacted]
        if leaks:
            blockers.append(f"{sample.get('name', 'sample')}:redaction_leak")
        checked.append(
            {
                "name": str(sample.get("name") or "sample"),
                "ok": not leaks,
                "redacted": redacted,
                "leak_count": len(leaks),
            }
        )
    status = "blocked" if blockers else "ready"
    return _domain(
        "observability_redaction",
        status,
        evidence={
            "enabled_by_default": bool(config.get("enabled_by_default", False)),
            "allowed_export_modes": _string_list(config.get("allowed_export_modes")),
            "pattern_count": len(patterns),
            "sample_rows": checked,
            "policy": str(config.get("policy") or ""),
        },
        blockers=blockers,
        actions=["keep telemetry exporters local/off until redaction samples and low-cardinality labels pass"],
    )


def _artifact_gate_domain(project_root: Path, name: str, config: dict[str, Any], *, scripts_key: str = "") -> dict[str, Any]:
    script_rows = []
    missing_scripts = []
    if scripts_key:
        for raw_path in _string_list(config.get(scripts_key)):
            path = _project_path(project_root, raw_path)
            exists = path.exists()
            script_rows.append({"path": str(path), "exists": exists})
            if not exists:
                missing_scripts.append(str(path))
    artifact_rows = [_artifact_row(project_root, path) for path in _string_list(config.get("required_artifacts"))]
    blocked_artifacts = [row["path"] for row in artifact_rows if str(row.get("status")) in BLOCKING_STATUSES]
    missing_artifacts = [row["path"] for row in artifact_rows if not row.get("exists")]
    if missing_scripts or blocked_artifacts:
        status = "blocked"
    elif missing_artifacts:
        status = "advisory"
    else:
        status = "ready"
    blockers = []
    if missing_scripts:
        blockers.append("required_scripts_missing")
    if blocked_artifacts:
        blockers.append("required_artifacts_blocked")
    return _domain(
        name,
        status,
        evidence={
            "script_rows": script_rows,
            "artifact_rows": artifact_rows,
            "missing_artifact_count": len(missing_artifacts),
            "policy": str(config.get("policy") or ""),
        },
        blockers=blockers,
        actions=[
            f"refresh missing {name} artifacts before live promotion" if missing_artifacts else "",
            f"repair blocked {name} artifacts before release" if blocked_artifacts else "",
        ],
    )


def _incident_rollback_domain(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    snapshot_rows = []
    for raw_path in _string_list(config.get("snapshot_paths")):
        path = _project_path(project_root, raw_path)
        if path.exists() and path.is_file():
            snapshot_rows.append({"path": str(path), "sha256": _hash_file(path), "bytes": path.stat().st_size})
        else:
            snapshot_rows.append({"path": str(path), "missing": True})
    promotion_packet_rows = [_artifact_row(project_root, path) for path in _string_list(config.get("promotion_packet_paths"))]
    rollback_commands = _string_list(config.get("rollback_commands"))
    missing_snapshots = [row["path"] for row in snapshot_rows if row.get("missing")]
    status = "blocked" if not rollback_commands else "advisory" if missing_snapshots else "ready"
    manifest = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "snapshot_rows": snapshot_rows,
        "promotion_packet_rows": promotion_packet_rows,
        "rollback_commands": rollback_commands,
        "snapshot_manifest_hash": _hash_jsonish(snapshot_rows),
        "policy": str(config.get("policy") or ""),
    }
    return _domain(
        "incident_and_rollback_system",
        status,
        evidence={
            "manifest": manifest,
            "rollback_manifest_path": str(_project_path(project_root, config.get("rollback_manifest_path") or "")),
            "missing_snapshot_count": len(missing_snapshots),
        },
        blockers=["rollback_commands_missing"] if not rollback_commands else [],
        actions=["review missing rollback snapshots" if missing_snapshots else ""],
    )


def _slo_error_budget_domain(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    artifact_rows = [_artifact_row(project_root, path) for path in _string_list(config.get("required_artifacts"))]
    blocked_artifacts = [row["path"] for row in artifact_rows if str(row.get("status")) in BLOCKING_STATUSES]
    missing_artifacts = [row["path"] for row in artifact_rows if not row.get("exists")]
    target_success = _safe_float(config.get("target_success_ratio"), 0.999)
    error_budget = max(1.0 - target_success, 0.0)
    budget_rows = []
    for row in artifact_rows:
        payload = load_json(Path(str(row.get("path") or ""))) if row.get("exists") else {}
        budget_rows.append(
            {
                "path": row.get("path"),
                "status": row.get("status"),
                "error_budget_remaining": payload.get("error_budget_remaining"),
                "budget_burn": payload.get("budget_burn"),
                "ok": row.get("ok"),
            }
        )
    if blocked_artifacts:
        status = "blocked"
    elif missing_artifacts:
        status = "advisory"
    else:
        status = "ready"
    return _domain(
        "slo_error_budget_policy",
        status,
        evidence={
            "target_success_ratio": target_success,
            "error_budget": round(error_budget, 6),
            "max_single_incident_budget_burn": _safe_float(config.get("max_single_incident_budget_burn"), 0.2),
            "budget_rows": budget_rows,
            "policy": str(config.get("policy") or ""),
        },
        blockers=["slo_artifact_blocked"] if blocked_artifacts else [],
        actions=["freeze feature promotion until SLO blockers recover" if blocked_artifacts else ""],
    )


def _live_money_production_bar_domain(project_root: Path, config: dict[str, Any], base_domains: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(domain.get("name") or ""): domain for domain in base_domains}
    now = datetime.now(timezone.utc)
    domain_requirements = config.get("required_domain_statuses") if isinstance(config.get("required_domain_statuses"), dict) else {}
    domain_rows: list[dict[str, Any]] = []
    domain_blockers: list[str] = []
    for name, raw_allowed in domain_requirements.items():
        allowed = set(_string_list(raw_allowed) if not isinstance(raw_allowed, str) else [raw_allowed])
        if not allowed:
            allowed = {"ready"}
        domain = by_name.get(str(name))
        actual = str((domain or {}).get("status") or "missing").strip().lower()
        ready = bool(domain and actual in allowed)
        if not ready:
            domain_blockers.append(f"{name}_domain_not_production_ready")
        domain_rows.append(
            {
                "name": str(name),
                "status": actual,
                "allowed_statuses": sorted(allowed),
                "ready": ready,
                "blockers": [] if ready else [f"{name}_domain_not_production_ready"],
            }
        )

    firewall = by_name.get("live_execution_risk_firewall", {})
    firewall_evidence = firewall.get("evidence") if isinstance(firewall.get("evidence"), dict) else {}
    require_read_only = bool(config.get("require_read_only_pre_canary", True))
    read_only_ready = bool(
        not require_read_only
        or (
            not bool(firewall_evidence.get("execution_armed", False))
            and bool(firewall_evidence.get("market_data_only", True))
            and not bool(firewall_evidence.get("live_order_allowed", False))
        )
    )

    capability_rows = [
        _artifact_capability_row(project_root, row, now=now)
        for row in (config.get("required_capabilities") if isinstance(config.get("required_capabilities"), list) else [])
        if isinstance(row, dict)
    ]
    capability_blockers = [
        f"{row['capability_id']}:{blocker}"
        for row in capability_rows
        if bool(row.get("required", True))
        for blocker in _string_list(row.get("blockers"))
    ]
    blockers = ordered_unique(
        [
            *domain_blockers,
            "live_execution_firewall_not_read_only_pre_canary" if not read_only_ready else "",
            *capability_blockers,
        ]
    )
    required_capabilities = [row for row in capability_rows if bool(row.get("required", True))]
    status = "ready" if not blockers else "blocked"
    return _domain(
        "live_money_production_bar",
        status,
        evidence={
            "policy": str(config.get("policy") or "all production capabilities must be fresh before live-money canary consideration"),
            "require_read_only_pre_canary": require_read_only,
            "pre_canary_firewall_read_only": read_only_ready,
            "domain_rows": domain_rows,
            "capability_rows": capability_rows,
            "required_capability_count": len(required_capabilities),
            "ready_required_capability_count": sum(1 for row in required_capabilities if bool(row.get("ready", False))),
        },
        blockers=blockers,
        actions=[
            "refresh or implement every blocked live-money production-bar capability before live canary consideration"
            if blockers
            else "",
            "keep live execution read-only until the production bar and live canary milestones are both ready",
        ],
    )


def _live_promotion_allowed(domains: list[dict[str, Any]]) -> bool:
    if any(str(domain.get("status") or "") in BLOCKING_STATUSES for domain in domains):
        return False
    firewall = next((domain for domain in domains if domain.get("name") == "live_execution_risk_firewall"), {})
    firewall_evidence = firewall.get("evidence") if isinstance(firewall.get("evidence"), dict) else {}
    if not bool(firewall_evidence.get("live_order_allowed", False)):
        return False
    return all(str(domain.get("status") or "") == "ready" for domain in domains)


def _overall_status(domains: list[dict[str, Any]]) -> str:
    statuses = [str(domain.get("status") or "advisory") for domain in domains]
    if any(status in BLOCKING_STATUSES for status in statuses):
        return "blocked"
    if any(status in {"advisory", "pending_install", "ready_guarded", "guarded", "thin"} for status in statuses):
        return "guarded"
    return "ready"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    dependency_batch: str = "",
    activation_profile: str = "",
    order_intents_path: Path | None = None,
    env: dict[str, str] | None = None,
    installed_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    config_path = config_path or project_root / "config" / "production_readiness_control_v1.json"
    config = load_json(config_path)
    base_domains = [
        _dependency_activation_domain(
            project_root,
            config,
            batch=dependency_batch,
            profile=activation_profile,
            installed_versions=installed_versions,
        ),
        _live_execution_risk_firewall(
            project_root,
            config.get("live_execution_risk_firewall") if isinstance(config.get("live_execution_risk_firewall"), dict) else {},
            env=env,
            order_intents_path=order_intents_path,
        ),
        _deterministic_replay_domain(
            project_root,
            config.get("deterministic_replay") if isinstance(config.get("deterministic_replay"), dict) else {},
        ),
        _observability_redaction_domain(
            config.get("observability_redaction") if isinstance(config.get("observability_redaction"), dict) else {},
        ),
        _artifact_gate_domain(
            project_root,
            "release_gates",
            config.get("release_gates") if isinstance(config.get("release_gates"), dict) else {},
            scripts_key="required_scripts",
        ),
        _artifact_gate_domain(
            project_root,
            "data_integrity_gates",
            config.get("data_integrity_gates") if isinstance(config.get("data_integrity_gates"), dict) else {},
        ),
        _incident_rollback_domain(
            project_root,
            config.get("incident_rollback") if isinstance(config.get("incident_rollback"), dict) else {},
        ),
        _slo_error_budget_domain(
            project_root,
            config.get("slo_error_budget") if isinstance(config.get("slo_error_budget"), dict) else {},
        ),
    ]
    production_bar_config = config.get("live_money_production_bar") if isinstance(config.get("live_money_production_bar"), dict) else {}
    domains = list(base_domains)
    if bool(production_bar_config.get("enabled", True)):
        domains.append(_live_money_production_bar_domain(project_root, production_bar_config, base_domains))
    status = _overall_status(domains)
    live_allowed = _live_promotion_allowed(domains)
    production_bar = next((domain for domain in domains if domain.get("name") == "live_money_production_bar"), {})
    production_bar_ready = bool(production_bar and str(production_bar.get("status") or "") == "ready")
    base_domains_blocked = any(str(domain.get("status") or "") in BLOCKING_STATUSES for domain in base_domains)
    canary_consideration_ready = bool(production_bar_ready and not base_domains_blocked)
    blockers = ordered_unique(
        f"{domain.get('name')}:{blocker}"
        for domain in domains
        for blocker in _string_list(domain.get("blockers"))
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status != "blocked",
        "overall_status": status,
        "live_runtime_promotion_allowed": live_allowed,
        "live_money_production_bar_ready": production_bar_ready,
        "live_money_canary_consideration_ready": canary_consideration_ready,
        "live_money_canary_consideration_blocked": not canary_consideration_ready,
        "domain_count": len(domains),
        "ready_domain_count": sum(1 for domain in domains if str(domain.get("status") or "") == "ready"),
        "guarded_domain_count": sum(1 for domain in domains if str(domain.get("status") or "") in {"guarded", "ready_guarded", "pending_install", "advisory", "thin"}),
        "blocked_domain_count": sum(1 for domain in domains if str(domain.get("status") or "") in BLOCKING_STATUSES),
        "domains": domains,
        "blockers": blockers,
        "recommended_actions": ordered_unique(
            [
                action
                for domain in domains
                for action in _string_list(domain.get("recommended_actions"))
            ]
            + [
                "keep live execution disabled until every production readiness domain is ready and the risk firewall is explicitly armed"
                if not live_allowed
                else "",
            ]
        ),
        "control_contract": {
            "covers_controls_1_through_8": True,
            "covers_live_money_production_bar": True,
            "dependency_installation_is_separate_from_activation": True,
            "live_execution_rejects_by_default": True,
            "production_live_ready_requires_all_domains_ready": True,
            "live_money_production_bar_required_before_canary": bool(production_bar_config.get("require_for_live_canary", True)),
            "control_policy": str(config.get("control_policy") or "production_grade_controls_before_live_feature_enablement"),
        },
        "artifact_paths": {
            "json": str(DEFAULT_OUT),
            "markdown": str(DEFAULT_MARKDOWN),
            "config": str(config_path),
        },
    }


def _rollback_manifest_from_payload(payload: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    for domain in payload.get("domains") or []:
        if not isinstance(domain, dict) or domain.get("name") != "incident_and_rollback_system":
            continue
        evidence = domain.get("evidence") if isinstance(domain.get("evidence"), dict) else {}
        path = Path(str(evidence.get("rollback_manifest_path") or ""))
        manifest = evidence.get("manifest") if isinstance(evidence.get("manifest"), dict) else {}
        return path if str(path) else None, manifest
    return None, {}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Production Readiness Control",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        f"Live runtime promotion allowed: `{payload.get('live_runtime_promotion_allowed', False)}`",
        "",
        "## Domains",
        "",
    ]
    for domain in payload.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        lines.append(
            f"- `{domain.get('name', '')}`: `{domain.get('status', '')}` "
            f"blockers=`{', '.join(domain.get('blockers') or []) or 'none'}`"
        )
    lines.extend(["", "## Recommended Actions", ""])
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    apply: bool = False,
) -> dict[str, Any]:
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    result = {"json": str(out_path), "markdown": str(markdown_path), "rollback_manifest_written": False}
    if apply:
        rollback_path, manifest = _rollback_manifest_from_payload(payload)
        if rollback_path is not None and manifest:
            rollback_path.parent.mkdir(parents=True, exist_ok=True)
            write_payload(rollback_path, manifest)
            result["rollback_manifest_written"] = True
            result["rollback_manifest_path"] = str(rollback_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate production-readiness controls across dependency, risk, replay, release, data, rollback, and SLO gates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dependency-batch", default="")
    parser.add_argument("--activation-profile", default="")
    parser.add_argument("--order-intents", default="")
    parser.add_argument("--apply", action="store_true", help="Write rollback manifest in addition to readiness artifacts.")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        config_path=Path(args.config),
        dependency_batch=args.dependency_batch,
        activation_profile=args.activation_profile,
        order_intents_path=Path(args.order_intents) if args.order_intents else None,
    )
    write_result = write_outputs(
        payload,
        out_path=Path(args.out_file),
        markdown_path=Path(args.markdown_file),
        apply=args.apply,
    )
    payload["write_result"] = write_result
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "production_readiness_control "
            f"status={payload.get('overall_status')} "
            f"domains={payload.get('domain_count', 0)} "
            f"blocked={payload.get('blocked_domain_count', 0)} "
            f"live_allowed={payload.get('live_runtime_promotion_allowed', False)}"
        )
    return 0 if args.exit_zero or bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
