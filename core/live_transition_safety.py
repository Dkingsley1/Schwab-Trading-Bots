from __future__ import annotations

from typing import Any, Mapping


ENTRY_INTERLOCKS = (
    "restart_state_unreconciled",
    "auth_not_ready",
    "auth_generation_changed",
    "quote_stale",
    "decision_source_degraded",
    "broker_reconciliation_mismatch",
    "drawdown_limit_breached",
    "durable_storage_unavailable",
    "production_evidence_not_ready",
    "operator_release_missing",
)


def _truthy(signals: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = signals.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "stable"}


def evaluate_release_interlock(
    signals: Mapping[str, Any],
    *,
    risk_reducing_exit: bool = False,
) -> dict[str, Any]:
    checks = {
        "restart_state_unreconciled": not _truthy(signals, "restart_reconciled"),
        "auth_not_ready": not _truthy(signals, "auth_ready"),
        "auth_generation_changed": not _truthy(signals, "auth_generation_stable"),
        "quote_stale": not _truthy(signals, "quote_fresh"),
        "decision_source_degraded": not _truthy(signals, "sources_ready"),
        "broker_reconciliation_mismatch": not _truthy(signals, "reconciliation_clean"),
        "drawdown_limit_breached": not _truthy(signals, "drawdown_within_limit"),
        "durable_storage_unavailable": not _truthy(signals, "storage_ready"),
        "production_evidence_not_ready": not _truthy(signals, "production_ready"),
        "operator_release_missing": not _truthy(signals, "operator_release_present"),
    }
    entry_lock_reasons = [name for name in ENTRY_INTERLOCKS if checks.get(name, False)]
    exit_blockers = [
        name
        for name, ready in (
            ("risk_reducing_exit_route_unavailable", _truthy(signals, "exit_route_ready")),
            ("broker_unreachable_for_exit", _truthy(signals, "broker_reachable")),
            ("auth_not_ready_for_exit", _truthy(signals, "auth_ready")),
        )
        if not ready
    ]
    entry_allowed = not entry_lock_reasons
    risk_exit_allowed = not exit_blockers
    allowed = risk_exit_allowed if risk_reducing_exit else entry_allowed
    return {
        "ok": bool(allowed),
        "entry_allowed": bool(entry_allowed),
        "risk_reducing_exit_allowed": bool(risk_exit_allowed),
        "auto_relocked": bool(entry_lock_reasons),
        "entry_lock_reasons": entry_lock_reasons,
        "exit_blockers": exit_blockers,
        "checks": checks,
        "mode": "risk_reducing_exit" if risk_reducing_exit else "new_entry",
        "policy": "every unsafe state automatically relocks entries; the independently guarded exit path remains available only when broker reachability and auth permit it",
    }


def _rows_by_key(rows: Any, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get(key) or "").strip(): row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and str(row.get(key) or "").strip()
    }


def reconcile_broker_truth(
    local: Mapping[str, Any],
    broker: Mapping[str, Any],
    *,
    quantity_tolerance: float = 0.0001,
    buying_power_tolerance: float = 1.0,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    local_orders = _rows_by_key(local.get("orders"), "order_id")
    broker_orders = _rows_by_key(broker.get("orders"), "order_id")
    for order_id in sorted(set(local_orders) | set(broker_orders)):
        local_row = local_orders.get(order_id)
        broker_row = broker_orders.get(order_id)
        if local_row is None or broker_row is None:
            mismatches.append(
                {
                    "surface": "orders",
                    "key": order_id,
                    "reason": "missing_local" if local_row is None else "missing_broker",
                }
            )
            continue
        local_status = str(local_row.get("status") or "").strip().lower()
        broker_status = str(broker_row.get("status") or "").strip().lower()
        if local_status != broker_status:
            mismatches.append({"surface": "orders", "key": order_id, "reason": "status_mismatch", "local": local_status, "broker": broker_status})

    local_fills = _rows_by_key(local.get("fills"), "order_id")
    broker_fills = _rows_by_key(broker.get("fills"), "order_id")
    for order_id in sorted(set(local_fills) | set(broker_fills)):
        local_qty = float((local_fills.get(order_id) or {}).get("filled_quantity", 0.0) or 0.0)
        broker_qty = float((broker_fills.get(order_id) or {}).get("filled_quantity", 0.0) or 0.0)
        if abs(local_qty - broker_qty) > max(float(quantity_tolerance), 0.0):
            mismatches.append({"surface": "fills", "key": order_id, "reason": "filled_quantity_mismatch", "local": local_qty, "broker": broker_qty})

    local_positions = _rows_by_key(local.get("positions"), "symbol")
    broker_positions = _rows_by_key(broker.get("positions"), "symbol")
    for symbol in sorted(set(local_positions) | set(broker_positions)):
        local_qty = float((local_positions.get(symbol) or {}).get("quantity", 0.0) or 0.0)
        broker_qty = float((broker_positions.get(symbol) or {}).get("quantity", 0.0) or 0.0)
        if abs(local_qty - broker_qty) > max(float(quantity_tolerance), 0.0):
            mismatches.append({"surface": "positions", "key": symbol, "reason": "position_quantity_mismatch", "local": local_qty, "broker": broker_qty})

    local_buying_power = float(local.get("buying_power", 0.0) or 0.0)
    broker_buying_power = float(broker.get("buying_power", 0.0) or 0.0)
    if abs(local_buying_power - broker_buying_power) > max(float(buying_power_tolerance), 0.0):
        mismatches.append(
            {
                "surface": "buying_power",
                "key": "account",
                "reason": "buying_power_mismatch",
                "local": local_buying_power,
                "broker": broker_buying_power,
            }
        )

    local_cancels = _rows_by_key(local.get("cancels"), "order_id")
    broker_cancels = _rows_by_key(broker.get("cancels"), "order_id")
    for order_id in sorted(set(local_cancels) | set(broker_cancels)):
        local_status = str((local_cancels.get(order_id) or {}).get("status") or "").strip().lower()
        broker_status = str((broker_cancels.get(order_id) or {}).get("status") or "").strip().lower()
        if local_status != broker_status:
            mismatches.append({"surface": "cancels", "key": order_id, "reason": "cancel_status_mismatch", "local": local_status, "broker": broker_status})

    by_surface: dict[str, int] = {}
    for row in mismatches:
        surface = str(row.get("surface") or "unknown")
        by_surface[surface] = by_surface.get(surface, 0) + 1
    return {
        "ok": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_count_by_surface": by_surface,
        "mismatches": mismatches,
        "surfaces_checked": ["orders", "fills", "positions", "buying_power", "cancels"],
    }


def canary_stage_contract(
    *,
    requested_weight: float,
    clean_evidence_windows: int,
    sleeve_count: int,
    open_position_count: int,
) -> dict[str, Any]:
    stages = (
        {"stage": "micro_0_25pct", "max_weight": 0.0025, "required_clean_windows": 1},
        {"stage": "micro_0_50pct", "max_weight": 0.005, "required_clean_windows": 3},
        {"stage": "micro_1pct_cap", "max_weight": 0.01, "required_clean_windows": 7},
    )
    requested = max(float(requested_weight or 0.0), 0.0)
    selected = next((row for row in stages if requested <= float(row["max_weight"])), None)
    blockers: list[str] = []
    if selected is None or requested <= 0.0:
        blockers.append("requested_weight_outside_microscopic_stages")
    if int(sleeve_count) != 1:
        blockers.append("initial_canary_requires_exactly_one_sleeve")
    if int(open_position_count) > 1:
        blockers.append("initial_canary_allows_at_most_one_open_position")
    required_windows = int((selected or {}).get("required_clean_windows", 1))
    if int(clean_evidence_windows) < required_windows:
        blockers.append("clean_evidence_windows_pending")
    return {
        "ok": not blockers,
        "requested_weight": requested,
        "selected_stage": dict(selected or {}),
        "clean_evidence_windows": int(clean_evidence_windows),
        "sleeve_count": int(sleeve_count),
        "open_position_count": int(open_position_count),
        "hard_max_weight": 0.01,
        "stages": [dict(row) for row in stages],
        "blockers": blockers,
        "automatic_scaling_allowed": False,
        "operator_release_required_for_each_stage": True,
    }
