"""Evidence-scoped reporting labels, without admission or execution authority."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_label_source(root: Path, relative_path: str) -> dict[str, Any]:
    from core.storage_router import inspect_storage_path

    route = inspect_storage_path(
        root / relative_path, boundary_root=root, allow_external=False
    )
    if route["status"] == "missing":
        return {}
    if route["status"] != "present":
        return {
            "_label_source_error": "route_rejected",
            "route_status": route["status"],
        }
    try:
        with os.fdopen(
            os.open(route["resolved_path"], os.O_RDONLY | os.O_NOFOLLOW), "rb"
        ) as handle:
            raw = handle.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            return {"_label_source_error": "size_limit_exceeded"}
        payload = json.loads(raw)
        return (
            payload
            if isinstance(payload, dict)
            else {"_label_source_error": "invalid_payload"}
        )
    except OSError:
        return {"_label_source_error": "unreadable"}
    except ValueError:
        return {"_label_source_error": "invalid_payload"}


def read_paper_hold_labels(root: Path) -> dict[str, Any]:
    return paper_hold_labels(
        *[
            read_label_source(root, f"governance/health/{name}_latest.json")
            for name in (
                "execution_lane_paper",
                "runtime_throttle_control",
                "local_storage_reserve_guard",
            )
        ]
    )


def evidence_label(
    payload: dict[str, Any],
    *,
    scope: str,
    source: str,
    max_age_seconds: float | None,
    now: datetime | None = None,
    reported_state: str | None = None,
) -> dict[str, Any]:
    """Retain the producer verdict, but never present old/unknown evidence as current."""
    now = now or datetime.now(timezone.utc)
    state = (
        str(
            reported_state
            or payload.get("overall_status")
            or payload.get("status")
            or ""
        )
        .strip()
        .lower()
    )
    if not state:
        state = (
            "producer_completed"
            if payload.get("ok") is True
            else "producer_not_ok" if payload.get("ok") is False else "unknown"
        )
    stamp = None
    field = ""
    freshness = "missing" if not payload else "timestamp_missing"
    for key in (
        "source_timestamp_utc",
        "observation_timestamp_utc",
        "measurement_timestamp_utc",
        "timestamp_utc",
        "generated_at_utc",
        "updated_at_utc",
    ):
        if key not in payload:
            continue
        field = key
        try:
            stamp = datetime.fromisoformat(str(payload[key]).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("naive_timestamp")
            stamp = stamp.astimezone(timezone.utc)
        except (TypeError, ValueError):
            freshness = "timestamp_invalid"
            stamp = None
        break
    age = (now - stamp).total_seconds() if stamp is not None else None
    if age is not None:
        freshness = (
            "future"
            if age < 0
            else (
                "age_budget_unspecified"
                if max_age_seconds is None
                else "stale" if age > max_age_seconds else "fresh"
            )
        )
    if field != "timestamp_utc" and "timestamp_utc" in payload:
        try:
            producer = datetime.fromisoformat(
                str(payload["timestamp_utc"]).replace("Z", "+00:00")
            )
            if producer.tzinfo is None:
                raise ValueError("naive_producer_timestamp")
            if producer > now:
                freshness = "future"
            elif stamp is not None and stamp > producer:
                freshness = "timestamp_inconsistent"
        except (TypeError, ValueError):
            freshness = "timestamp_invalid"
    if payload.get("_label_source_error"):
        freshness = str(payload["_label_source_error"])
    verdicts = {
        key: str(payload[key]).strip().lower()
        for key in ("overall_status", "status")
        if payload.get(key)
    }
    values = set(verdicts.values())
    conflict = bool(
        values & {"ready", "ok", "healthy"}
        and values & {"blocked", "error", "failed", "fail", "regressed", "degraded"}
    )
    status = (
        ("conflicting_status" if conflict else state)
        if freshness == "fresh"
        else f"evidence_{freshness}"
    )
    return {
        "status": status,
        "reported_status": state,
        "producer_status_fields": verdicts,
        "status_conflict": conflict,
        "scope": scope,
        "source": source,
        "evidence_status": freshness,
        "timestamp_field": field,
        "observation_timestamp_utc": stamp.isoformat() if stamp is not None else None,
        "age_seconds": round(age, 3) if age is not None else None,
        "max_age_seconds": max_age_seconds,
        "fresh": freshness == "fresh",
        "display": f"{scope}: {status}"
        + (f" (last reported: {state})" if freshness != "fresh" else ""),
        "authority": "reporting_only",
    }


def paper_hold_labels(
    execution: dict[str, Any],
    throttle: dict[str, Any],
    storage: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    sources = {
        name: evidence_label(
            payload,
            scope=name,
            source=f"governance/health/{filename}",
            max_age_seconds=age,
            now=now,
        )
        for name, payload, filename, age in (
            ("execution", execution, "execution_lane_paper_latest.json", 120),
            ("runtime_policy", throttle, "runtime_throttle_control_latest.json", 180),
            ("local_storage", storage, "local_storage_reserve_guard_latest.json", 180),
        )
    }
    hold = _dict(execution.get("execution_safety_hold"))
    observed = bool(sources["execution"]["fresh"] and hold.get("active") is True)
    reasons: list[str] = []
    if sources["local_storage"]["fresh"]:
        reserve = _dict(storage.get("local_storage_reserve"))
        if reserve.get("pressure_active") is True:
            reasons.append("local_storage_reserve_pressure")
    if sources["runtime_policy"]["fresh"]:
        policy = _dict(throttle.get("paper_execution_policy"))
        if policy.get("pause_paper_execution") is True:
            if policy.get("pressure_pause_active") is True:
                reasons.append(
                    str(
                        policy.get("pressure_pause_reason")
                        or "paper_execution_cpu_pressure"
                    )
                )
            else:
                reasons.append(str(policy.get("reason") or "paper_admission_blocked"))
            blockers = policy.get("blockers")
            if isinstance(blockers, list):
                reasons.extend(
                    item for item in blockers if isinstance(item, str) and item
                )
    raw_reason = str(hold.get("reason") or "")
    if observed and not reasons:
        reasons.append(
            "runtime_hold_cause_unverified"
            if raw_reason == "paper_execution_paused_for_runtime_pressure"
            else raw_reason or "runtime_hold_cause_unverified"
        )
    breaker = _dict(execution.get("runtime_execution_breaker"))
    breaker_reasons = []
    if sources["execution"]["fresh"] and breaker.get("active") is True:
        raw = breaker.get("reasons")
        breaker_reasons = (
            [item for item in raw if isinstance(item, str)]
            if isinstance(raw, list)
            else []
        )
    return {
        "status": (
            "observed_paused"
            if observed
            else (
                "hold_not_observed"
                if sources["execution"]["fresh"]
                else "execution_evidence_unavailable"
            )
        ),
        "observed_runtime_hold": observed,
        "reported_runtime_reason": raw_reason,
        "current_policy_reasons": list(dict.fromkeys(reasons)),
        "execution_breaker_reasons": breaker_reasons,
        "source_evidence": sources,
        "policy_reasons_are_not_execution_receipts": True,
        "authority": "reporting_only_no_hold_release",
    }


def bot_definition_labels(
    bot: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Configuration and static source evidence do not prove a bot is running."""
    complete = record.get("definition_complete") is True
    process = _dict(record.get("process_definition"))
    declared_lifecycle = str(bot.get("lifecycle_state") or "unknown")
    retired = (
        bot.get("deleted") is True
        or bot.get("deleted_from_rotation") is True
        or declared_lifecycle in {"deleted", "retired", "archived"}
    )
    return {
        "registry": (
            "declared_retired"
            if retired
            else "declared_active" if bot.get("active") is True else "declared_inactive"
        ),
        "declared_lifecycle": declared_lifecycle,
        "collection": (
            "configured_enabled"
            if bot.get("data_collection_active") is True
            else (
                "configured_disabled"
                if bot.get("data_collection_active") is False
                else "not_declared"
            )
        ),
        "definition": (
            "complete" if complete else "incomplete" if record else "not_assessed"
        ),
        "definition_scope": str(
            record.get("completion_scope") or "operating_definition"
        ),
        "implementation": (
            str(process.get("implementation_kind") or "unknown")
            if complete
            else "source_binding_unverified"
        ),
        "process": (
            "defined_not_runtime_verified"
            if complete and process.get("definition_valid") is True
            else "not_verified"
        ),
        "training_labels": (
            "contract_declared_not_outcomes_verified"
            if isinstance(bot.get("label_contract"), dict)
            or isinstance(bot.get("universal_label_contract"), dict)
            else "contract_missing"
        ),
        "runtime": "not_assessed_by_definition_audit",
        "economic_evidence": "not_assessed_by_definition_audit",
        "authority": "reporting_only_no_activation_or_promotion",
    }
