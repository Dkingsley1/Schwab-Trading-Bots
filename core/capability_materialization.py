"""Source-backed materialization for high-priority collector capabilities."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

from core.collector_capability_routing import EXPECTED_SAFETY_FLAGS, canonical_hash


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _ordered_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _timestamp(value: Any) -> str:
    if value is None:
        return ""
    if str(value) in {"NaT", "NaN", "nan"}:
        return ""
    try:
        if bool(value is not value):
            return ""
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return str(value)
    return str(value)


def validate_materialization_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if _safe_int(policy.get("schema_version")) != 1:
        errors.append("capability_materialization_schema_version_invalid")
    if str(policy.get("policy_id") or "") != "capability_materialization_v1":
        errors.append("capability_materialization_policy_id_invalid")
    if str(policy.get("operating_mode") or "") != "source_backed_shadow_only":
        errors.append("capability_materialization_not_shadow_only")

    authority = _as_dict(policy.get("authority_contract"))
    for key in EXPECTED_SAFETY_FLAGS:
        if authority.get(key) is not False:
            errors.append(f"capability_materialization_authority_{key}_must_be_false")

    calendar_contract = _as_dict(policy.get("calendar_contract"))
    calendar_ids = [
        str(_as_dict(row).get("calendar_id") or "").strip()
        for row in _as_list(calendar_contract.get("calendars"))
    ]
    if not calendar_ids or any(not item for item in calendar_ids):
        errors.append("capability_materialization_calendars_missing")
    if len(calendar_ids) != len(set(calendar_ids)):
        errors.append("capability_materialization_calendar_ids_duplicate")
    if _safe_int(calendar_contract.get("lookahead_days"), 0) < 14:
        errors.append("capability_materialization_calendar_horizon_too_short")

    declared = [
        str(_as_dict(row).get("capability_id") or "").strip()
        for row in _as_list(policy.get("materialized_capabilities"))
    ]
    required = {
        "trading_calendars",
        "market_session_state",
        "derivatives_contract_master",
        "stress_scenarios",
    }
    if set(declared) != required:
        errors.append("capability_materialization_required_capabilities_incomplete")
    if len(declared) != len(set(declared)):
        errors.append("capability_materialization_capabilities_duplicate")
    if not str(policy.get("derivative_contract_master_path") or "").strip():
        errors.append("capability_materialization_derivative_master_missing")
    if not _as_list(policy.get("stress_scenario_paths")):
        errors.append("capability_materialization_stress_scenarios_missing")
    return _ordered_unique(errors)


def _calendar_materialization(
    policy: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[dict[str, Any], list[str]]:
    contract = _as_dict(policy.get("calendar_contract"))
    minimum_version = str(contract.get("minimum_version") or "0")
    errors: list[str] = []
    try:
        import exchange_calendars as exchange_calendars
        import pandas as pd

        package_version = metadata.version("exchange-calendars")
    except Exception as exc:
        return {
            "library": "exchange-calendars",
            "library_version": "",
            "library_ready": False,
            "calendar_count": 0,
            "session_state_count": 0,
            "calendars": [],
        }, [f"exchange_calendars_unavailable:{type(exc).__name__}"]

    try:
        library_ready = Version(package_version) >= Version(minimum_version)
    except InvalidVersion:
        library_ready = False
    if not library_ready:
        errors.append(f"exchange_calendars_version_below_floor:{package_version}<{minimum_version}")

    lookback_days = max(_safe_int(contract.get("lookback_days"), 7), 1)
    lookahead_days = max(_safe_int(contract.get("lookahead_days"), 120), 14)
    preview_count = min(max(_safe_int(contract.get("preview_session_count"), 8), 1), 16)
    start_label = pd.Timestamp((now - timedelta(days=lookback_days)).date())
    end_label = pd.Timestamp((now + timedelta(days=lookahead_days)).date())
    now_minute = pd.Timestamp(now).floor("min")
    calendar_rows: list[dict[str, Any]] = []

    for raw_calendar in _as_list(contract.get("calendars")):
        configured = _as_dict(raw_calendar)
        calendar_id = str(configured.get("calendar_id") or "").strip()
        required = bool(configured.get("required", False))
        row: dict[str, Any] = {
            "calendar_id": calendar_id,
            "market_id": str(configured.get("market_id") or ""),
            "asset_classes": _ordered_unique(_as_list(configured.get("asset_classes"))),
            "required": required,
            "ready": False,
            "session_state": "unknown",
            "current_session": "",
            "previous_close_utc": "",
            "next_open_utc": "",
            "next_close_utc": "",
            "session_count": 0,
            "schedule_preview": [],
            "schedule_receipt_sha256": "",
            "error": "",
        }
        try:
            calendar = exchange_calendars.get_calendar(calendar_id)
            schedule = calendar.schedule.loc[start_label:end_label]
            preview_rows: list[dict[str, Any]] = []
            for session_label, values in schedule.iterrows():
                preview_rows.append(
                    {
                        "session": str(session_label.date()),
                        "open_utc": _timestamp(values.get("open")),
                        "break_start_utc": _timestamp(values.get("break_start")),
                        "break_end_utc": _timestamp(values.get("break_end")),
                        "close_utc": _timestamp(values.get("close")),
                    }
                )

            is_open = bool(calendar.is_open_on_minute(now_minute, ignore_breaks=False))
            prior = [item for item in preview_rows if item["close_utc"] and item["close_utc"] <= now.isoformat()]
            upcoming = [item for item in preview_rows if item["open_utc"] and item["open_utc"] > now.isoformat()]
            active = [
                item
                for item in preview_rows
                if item["open_utc"] <= now.isoformat() < item["close_utc"]
            ]
            preview_start = 0
            if upcoming:
                preview_start = max(preview_rows.index(upcoming[0]) - 1, 0)
            elif active:
                preview_start = max(preview_rows.index(active[0]) - 1, 0)
            preview = preview_rows[preview_start : preview_start + preview_count]
            row.update(
                {
                    "ready": bool(library_ready and preview_rows),
                    "session_state": "open" if is_open else "closed",
                    "current_session": active[0]["session"] if active else "",
                    "previous_close_utc": prior[-1]["close_utc"] if prior else "",
                    "next_open_utc": upcoming[0]["open_utc"] if upcoming else "",
                    "next_close_utc": (
                        active[0]["close_utc"]
                        if active
                        else (upcoming[0]["close_utc"] if upcoming else "")
                    ),
                    "session_count": len(preview_rows),
                    "schedule_preview": preview,
                    "schedule_receipt_sha256": canonical_hash(preview_rows),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}:{exc}"
            if required:
                errors.append(f"required_calendar_failed:{calendar_id}")
        if required and not bool(row.get("ready", False)):
            errors.append(f"required_calendar_not_ready:{calendar_id}")
        calendar_rows.append(row)

    ready_count = sum(1 for row in calendar_rows if row.get("ready") is True)
    return {
        "library": "exchange-calendars",
        "library_version": package_version,
        "minimum_library_version": minimum_version,
        "library_ready": library_ready,
        "calendar_count": len(calendar_rows),
        "ready_calendar_count": ready_count,
        "session_state_count": sum(
            1 for row in calendar_rows if row.get("session_state") in {"open", "closed"}
        ),
        "lookback_days": lookback_days,
        "lookahead_days": lookahead_days,
        "calendars": calendar_rows,
    }, _ordered_unique(errors)


def _derivative_materialization(
    derivative_master: Mapping[str, Any],
    *,
    source_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows = [_as_dict(row) for row in _as_list(derivative_master.get("contracts"))]
    roots = [str(row.get("root") or "").strip().upper() for row in rows]
    if _safe_int(derivative_master.get("schema_version")) != 1:
        errors.append("derivative_contract_master_schema_invalid")
    if any(not root for root in roots) or len(roots) != len(set(roots)):
        errors.append("derivative_contract_master_roots_invalid")
    source_references = _as_dict(derivative_master.get("source_references"))
    valid_roots: list[str] = []
    contract_rows: list[dict[str, Any]] = []
    for row, root in zip(rows, roots, strict=True):
        multiplier = _safe_float(row.get("contract_multiplier"), 0.0)
        tick = _safe_float(row.get("minimum_tick"), 0.0)
        tick_value = _safe_float(row.get("tick_value"), 0.0)
        source_reference = str(row.get("source_reference") or "")
        row_errors: list[str] = []
        if multiplier <= 0.0 or tick <= 0.0:
            row_errors.append("non_positive_multiplier_or_tick")
        if not math.isclose(multiplier * tick, tick_value, rel_tol=1e-9, abs_tol=1e-9):
            row_errors.append("tick_value_mismatch")
        if not str(row.get("calendar_id") or ""):
            row_errors.append("calendar_mapping_missing")
        if not source_reference or not str(source_references.get(source_reference) or ""):
            row_errors.append("source_reference_missing")
        if not row_errors:
            valid_roots.append(root)
        else:
            errors.extend(f"contract_{root or 'missing'}:{item}" for item in row_errors)
        contract_rows.append({**row, "root": root, "ready": not row_errors, "errors": row_errors})

    required_roots = set(_ordered_unique(_as_list(derivative_master.get("required_runtime_roots"))))
    missing_required = sorted(required_roots - set(valid_roots))
    if missing_required:
        errors.append(f"required_derivative_roots_unresolved:{','.join(missing_required)}")
    config_receipt = _file_sha256(source_path) if source_path.is_file() else ""
    if not config_receipt:
        errors.append("derivative_contract_master_receipt_missing")
    return {
        "master_id": str(derivative_master.get("master_id") or ""),
        "as_of_date": str(derivative_master.get("as_of_date") or ""),
        "contract_count": len(contract_rows),
        "valid_contract_count": len(valid_roots),
        "required_runtime_roots": sorted(required_roots),
        "resolved_required_roots": sorted(required_roots & set(valid_roots)),
        "missing_required_roots": missing_required,
        "contracts": contract_rows,
        "source_references": source_references,
        "config_path": str(source_path),
        "config_receipt_sha256": config_receipt,
    }, _ordered_unique(errors)


def _stress_materialization(
    project_root: Path,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    for raw_path in _as_list(policy.get("stress_scenario_paths")):
        relative_path = str(raw_path or "").strip()
        path = project_root / relative_path
        payload = _load_json(path)
        scenario_id = str(payload.get("scenario_id") or "").strip()
        source = payload.get("source")
        source_present = bool(source)
        replay_contract_present = bool(
            _as_dict(payload.get("replay_contract"))
            or _as_dict(payload.get("usage_policy"))
            or _as_list(payload.get("stress_modules"))
            or (
                _as_dict(payload.get("scope"))
                and (
                    _as_dict(payload.get("key_stress_anchors"))
                    or _as_list(payload.get("domestic_variables"))
                    or _as_list(payload.get("international_variables"))
                )
            )
        )
        receipt = _file_sha256(path) if path.is_file() else ""
        row_errors: list[str] = []
        if not payload:
            row_errors.append("payload_missing")
        if not scenario_id:
            row_errors.append("scenario_id_missing")
        if not source_present:
            row_errors.append("source_missing")
        if not replay_contract_present:
            row_errors.append("replay_or_usage_contract_missing")
        if not receipt:
            row_errors.append("content_receipt_missing")
        if scenario_id:
            scenario_ids.append(scenario_id)
        if row_errors:
            errors.extend(f"stress_scenario_{relative_path}:{item}" for item in row_errors)
        rows.append(
            {
                "path": relative_path,
                "scenario_id": scenario_id,
                "component_kind": (
                    "stress_modules"
                    if _as_list(payload.get("stress_modules"))
                    else ("source_plumbing" if payload.get("plumbing_id") else "scenario_definition")
                ),
                "ready": not row_errors,
                "errors": row_errors,
                "source": source,
                "content_receipt_sha256": receipt,
            }
        )
    return {
        "component_count": len(rows),
        "ready_component_count": sum(1 for row in rows if row["ready"]),
        "scenario_count": len(set(scenario_ids)),
        "scenario_ids": sorted(set(scenario_ids)),
        "components": rows,
    }, _ordered_unique(errors)


def _capability_row(
    capability_id: str,
    *,
    proof_family: str,
    evidence_count: int,
    minimum_evidence_count: int,
    scope: list[str],
    source_receipts: list[dict[str, Any]],
    proof_payload: Any,
    errors: list[str],
    now: datetime,
) -> dict[str, Any]:
    usable = bool(not errors and evidence_count >= minimum_evidence_count and source_receipts and scope)
    return {
        "capability_id": capability_id,
        "status": "ready" if usable else "blocked",
        "usable": usable,
        "proof_semantics": "direct",
        "proof_family": proof_family,
        "evidence_count": evidence_count,
        "minimum_evidence_count": minimum_evidence_count,
        "point_in_time_timestamp_utc": now.isoformat(),
        "scope": scope,
        "source_receipts": source_receipts,
        "proof_receipt_sha256": canonical_hash(proof_payload),
        "errors": errors,
    }


def build_materialized_capabilities(
    project_root: Path,
    policy: Mapping[str, Any],
    derivative_master: Mapping[str, Any],
    *,
    derivative_master_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    policy_errors = validate_materialization_policy(policy)
    calendar, calendar_errors = _calendar_materialization(policy, now=current)
    derivatives, derivative_errors = _derivative_materialization(
        derivative_master,
        source_path=derivative_master_path,
    )
    stress, stress_errors = _stress_materialization(project_root, policy)
    specs = {
        str(_as_dict(row).get("capability_id") or ""): _as_dict(row)
        for row in _as_list(policy.get("materialized_capabilities"))
    }

    calendar_receipts = [
        {
            "source": f"exchange_calendars:{row.get('calendar_id')}",
            "version": calendar.get("library_version"),
            "sha256": row.get("schedule_receipt_sha256"),
        }
        for row in _as_list(calendar.get("calendars"))
        if row.get("schedule_receipt_sha256")
    ]
    derivative_receipts = [
        {
            "source": "derivative_contract_master_config",
            "path": str(derivative_master_path),
            "sha256": derivatives.get("config_receipt_sha256"),
        }
    ] if derivatives.get("config_receipt_sha256") else []
    stress_receipts = [
        {
            "source": str(row.get("scenario_id") or row.get("path") or "stress_scenario"),
            "path": row.get("path"),
            "sha256": row.get("content_receipt_sha256"),
        }
        for row in _as_list(stress.get("components"))
        if row.get("content_receipt_sha256")
    ]

    rows = [
        _capability_row(
            "trading_calendars",
            proof_family=str(specs["trading_calendars"].get("proof_family") or ""),
            evidence_count=_safe_int(calendar.get("ready_calendar_count")),
            minimum_evidence_count=_safe_int(specs["trading_calendars"].get("minimum_evidence_count")),
            scope=["EQUITY", "ETF", "OPTION", "FUTURE", "FUTURE_OPTION", "CRYPTO"],
            source_receipts=calendar_receipts,
            proof_payload=calendar,
            errors=calendar_errors,
            now=current,
        ),
        _capability_row(
            "market_session_state",
            proof_family=str(specs["market_session_state"].get("proof_family") or ""),
            evidence_count=_safe_int(calendar.get("session_state_count")),
            minimum_evidence_count=_safe_int(specs["market_session_state"].get("minimum_evidence_count")),
            scope=["EQUITY", "ETF", "OPTION", "FUTURE", "FUTURE_OPTION", "CRYPTO"],
            source_receipts=calendar_receipts,
            proof_payload=[
                {
                    "calendar_id": row.get("calendar_id"),
                    "session_state": row.get("session_state"),
                    "current_session": row.get("current_session"),
                    "next_open_utc": row.get("next_open_utc"),
                    "next_close_utc": row.get("next_close_utc"),
                }
                for row in _as_list(calendar.get("calendars"))
            ],
            errors=calendar_errors,
            now=current,
        ),
        _capability_row(
            "derivatives_contract_master",
            proof_family=str(specs["derivatives_contract_master"].get("proof_family") or ""),
            evidence_count=_safe_int(derivatives.get("valid_contract_count")),
            minimum_evidence_count=_safe_int(
                specs["derivatives_contract_master"].get("minimum_evidence_count")
            ),
            scope=["FUTURE", "FUTURE_OPTION"],
            source_receipts=derivative_receipts,
            proof_payload=derivatives,
            errors=derivative_errors,
            now=current,
        ),
        _capability_row(
            "stress_scenarios",
            proof_family=str(specs["stress_scenarios"].get("proof_family") or ""),
            evidence_count=_safe_int(stress.get("scenario_count")),
            minimum_evidence_count=_safe_int(specs["stress_scenarios"].get("minimum_evidence_count")),
            scope=["cross_asset", "portfolio_risk", "research_replay"],
            source_receipts=stress_receipts,
            proof_payload=stress,
            errors=stress_errors,
            now=current,
        ),
    ]
    ready_ids = [str(row["capability_id"]) for row in rows if row["usable"]]
    all_errors = _ordered_unique(policy_errors + calendar_errors + derivative_errors + stress_errors)
    structural_ok = not policy_errors
    all_ready = len(ready_ids) == len(rows)
    authority = {key: False for key in EXPECTED_SAFETY_FLAGS}
    payload: dict[str, Any] = {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": structural_ok,
        "overall_status": "ready" if structural_ok and all_ready else ("degraded" if structural_ok else "blocked"),
        "paper_soak_ready": structural_ok,
        "live_promotion_ready": bool(structural_ok and all_ready),
        "capability_count": len(rows),
        "ready_capability_count": len(ready_ids),
        "ready_capability_ids": ready_ids,
        "blocked_capability_ids": [str(row["capability_id"]) for row in rows if not row["usable"]],
        "proof_coverage_ratio": round(len(ready_ids) / len(rows), 6) if rows else 0.0,
        "capabilities": rows,
        "calendar_materialization": calendar,
        "derivative_contract_materialization": derivatives,
        "stress_scenario_materialization": stress,
        "errors": all_errors,
        "authority_contract": authority,
        "control_contract": {
            "capability_level_proof_required": True,
            "point_in_time_state_published": True,
            "content_addressed_source_receipts": True,
            "partial_materialization_fails_capability_closed": True,
            "network_fetches": False,
            "runtime_decision_changes": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
    }
    payload["materialization_receipt_sha256"] = canonical_hash(
        {
            "policy_id": payload["policy_id"],
            "capabilities": rows,
            "authority_contract": authority,
        }
    )
    return payload
