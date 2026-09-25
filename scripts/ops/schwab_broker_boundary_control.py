#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.accountability import safe_write_json_atomic
    from scripts.ops.long_runtime_common import iso_now
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.accountability import safe_write_json_atomic
    from .long_runtime_common import iso_now


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "schwab_broker_boundary_control_latest.json"
)
DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT / "governance" / "runtime" / "schwab_broker_boundary_baseline.json"
)
DEFAULT_QUARANTINE_PATH = (
    PROJECT_ROOT / "governance" / "health" / "SCHWAB_BROKER_BOUNDARY_QUARANTINE.json"
)
DEFAULT_ALERT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "schwab_broker_boundary_alert_latest.json"
)


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def snapshot_complete(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot.get("ok", False)
        and not snapshot.get("account_snapshot_partial", False)
        and int(snapshot.get("failed_account_count", 0) or 0) == 0
        and int(snapshot.get("account_count", 0) or 0) > 0
        and int(snapshot.get("account_count", 0) or 0)
        == int(snapshot.get("discovered_account_count", 0) or 0)
    )


def _policy_slots(policy_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context = (
        policy_context.get("account_policy_context")
        if isinstance(policy_context.get("account_policy_context"), dict)
        else {}
    )
    slots = context.get("configured_account_slots")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(slots, list):
        for row in slots:
            if not isinstance(row, dict):
                continue
            key = str(row.get("account_policy_key") or "").strip()
            if key:
                out[key] = row
    return out


def build_boundary_signature(
    position_study: dict[str, Any],
    policy_context: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    slots = _policy_slots(policy_context)
    accounts: list[dict[str, Any]] = []
    study_rows = position_study.get("accounts")
    if isinstance(study_rows, list):
        for row in study_rows:
            if not isinstance(row, dict):
                continue
            truth = (
                row.get("account_capability_truth")
                if isinstance(row.get("account_capability_truth"), dict)
                else {}
            )
            operator = (
                truth.get("operator_classification")
                if isinstance(truth.get("operator_classification"), dict)
                else {}
            )
            provider = (
                truth.get("provider_account")
                if isinstance(truth.get("provider_account"), dict)
                else {}
            )
            inventory = (
                truth.get("provider_field_inventory")
                if isinstance(truth.get("provider_field_inventory"), dict)
                else {}
            )
            key = str(
                row.get("account_policy_key")
                or operator.get("account_policy_key")
                or ""
            ).strip()
            policy = slots.get(key, {})
            provider_fields = (
                provider.get("fields")
                if isinstance(provider.get("fields"), dict)
                else {}
            )
            accounts.append(
                {
                    "account_policy_key": key,
                    "provider_account_type": str(
                        provider.get("provider_account_type")
                        or row.get("account_type")
                        or ""
                    ),
                    "account_kind": str(
                        operator.get("account_kind")
                        or row.get("operator_account_kind")
                        or policy.get("account_type")
                        or ""
                    ),
                    "tax_wrapper": str(
                        operator.get("tax_wrapper") or row.get("tax_wrapper") or ""
                    ),
                    "tax_treatment": str(
                        operator.get("tax_treatment")
                        or row.get("tax_treatment")
                        or policy.get("tax_treatment")
                        or ""
                    ),
                    "trading_access": str(
                        operator.get("trading_access")
                        or row.get("operator_trading_type")
                        or policy.get("trading_access")
                        or ""
                    ),
                    "borrowing_allowed": bool(
                        operator.get(
                            "borrowing_allowed",
                            row.get(
                                "borrowing_allowed",
                                policy.get("borrowing_allowed", False),
                            ),
                        )
                    ),
                    "margin_interest_possible": bool(
                        operator.get(
                            "margin_interest_possible",
                            policy.get("margin_interest_possible", False),
                        )
                    ),
                    "option_access": str(operator.get("option_access") or "unknown"),
                    "existing_positions_authority": str(
                        operator.get("existing_positions_authority") or "observe_only"
                    ),
                    "provider_capability_flags": {
                        str(name): provider_fields.get(name)
                        for name in (
                            "type",
                            "isClosingOnlyRestricted",
                            "isDayTrader",
                            "isIntradayMargin",
                            "isPortfolioMargin",
                            "pfcbFlag",
                        )
                        if name in provider_fields
                    },
                    "provider_field_inventory": {
                        str(section): sorted(str(item) for item in fields)
                        for section, fields in sorted(inventory.items())
                        if isinstance(fields, list)
                    },
                }
            )
    accounts.sort(key=lambda row: str(row.get("account_policy_key") or ""))
    return {
        "schema_version": 1,
        "broker": "schwab",
        "snapshot_mode": str(snapshot.get("account_snapshot_mode") or ""),
        "account_count": int(snapshot.get("account_count", len(accounts)) or 0),
        "accounts": accounts,
        "raw_account_numbers_present": False,
        "raw_account_hashes_present": False,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], child))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            out.update(_flatten(item, child))
        return out
    return {prefix: value}


def compare_boundary_signatures(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    before = _flatten(baseline)
    after = _flatten(current)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        rows.append(
            {
                "path": path,
                "before": before.get(path),
                "after": after.get(path),
            }
        )
    return rows


def _live_authority_enabled(candidate_state: dict[str, Any]) -> bool:
    return bool(
        candidate_state.get("live_execution_authority", False)
        or candidate_state.get("live_execution_enabled", False)
        or candidate_state.get("live_orders_allowed", False)
    )


def build_payload(
    project_root: Path,
    *,
    apply: bool = False,
    accept_baseline: bool = False,
    reason: str = "",
    notify: bool = False,
) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    runtime = project_root / "governance" / "runtime"
    study = _read(health / "account_position_study_latest.json")
    policy = _read(health / "account_policy_context_latest.json")
    snapshot = _read(health / "schwab_account_snapshot_refresh_latest.json")
    candidate = _read(runtime / "production_candidate_state.json")
    baseline_path = runtime / "schwab_broker_boundary_baseline.json"
    quarantine_path = health / "SCHWAB_BROKER_BOUNDARY_QUARANTINE.json"
    alert_path = health / "schwab_broker_boundary_alert_latest.json"
    out_path = health / "schwab_broker_boundary_control_latest.json"
    live_authority = _live_authority_enabled(candidate)
    upstream_ready = bool(
        study.get("ok", False)
        and policy.get("ok", False)
        and snapshot_complete(snapshot)
    )
    signature = build_boundary_signature(study, policy, snapshot)
    signature_sha = _sha256(signature)
    baseline_record = _read(baseline_path)
    baseline_signature = (
        baseline_record.get("signature")
        if isinstance(baseline_record.get("signature"), dict)
        else {}
    )
    baseline_bootstrapped = False
    baseline_accepted = False
    blockers: list[str] = []
    changes: list[dict[str, Any]] = []

    if not upstream_ready:
        blockers.append("schwab_boundary_upstream_truth_incomplete")
    elif not baseline_signature:
        if apply and not live_authority:
            baseline_bootstrapped = True
            baseline_signature = signature
        else:
            blockers.append("schwab_boundary_baseline_missing")
    else:
        changes = compare_boundary_signatures(baseline_signature, signature)
        if changes:
            blockers.append("schwab_schema_or_capability_drift")

    if accept_baseline:
        if live_authority:
            blockers.append("cannot_accept_boundary_drift_while_live_authority_enabled")
        elif not str(reason or "").strip():
            blockers.append("baseline_acceptance_reason_required")
        elif upstream_ready:
            baseline_signature = signature
            changes = []
            blockers = [
                item
                for item in blockers
                if item not in {
                    "schwab_schema_or_capability_drift",
                    "schwab_boundary_baseline_missing",
                }
            ]
            baseline_accepted = True

    if apply and (baseline_bootstrapped or baseline_accepted):
        baseline_payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "broker": "schwab",
            "signature_sha256": signature_sha,
            "signature": signature,
            "acceptance": {
                "bootstrap": baseline_bootstrapped,
                "explicit": baseline_accepted,
                "reason": str(reason or "bootstrap_with_live_authority_disabled"),
                "live_execution_authority": False,
            },
        }
        safe_write_json_atomic(
            str(baseline_path),
            baseline_payload,
            project_root=str(project_root),
            source="schwab_broker_boundary_control.baseline",
        )

    quarantine_active = bool(blockers)
    quarantine = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "active": quarantine_active,
        "broker": "schwab",
        "blockers": blockers,
        "change_count": len(changes),
        "changes": changes[:100],
        "live_execution_authority": False,
        "policy": "unexpected Schwab schema or capability changes cannot grant execution authority",
    }
    alert = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "active": bool(changes),
        "severity": "critical" if changes else "info",
        "event": "schwab_account_capability_change" if changes else "schwab_boundary_stable",
        "message": (
            f"Schwab account schema/capability drift detected ({len(changes)} changes)"
            if changes
            else "Schwab account schema and capabilities match the accepted baseline"
        ),
        "change_count": len(changes),
        "changed_paths": [str(row.get("path") or "") for row in changes[:100]],
        "live_execution_authority": False,
    }
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": "ready" if not blockers else "blocked",
        "grade": "A+" if not blockers else "F",
        "broker": "schwab",
        "upstream_truth_ready": upstream_ready,
        "snapshot_complete": snapshot_complete(snapshot),
        "baseline_present": bool(baseline_signature),
        "baseline_bootstrapped": baseline_bootstrapped,
        "baseline_accepted": baseline_accepted,
        "signature_sha256": signature_sha,
        "change_count": len(changes),
        "changes": changes[:100],
        "quarantine_active": quarantine_active,
        "blockers": blockers,
        "live_execution_authority": False,
        "redaction": {
            "raw_account_numbers_emitted": False,
            "raw_account_hashes_emitted": False,
        },
    }
    if apply:
        safe_write_json_atomic(
            str(quarantine_path),
            quarantine,
            project_root=str(project_root),
            source="schwab_broker_boundary_control.quarantine",
        )
        safe_write_json_atomic(
            str(alert_path),
            alert,
            project_root=str(project_root),
            source="schwab_broker_boundary_control.alert",
        )
        if notify and changes:
            try:
                from scripts.pager_alert_router import send

                payload["notification"] = send(alert)
            except Exception as exc:
                payload["notification"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
        safe_write_json_atomic(
            str(out_path),
            payload,
            project_root=str(project_root),
            source="schwab_broker_boundary_control",
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine unreviewed Schwab account schema and capability changes."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(
        args.project_root.resolve(),
        apply=args.apply,
        accept_baseline=args.accept_baseline,
        reason=args.reason,
        notify=args.notify,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schwab_broker_boundary_control "
            f"status={payload['overall_status']} changes={payload['change_count']} "
            f"quarantine={int(payload['quarantine_active'])}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
