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
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "account_policy_context_latest.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "account_policy_registry.json"

DEFAULT_ACCOUNT_SLOTS = [
    {
        "account_policy_key": "schwab_roth_ira_primary",
        "account_label": "Roth IRA",
        "account_type": "roth",
        "tax_treatment": "tax_advantaged",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_ACCOUNT_HASH",
            "SCHWAB_ROTH_ACCOUNT_HASH",
            "SCHWAB_ROTH_IRA_ACCOUNT_HASH",
            "SCHWAB_ROTH_ACCOUNT_NUMBER",
            "SCHWAB_ROTH_IRA_ACCOUNT_NUMBER",
        ],
    },
    {
        "account_policy_key": "schwab_cash_account_1",
        "account_label": "Cash Account 1",
        "account_type": "cash",
        "tax_treatment": "taxable",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_CASH_ACCOUNT_1_HASH",
            "SCHWAB_TAXABLE_ACCOUNT_1_HASH",
            "SCHWAB_CASH_ACCOUNT_1_NUMBER",
            "SCHWAB_TAXABLE_ACCOUNT_1_NUMBER",
        ],
    },
    {
        "account_policy_key": "schwab_cash_account_2",
        "account_label": "Cash Account 2",
        "account_type": "cash",
        "tax_treatment": "taxable",
        "broker": "schwab",
        "env_names": [
            "SCHWAB_CASH_ACCOUNT_2_HASH",
            "SCHWAB_TAXABLE_ACCOUNT_2_HASH",
            "SCHWAB_CASH_ACCOUNT_2_NUMBER",
            "SCHWAB_TAXABLE_ACCOUNT_2_NUMBER",
        ],
    },
]


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _slot_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    env_names = [str(item) for item in _as_list(raw.get("env_names") or raw.get("env_bindings")) if str(item)]
    env_bindings = []
    for raw_name in env_names:
        name = str(raw_name.get("name") if isinstance(raw_name, dict) else raw_name).strip()
        if not name:
            continue
        env_bindings.append({"name": name, "present": bool(os.environ.get(name))})
    return {
        "account_policy_key": str(raw.get("account_policy_key") or raw.get("key") or "unknown_account"),
        "account_label": str(raw.get("account_label") or raw.get("label") or "Unknown Account"),
        "account_type": str(raw.get("account_type") or "unknown"),
        "tax_treatment": str(raw.get("tax_treatment") or "unknown"),
        "broker": str(raw.get("broker") or "unknown"),
        "env_bindings": env_bindings,
        "bot_visible": bool(raw.get("bot_visible", True)),
        "auto_order_enabled": bool(raw.get("auto_order_enabled", False)),
        "requires_operator_confirmation": bool(raw.get("requires_operator_confirmation", True)),
    }


def _load_registry_slots(registry_path: Path) -> tuple[list[dict[str, Any]], bool]:
    registry = load_json(registry_path)
    rows = _as_list(registry.get("account_slots") or registry.get("configured_account_slots"))
    if rows:
        return [_slot_from_raw(row) for row in rows if isinstance(row, dict)], True
    return [_slot_from_raw(row) for row in DEFAULT_ACCOUNT_SLOTS], False


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    slots, registry_present = _load_registry_slots(registry_path)
    roth_slots = sum(1 for row in slots if row.get("account_type") == "roth")
    cash_slots = sum(1 for row in slots if row.get("account_type") == "cash")
    auto_order_enabled = any(bool(row.get("auto_order_enabled", False)) for row in slots)
    missing_bindings = [
        {
            "account_policy_key": row.get("account_policy_key"),
            "env_name": binding.get("name"),
        }
        for row in slots
        for binding in _as_list(row.get("env_bindings"))
        if isinstance(binding, dict) and not bool(binding.get("present", False))
    ]
    next_actions = []
    if missing_bindings:
        next_actions.append("set account hash environment variables when live-micro account binding is intentionally approved")
    if auto_order_enabled:
        next_actions.append("turn off account auto-order flags before running any paper-to-live review")
    overall_status = "ready" if slots and not auto_order_enabled else "blocked"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "account_policy_context": {
            "schema_version": 1,
            "registry_path": str(registry_path),
            "registry_present": registry_present,
            "configured_account_slots": slots,
            "slot_count": len(slots),
            "unmatched_schwab_cash_fallback": {
                "enabled": True,
                "requires_anchor_match": True,
                "account_label_prefix": "Cash Account",
            },
            "redaction_contract": {
                "account_numbers_exposed_in_policy": False,
                "account_hashes_exposed_in_policy": False,
                "bot_context_key": "bot_visible_account_context",
                "auto_order_enabled_default": False,
                "operator_confirmation_default": True,
            },
        },
        "coverage": {
            "roth_slots": roth_slots,
            "cash_slots": cash_slots,
            "configured_account_slots": len(slots),
            "target_roth_slots": 1,
            "target_cash_slots": 2,
        },
        "bot_contract": {
            "bots_should_read": "bot_visible_account_context",
            "raw_account_numbers_required_for_policy": False,
            "raw_account_hashes_required_for_policy": False,
            "auto_order_enabled": auto_order_enabled,
            "operator_confirmation_required": True,
            "supported_account_types": sorted({str(row.get("account_type") or "unknown") for row in slots} | {"unknown"}),
        },
        "missing_env_bindings": missing_bindings[:40],
        "next_actions": next_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh redacted account policy context for bots and income-readiness controls.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    registry_path = Path(args.registry_path).expanduser()
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    payload = build_payload(project_root, registry_path=registry_path)
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "account_policy_context "
            f"status={payload.get('overall_status')} "
            f"slots={payload.get('coverage', {}).get('configured_account_slots')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
