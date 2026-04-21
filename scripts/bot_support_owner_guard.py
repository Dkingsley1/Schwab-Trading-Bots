#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNERSHIP_PATH = PROJECT_ROOT / "governance" / "ownership" / "bot_support_owners.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_support_owner_guard_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_owner(bot_id: str, ownership_payload: dict[str, Any]) -> tuple[str, str]:
    exact = ownership_payload.get("owners_by_bot_id") if isinstance(ownership_payload.get("owners_by_bot_id"), dict) else {}
    prefixes = ownership_payload.get("owners_by_prefix") if isinstance(ownership_payload.get("owners_by_prefix"), dict) else {}

    normalized_id = str(bot_id or "").strip().lower()
    exact_owner = _normalized_text(exact.get(normalized_id) or exact.get(bot_id))
    if exact_owner:
        return exact_owner, "owners_by_bot_id"

    best_prefix = ""
    best_owner = ""
    for raw_prefix, raw_owner in prefixes.items():
        prefix = _normalized_text(raw_prefix).lower()
        owner = _normalized_text(raw_owner)
        if not prefix or not owner:
            continue
        if normalized_id.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_owner = owner
    if best_owner:
        return best_owner, f"owners_by_prefix:{best_prefix}"

    default_owner = _normalized_text(ownership_payload.get("default_owner"))
    if default_owner:
        return default_owner, "default_owner"
    return "", ""


def _in_scope(row: dict[str, Any]) -> bool:
    lifecycle_state = _normalized_text(row.get("lifecycle_state")).lower()
    if lifecycle_state in {"retired", "deleted", "deactivated"}:
        return False
    if bool(row.get("deleted_from_rotation", False)):
        return False
    return bool(row.get("active", False)) or lifecycle_state == "probation"


def build_payload(
    *,
    registry: dict[str, Any],
    ownership_payload: dict[str, Any],
) -> dict[str, Any]:
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    team_roster = ownership_payload.get("team_roster") if isinstance(ownership_payload.get("team_roster"), dict) else {}
    exact = ownership_payload.get("owners_by_bot_id") if isinstance(ownership_payload.get("owners_by_bot_id"), dict) else {}
    prefixes = ownership_payload.get("owners_by_prefix") if isinstance(ownership_payload.get("owners_by_prefix"), dict) else {}
    default_owner = _normalized_text(ownership_payload.get("default_owner"))

    rows: list[dict[str, Any]] = []
    missing_owner_count = 0
    invalid_owner_count = 0
    covered_count = 0

    for raw_row in sub_bots:
        if not isinstance(raw_row, dict) or not _in_scope(raw_row):
            continue
        bot_id = _normalized_text(raw_row.get("bot_id"))
        if not bot_id:
            continue
        owner, owner_source = _resolve_owner(bot_id, ownership_payload)
        roster_entry = team_roster.get(owner) if isinstance(team_roster.get(owner), dict) else {}
        owner_in_roster = bool(owner) and (not team_roster or bool(roster_entry))
        failed_contracts: list[str] = []
        if not owner:
            failed_contracts.append("support_owner_missing")
            missing_owner_count += 1
        elif not owner_in_roster:
            failed_contracts.append("support_owner_not_in_team_roster")
            invalid_owner_count += 1
        else:
            covered_count += 1
        rows.append(
            {
                "bot_id": bot_id,
                "active": bool(raw_row.get("active", False)),
                "lifecycle_state": lifecycle_state if (lifecycle_state := _normalized_text(raw_row.get("lifecycle_state")).lower()) else "active",
                "bot_role": _normalized_text(raw_row.get("bot_role") or raw_row.get("role")),
                "support_owner": owner,
                "support_owner_source": owner_source,
                "owner_roster_found": owner_in_roster,
                "failed_contracts": failed_contracts,
            }
        )

    contract_ready = bool(
        ownership_payload.get("schema_version") is not None
        and (bool(exact) or bool(prefixes) or bool(default_owner))
    )
    blocking_rows = [row for row in rows if row.get("failed_contracts")]
    ok = bool(contract_ready and not blocking_rows)

    top_actions: list[str] = []
    if not contract_ready:
        top_actions.append("populate governance/ownership/bot_support_owners.json with schema_version and named owner mappings")
    if missing_owner_count:
        top_actions.append("assign a named support owner to every active or probation bot before it is eligible for retrain or promotion")
    if invalid_owner_count:
        top_actions.append("fix support-owner names so they resolve to team_roster entries instead of orphaned aliases")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "contract_ready": contract_ready,
        "summary": {
            "in_scope_bot_count": len(rows),
            "covered_bot_count": covered_count,
            "missing_owner_count": missing_owner_count,
            "invalid_owner_count": invalid_owner_count,
            "configured_exact_owners": len(exact),
            "configured_prefix_owners": len(prefixes),
            "default_owner_present": bool(default_owner),
            "team_roster_count": len(team_roster),
        },
        "blocking_bot_count": len(blocking_rows),
        "blocking_bots": blocking_rows[:40],
        "bots": rows[:80],
        "top_actions": top_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Require named support owners for active and probation bots.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--ownership-file", default=str(DEFAULT_OWNERSHIP_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        registry=_load_json(Path(args.registry)),
        ownership_payload=_load_json(Path(args.ownership_file)),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary", {})
        print(
            "bot_support_owner_guard "
            f"ok={str(payload['ok']).lower()} "
            f"in_scope={int(summary.get('in_scope_bot_count', 0) or 0)} "
            f"blocking={int(payload.get('blocking_bot_count', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
