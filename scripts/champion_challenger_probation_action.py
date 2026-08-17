#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_action_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = _normalized_text(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_payload(
    *,
    probation_guard: dict[str, Any],
    champion_registry: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    current_champion = champion_registry.get("champion") if isinstance(champion_registry.get("champion"), dict) else {}
    current_name = _normalized_text(current_champion.get("name"))
    rollback_candidate = _normalized_text(
        probation_guard.get("rollback_candidate")
        or current_champion.get("rollback_candidate")
    )
    failed_checks = probation_guard.get("failed_checks") if isinstance(probation_guard.get("failed_checks"), list) else []

    frozen_candidates = _dedupe_names(
        [
            *[
                _normalized_text((row or {}).get("bot_id") if isinstance(row, dict) else row)
                for row in (champion_registry.get("probation_candidates") or [])
            ],
            *[
                _normalized_text((row or {}).get("bot_id") if isinstance(row, dict) else row)
                for row in (champion_registry.get("challengers") or [])
            ],
            *[
                _normalized_text((row or {}).get("bot_id"))
                for row in (
                    probation_guard.get("monitored_candidates")
                    if isinstance(probation_guard.get("monitored_candidates"), list)
                    else []
                )
                if isinstance(row, dict)
            ],
        ]
    )

    action = "none"
    reason = ""
    if not bool(probation_guard.get("ok", False)):
        if rollback_candidate and rollback_candidate != current_name:
            action = "rollback_to_candidate"
            reason = "probation_guard_requested_rollback"
        else:
            action = "freeze_probation_promotion"
            reason = "probation_guard_requested_freeze"

    top_actions: list[str] = []
    if action == "rollback_to_candidate":
        top_actions.append(f"restore champion registry to {rollback_candidate} and freeze challenger promotion until guard recovery")
    elif action == "freeze_probation_promotion":
        top_actions.append("freeze probation and challenger promotion until paper execution and latency recover")

    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": action == "none",
        "action_required": action != "none",
        "action": action,
        "reason": reason,
        "failed_checks": failed_checks,
        "promotion_frozen": action != "none",
        "rollback_candidate": rollback_candidate,
        "current_champion": current_name,
        "frozen_candidate_ids": frozen_candidates,
        "applyable": action != "none",
        "applied": False,
        "top_actions": top_actions,
    }


def apply_action(
    *,
    payload: dict[str, Any],
    champion_registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    registry = deepcopy(champion_registry)
    registry.setdefault("history", [])
    registry.setdefault("champion", {})

    event = {
        "timestamp_utc": now,
        "action": payload.get("action"),
        "reason": payload.get("reason"),
        "failed_checks": payload.get("failed_checks") if isinstance(payload.get("failed_checks"), list) else [],
        "rollback_candidate": _normalized_text(payload.get("rollback_candidate")),
    }

    current_champion = registry.get("champion") if isinstance(registry.get("champion"), dict) else {}
    current_name = _normalized_text(current_champion.get("name"))
    rollback_candidate = _normalized_text(payload.get("rollback_candidate"))

    registry["promotion_frozen"] = bool(payload.get("promotion_frozen", False))
    registry["promotion_freeze_reason"] = event["failed_checks"]
    registry["promotion_freeze_updated_at_utc"] = now

    if payload.get("action") == "rollback_to_candidate" and rollback_candidate:
        if current_name and current_name != rollback_candidate:
            rollback_history_row = dict(current_champion)
            rollback_history_row["rolled_back_at_utc"] = now
            rollback_history_row["rolled_back_reason"] = payload.get("reason")
            registry["history"].append(rollback_history_row)
        registry["champion"] = {
            "name": rollback_candidate,
            "since_utc": now,
            "stage": "restored",
            "rollback_candidate": current_name,
            "restored_from": current_name,
        }

    frozen_candidates = payload.get("frozen_candidate_ids") if isinstance(payload.get("frozen_candidate_ids"), list) else []
    registry["frozen_candidates"] = _dedupe_names(
        [
            *[
                _normalized_text((row or {}).get("bot_id") if isinstance(row, dict) else row)
                for row in (registry.get("frozen_candidates") or [])
            ],
            *[str(item) for item in frozen_candidates],
        ]
    )
    registry["probation_candidates"] = []
    registry["challengers"] = []
    registry["last_event"] = event

    applied_payload = dict(payload)
    applied_payload["applied"] = True
    applied_payload["applied_at_utc"] = now
    applied_payload["registry_event"] = event
    return applied_payload, registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or roll back champion/challenger promotion when probation fails.")
    parser.add_argument("--probation-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"))
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "registry.json"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry_payload = _load_json(registry_path)
    payload = build_payload(
        probation_guard=_load_json(Path(args.probation_guard_file)),
        champion_registry=registry_payload,
    )

    if args.apply and payload.get("applyable", False):
        payload, registry_payload = apply_action(payload=payload, champion_registry=registry_payload)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "champion_challenger_probation_action "
            f"ok={str(payload['ok']).lower()} "
            f"action={payload.get('action', 'none')} "
            f"applied={str(payload.get('applied', False)).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
