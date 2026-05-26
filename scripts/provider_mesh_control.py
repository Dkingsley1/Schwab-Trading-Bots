#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "provider_mesh_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _collector_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _parse_iso_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _twelve_data_cooldown(payload: dict[str, Any]) -> dict[str, Any]:
    active = False
    cooldown_until = _parse_iso_ts(payload.get("cooldown_until_utc"))
    now = datetime.now(timezone.utc)
    if cooldown_until is not None and cooldown_until > now:
        active = True
    return {
        "active": bool(active),
        "kind": str(payload.get("kind") or ""),
        "symbol": str(payload.get("symbol") or ""),
        "cooldown_until_utc": payload.get("cooldown_until_utc"),
        "remaining_seconds": max(int((cooldown_until - now).total_seconds()), 0) if cooldown_until is not None and active else 0,
        "failure_count": int(payload.get("failure_count", 0) or 0),
    }


def _group_status(*, total: int, contract_ok: int, snapshot_ready: int, degraded_ok: bool = False) -> str:
    if total <= 0:
        return "missing"
    if contract_ok >= total and snapshot_ready >= total:
        return "ready"
    if degraded_ok and snapshot_ready >= total:
        return "degraded"
    if contract_ok <= 0 and snapshot_ready <= 0:
        return "blocked"
    return "degraded"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    collector_contracts = _load_json(health_root / "collector_contracts_latest.json")
    source_verification = _load_json(health_root / "source_verification_latest.json")
    fx_guard = _load_json(health_root / "fx_twelve_data_guard_latest.json")

    rows = _collector_rows(collector_contracts)
    required_rows = [row for row in rows if bool(row.get("required", False))]
    optional_rows = [row for row in rows if not bool(row.get("required", False))]

    required_contract_ok = sum(1 for row in required_rows if bool(row.get("contract_ok", False)))
    required_snapshot_ready = sum(
        1
        for row in required_rows
        if bool(row.get("payload_present", False)) and int(row.get("payload_size_bytes", 0) or 0) > 0
    )
    optional_contract_ok = sum(1 for row in optional_rows if bool(row.get("contract_ok", False)))
    optional_snapshot_ready = sum(
        1
        for row in optional_rows
        if bool(row.get("payload_present", False)) and int(row.get("payload_size_bytes", 0) or 0) > 0
    )

    source_overall = source_verification.get("overall") if isinstance(source_verification.get("overall"), dict) else {}
    source_counts = source_overall.get("counts") if isinstance(source_overall.get("counts"), dict) else {}
    all_verified = bool(source_overall.get("all_verified", False))
    all_cross_verified = bool(source_overall.get("all_cross_verified", False))

    cooldown = _twelve_data_cooldown(fx_guard)

    required_status = _group_status(
        total=len(required_rows),
        contract_ok=required_contract_ok,
        snapshot_ready=required_snapshot_ready,
    )
    verification_status = "ready" if all_verified else ("degraded" if bool(source_overall) else "missing")
    verification_depth_status = "cross_verified" if all_cross_verified else "single_source_verified"
    quota_status = "ready"
    if cooldown["active"]:
        quota_status = "degraded" if required_snapshot_ready > 0 else "blocked"
    elif int(collector_contracts.get("soft_failure_count", 0) or 0) > 0:
        quota_status = "degraded"

    overall_status = "ready"
    if required_status == "blocked":
        overall_status = "blocked"
    elif any(status in {"degraded", "missing"} for status in (required_status, verification_status, quota_status)):
        overall_status = "degraded"

    recommended_actions = _ordered_unique(
        [
            "treat provider cooldowns as mesh-level state and serve last-good snapshots until the provider recovers" if cooldown["active"] else "",
            "raise required collector snapshot coverage so required lanes keep a usable last-good state during provider outages" if required_snapshot_ready < len(required_rows) else "",
            "repair required collector failures before trusting live context-driven decisions" if collector_contracts.get("required_failures") else "",
            "cross-verify more sources to raise optional verification depth from ready to A+"
            if all_verified and not all_cross_verified and bool(source_overall)
            else "",
            "keep optional collectors on a degraded path instead of letting them block the required context mesh" if int(collector_contracts.get("soft_failure_count", 0) or 0) > 0 else "",
        ]
    )

    average_quality_score = float(collector_contracts.get("average_quality_score", 0.0) or 0.0)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "summary": {
            "collector_count": len(rows),
            "required_collectors": len(required_rows),
            "optional_collectors": len(optional_rows),
            "required_contract_ok": required_contract_ok,
            "required_snapshot_ready": required_snapshot_ready,
            "optional_contract_ok": optional_contract_ok,
            "optional_snapshot_ready": optional_snapshot_ready,
            "average_quality_score": round(average_quality_score, 6),
            "soft_failure_count": int(collector_contracts.get("soft_failure_count", 0) or 0),
            "required_failure_count": int(collector_contracts.get("required_failure_count", 0) or 0),
        },
        "provider_groups": {
            "required_context": {
                "status": required_status,
                "summary": f"contract_ok={required_contract_ok}/{len(required_rows)} snapshot_ready={required_snapshot_ready}/{len(required_rows)}",
                "collectors": [str(row.get("name") or "") for row in required_rows],
            },
            "optional_context": {
                "status": _group_status(
                    total=len(optional_rows),
                    contract_ok=optional_contract_ok,
                    snapshot_ready=optional_snapshot_ready,
                    degraded_ok=True,
                ),
                "summary": f"contract_ok={optional_contract_ok}/{len(optional_rows)} snapshot_ready={optional_snapshot_ready}/{len(optional_rows)}",
                "collectors": [str(row.get("name") or "") for row in optional_rows],
            },
            "verification_mesh": {
                "status": verification_status,
                "depth_status": verification_depth_status,
                "summary": (
                    f"cross_verified={int(source_counts.get('cross_verified', 0) or 0)} "
                    f"single_verified={int(source_counts.get('single_verified', source_counts.get('single_source_verified', 0)) or 0)} "
                    f"unverified={int(source_counts.get('single_unverified', source_counts.get('single_source_unverified', 0)) or 0)}"
                ),
                "all_verified": all_verified,
                "all_cross_verified": all_cross_verified,
            },
            "quota_limited_providers": {
                "status": quota_status,
                "summary": (
                    f"cooldowns_active={int(cooldown['active'])} "
                    f"soft_failures={int(collector_contracts.get('soft_failure_count', 0) or 0)}"
                ),
                "active_cooldowns": [cooldown] if cooldown["active"] else [],
            },
        },
        "mesh_contracts": [
            {
                "name": str(row.get("name") or ""),
                "required": bool(row.get("required", False)),
                "safe_to_degrade": bool(row.get("safe_to_degrade", False)),
                "contract_ok": bool(row.get("contract_ok", False)),
                "payload_present": bool(row.get("payload_present", False)),
                "payload_size_bytes": int(row.get("payload_size_bytes", 0) or 0),
                "quality_score": float(row.get("quality_score", 0.0) or 0.0),
            }
            for row in rows
        ],
        "cooldowns": [cooldown] if cooldown["active"] else [],
        "required_failures": collector_contracts.get("required_failures", []),
        "soft_failures": collector_contracts.get("soft_failures", []),
        "recommended_actions": recommended_actions,
        "top_actions": recommended_actions[:4],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish provider-mesh readiness across required collectors, verification, and cooldown state.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "provider_mesh "
            f"overall_status={payload.get('overall_status', '')} "
            f"required_contract_ok={int(((payload.get('summary') or {}).get('required_contract_ok', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
