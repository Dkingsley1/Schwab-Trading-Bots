#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "capital_growth_awareness_bridge_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _status(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return "missing"


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    if not rows and isinstance(registry.get("bots"), list):
        rows = registry.get("bots") or []
    return [row for row in rows if isinstance(row, dict)]


def _role_counts(registry: dict[str, Any]) -> dict[str, int]:
    counts = {
        "grand_master": 0,
        "master": 0,
        "sub_bot": 0,
        "infrastructure": 0,
    }
    for row in _registry_rows(registry):
        role = str(row.get("bot_role") or row.get("role") or "").strip().lower()
        bot_id = str(row.get("bot_id") or row.get("id") or "").strip().lower()
        if "grand" in role or "grand_master" in bot_id or "grandmaster" in bot_id:
            counts["grand_master"] += 1
        elif "master" in role or "master_bridge" in bot_id:
            counts["master"] += 1
        elif "infra" in role or "guard" in bot_id or "watch" in bot_id:
            counts["infrastructure"] += 1
        else:
            counts["sub_bot"] += 1
    return counts


def _sleeve_packets(growth: dict[str, Any]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row in _as_list(growth.get("sleeve_growth_plan")):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "unknown").strip() or "unknown"
        action = str(row.get("capital_action") or "").strip() or "observe_or_repair"
        reason = str(row.get("budget_reason") or "").strip()
        packets.append(
            {
                "profile": profile,
                "growth_grade": str(row.get("growth_grade") or ""),
                "capital_action": action,
                "paper_sim_budget_usd": round(_safe_float(row.get("paper_sim_budget_usd"), 0.0), 2),
                "live_micro_budget_usd": round(_safe_float(row.get("live_micro_budget_usd"), 0.0), 2),
                "budget_reason": reason,
                "instructions": _instructions_for_action(action, reason),
            }
        )
    return packets


def _instructions_for_action(action: str, reason: str) -> list[str]:
    if action == "candidate_for_growth":
        return [
            "masters may allocate more paper attention after repeatability checks",
            "sub-bots should keep collecting confirmation and disconfirmation evidence",
            "infra must keep fill, storage, runtime, and attribution ledgers fresh",
        ]
    if action == "cap_or_quarantine":
        return [
            "grand master should prevent blind scaling for this sleeve",
            "masters should route the sleeve to repair, data-depth, or quarantine review",
            "sub-bots should collect better labels instead of increasing exposure",
            "infra should watch for stale data, thin samples, or weak attribution",
        ]
    return [
        "masters should observe or repair before growth",
        "sub-bots should collect more usable evidence and avoid overacting",
        "infra should keep paper attribution and position ledgers fresh",
    ]


def _role_packets(growth: dict[str, Any], sleeve_packets: list[dict[str, Any]]) -> dict[str, Any]:
    control = _as_dict(growth.get("capital_growth_control"))
    live = _as_dict(growth.get("live_money_scaling"))
    policy = _as_dict(growth.get("money_tree_growth_policy"))
    weak = _as_list(_as_dict(growth.get("readiness")).get("quarantined_or_weak_profiles"))
    candidate_count = sum(1 for row in sleeve_packets if row.get("capital_action") == "candidate_for_growth")
    capped_count = sum(1 for row in sleeve_packets if row.get("capital_action") == "cap_or_quarantine")
    observe_count = sum(1 for row in sleeve_packets if row.get("capital_action") == "observe_or_repair")
    base_rules = [
        "paper simulation can plan growth at any account size",
        "live money remains blocked until the live-money scaling gate clears",
        "scale sleeves independently instead of scaling the whole system blindly",
        "increase only after realized, attributed, repeatable edge",
        "decrease or quarantine on drawdown, weak attribution, confirmation bias, or thin evidence",
    ]
    return {
        "grand_master": {
            "aware": True,
            "authority": "portfolio_growth_arbiter",
            "must_enforce": base_rules,
            "capital_growth_grade": control.get("grade"),
            "live_money_grade": live.get("grade"),
            "live_money_allowed": bool(live.get("allowed", False)),
            "summary": {
                "candidate_growth_sleeves": candidate_count,
                "observe_or_repair_sleeves": observe_count,
                "capped_or_quarantined_sleeves": capped_count,
            },
        },
        "masters": {
            "aware": True,
            "authority": "per_sleeve_budget_and_repair_routing",
            "must_enforce": [
                "read the per-sleeve capital_action before widening a sleeve",
                "send observe_or_repair sleeves to data quality, labels, or calibration",
                "do not override cap_or_quarantine without fresh attribution evidence",
            ],
            "sleeve_packets": sleeve_packets,
        },
        "sub_bots": {
            "aware": True,
            "authority": "evidence_collection_and_signal_hygiene",
            "must_enforce": [
                "collect useful disconfirming evidence, not only confirming evidence",
                "prefer better labels and attribution over more exposure",
                "do not treat paper budget as live buying power",
            ],
            "weak_profiles_to_repair": [str(item) for item in weak],
        },
        "infrastructure": {
            "aware": True,
            "authority": "gatekeeping_and_freshness_enforcement",
            "must_enforce": [
                "keep storage, runtime, training, fill, position ledger, and attribution surfaces fresh",
                "block live scaling when live-money blockers are present",
                "treat capital rotation as advisory or paper-only until explicit live-money graduation clears",
                "log capital-growth changes as reference frames for future troubleshooting",
            ],
            "watched_artifacts": [
                "governance/health/capital_growth_intelligence_latest.json",
                "governance/health/capital_rotation_control_latest.json",
                "governance/health/paper_profitability_control_latest.json",
                "governance/health/sleeve_profitability_dashboard_latest.json",
                "governance/health/income_readiness_latest.json",
                "governance/health/training_runtime_control_latest.json",
                "governance/health/storage_quota_guard_latest.json",
            ],
        },
        "policy": policy,
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    growth = load_json(health / "capital_growth_intelligence_latest.json")
    rotation = load_json(health / "capital_rotation_control_latest.json")
    registry = load_json(project_root / "master_bot_registry.json")
    sleeve_packets = _sleeve_packets(growth)
    role_counts = _role_counts(registry)
    role_packets = _role_packets(growth, sleeve_packets)
    live = _as_dict(growth.get("live_money_scaling"))
    control = _as_dict(growth.get("capital_growth_control"))
    blockers = [str(item) for item in _as_list(live.get("blockers"))]
    awareness_ready = bool(growth) and bool(sleeve_packets) and bool(control.get("allows_paper_money_tree_simulation", False))
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": awareness_ready,
        "overall_status": "ready" if awareness_ready else "needs_growth_plan",
        "capital_growth_status": _status(growth),
        "capital_growth_grade": control.get("grade"),
        "live_money_scaling_allowed": bool(live.get("allowed", False)),
        "live_money_scaling_blockers": blockers,
        "capital_rotation_status": _status(rotation),
        "capital_rotation_live_money_allowed": bool(
            _as_dict(rotation.get("runtime_contract")).get("live_money_rotation_allowed", False)
        ),
        "capital_rotation_action_mode": str(
            _as_dict(rotation.get("runtime_contract")).get("paper_rotation_action_mode") or ""
        ),
        "awareness_scope": {
            "grand_master": True,
            "masters": True,
            "sub_bots": True,
            "infrastructure": True,
            "all_visible_sleeves": True,
            "sleeve_count": len(sleeve_packets),
            "role_counts": role_counts,
        },
        "role_packets": role_packets,
        "sleeve_packets": sleeve_packets,
        "communication_edges": [
            {"from": "capital_growth_intelligence", "to": "capital_growth_awareness_bridge", "reason": "money-tree policy normalization"},
            {"from": "capital_rotation_control", "to": "capital_growth_awareness_bridge", "reason": "capital-flow movement map and paper-only action policy"},
            {"from": "capital_growth_awareness_bridge", "to": "grand_master", "reason": "portfolio-level growth arbitration"},
            {"from": "capital_growth_awareness_bridge", "to": "masters", "reason": "per-sleeve budget, repair, and cap rules"},
            {"from": "capital_growth_awareness_bridge", "to": "sub_bots", "reason": "evidence collection and signal hygiene"},
            {"from": "capital_growth_awareness_bridge", "to": "infrastructure", "reason": "freshness, gate, storage, training, and ledger enforcement"},
            {"from": "capital_growth_awareness_bridge", "to": "system_self_model", "reason": "shared awareness bus"},
        ],
        "recommended_commands": {
            "refresh_growth_plan": ["./scripts/ops/opsctl.sh", "capital-growth-intelligence", "--apply", "--json"],
            "refresh_capital_rotation": ["./scripts/ops/opsctl.sh", "capital-rotation-control", "--json"],
            "refresh_awareness_bridge": ["./scripts/ops/opsctl.sh", "capital-growth-awareness", "--json"],
            "refresh_self_model": ["./scripts/ops/opsctl.sh", "system-self-model", "--json"],
        },
        "source_artifacts": [
            str(health / "capital_growth_intelligence_latest.json"),
            str(health / "capital_rotation_control_latest.json"),
            str(project_root / "master_bot_registry.json"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish role-specific awareness packets for capital growth intelligence.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(Path(args.project_root).expanduser().resolve())
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        scope = _as_dict(payload.get("awareness_scope"))
        print(
            "capital_growth_awareness "
            f"status={payload.get('overall_status')} "
            f"grade={payload.get('capital_growth_grade')} "
            f"sleeves={_safe_int(scope.get('sleeve_count'), 0)} "
            f"live_allowed={int(bool(payload.get('live_money_scaling_allowed')))}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
