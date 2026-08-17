#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "retrain_lane_scheduler_latest.json"
LANE_ORDER = ("mature", "probation", "new", "infrastructure")
LANE_SHARES = {
    "mature": 0.50,
    "probation": 0.25,
    "new": 0.15,
    "infrastructure": 0.10,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _normalized_bot_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _collect_new_bot_ids(admission_guard: dict[str, Any]) -> set[str]:
    rows = admission_guard.get("candidates") if isinstance(admission_guard.get("candidates"), list) else []
    return {
        _normalized_bot_id((row or {}).get("bot_id"))
        for row in rows
        if isinstance(row, dict) and _normalized_bot_id((row or {}).get("bot_id"))
    }


def _collect_probation_ids(probation_guard: dict[str, Any]) -> set[str]:
    rows = probation_guard.get("monitored_candidates") if isinstance(probation_guard.get("monitored_candidates"), list) else []
    return {
        _normalized_bot_id((row or {}).get("bot_id"))
        for row in rows
        if isinstance(row, dict) and _normalized_bot_id((row or {}).get("bot_id"))
    }


def _is_infrastructure_role(value: Any) -> bool:
    return str(value or "").strip().lower() == "infrastructure_sub_bot"


def _allocate_quotas(counts: dict[str, int], max_targets: int) -> dict[str, int]:
    total_available = sum(max(int(counts.get(lane, 0)), 0) for lane in LANE_ORDER)
    if max_targets <= 0 or max_targets >= total_available:
        return {lane: max(int(counts.get(lane, 0)), 0) for lane in LANE_ORDER}

    quotas = {lane: 0 for lane in LANE_ORDER}
    remaining = min(max_targets, total_available)
    for lane in LANE_ORDER:
        lane_available = max(int(counts.get(lane, 0)), 0)
        if lane_available <= 0:
            continue
        lane_quota = max(1, int(round(max_targets * float(LANE_SHARES[lane]))))
        quotas[lane] = min(lane_available, lane_quota)
    while sum(quotas.values()) > remaining:
        for lane in ("new", "infrastructure", "probation", "mature"):
            if sum(quotas.values()) <= remaining:
                break
            if quotas[lane] > 0:
                quotas[lane] -= 1
    while sum(quotas.values()) < remaining:
        progressed = False
        for lane in LANE_ORDER:
            lane_available = max(int(counts.get(lane, 0)), 0)
            if quotas[lane] < lane_available:
                quotas[lane] += 1
                progressed = True
                if sum(quotas.values()) >= remaining:
                    break
        if not progressed:
            break
    return quotas


def build_payload(
    *,
    registry: dict[str, Any],
    walk_forward: dict[str, Any],
    new_bot_admission_guard: dict[str, Any],
    probation_guard: dict[str, Any],
    target_bot_ids: list[str] | None,
    max_targets: int,
    new_bot_max_runs: int,
) -> dict[str, Any]:
    target_ids = [_normalized_bot_id(item) for item in (target_bot_ids or []) if _normalized_bot_id(item)]
    target_set = set(target_ids)
    order_index = {bot_id: idx for idx, bot_id in enumerate(target_ids)}

    registry_rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    wf_rows = walk_forward.get("bots") if isinstance(walk_forward.get("bots"), dict) else {}
    admission_ids = _collect_new_bot_ids(new_bot_admission_guard)
    probation_ids = _collect_probation_ids(probation_guard)

    lane_rows: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_ORDER}
    for raw_row in registry_rows:
        if not isinstance(raw_row, dict):
            continue
        bot_id = _normalized_bot_id(raw_row.get("bot_id"))
        if not bot_id:
            continue
        if target_set and bot_id not in target_set:
            continue
        lifecycle_state = str(raw_row.get("lifecycle_state") or "").strip().lower()
        if lifecycle_state in {"retired", "deleted", "deactivated"} or bool(raw_row.get("deleted_from_rotation", False)):
            continue
        if not (bool(raw_row.get("active", False)) or lifecycle_state == "probation"):
            continue
        wf_row = wf_rows.get(bot_id) if isinstance(wf_rows.get(bot_id), dict) else {}
        runs = _to_int((wf_row or {}).get("runs"), 0)
        if _is_infrastructure_role(raw_row.get("bot_role") or raw_row.get("role")):
            lane = "infrastructure"
        elif bot_id in probation_ids or lifecycle_state == "probation":
            lane = "probation"
        elif bot_id in admission_ids or runs <= int(new_bot_max_runs):
            lane = "new"
        else:
            lane = "mature"
        lane_rows[lane].append(
            {
                "bot_id": bot_id,
                "walk_forward_runs": runs,
                "priority_order": order_index.get(bot_id, len(order_index) + len(lane_rows[lane])),
            }
        )

    for lane in LANE_ORDER:
        lane_rows[lane].sort(
            key=lambda row: (
                int(row.get("priority_order", 0) or 0),
                int(row.get("walk_forward_runs", 0) or 0),
                str(row.get("bot_id") or ""),
            )
        )

    counts = {lane: len(lane_rows[lane]) for lane in LANE_ORDER}
    quotas = _allocate_quotas(counts, max_targets)
    selected_lane_rows = {
        lane: lane_rows[lane][: max(int(quotas.get(lane, 0)), 0)]
        for lane in LANE_ORDER
    }

    interleaved_bot_ids: list[str] = []
    lane_queues = {lane: list(rows) for lane, rows in selected_lane_rows.items()}
    while any(lane_queues[lane] for lane in LANE_ORDER):
        for lane in LANE_ORDER:
            if lane_queues[lane]:
                interleaved_bot_ids.append(str(lane_queues[lane].pop(0)["bot_id"]))

    ok = bool(any(counts.values()))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "max_targets": int(max_targets),
        "selected_bot_ids": interleaved_bot_ids,
        "summary": {
            "candidate_count": sum(counts.values()),
            "selected_count": len(interleaved_bot_ids),
            "lane_count": sum(1 for lane in LANE_ORDER if counts[lane] > 0),
        },
        "lanes": {
            lane: {
                "candidate_count": counts[lane],
                "quota": int(quotas.get(lane, 0) or 0),
                "selected_count": len(selected_lane_rows[lane]),
                "bot_ids": [str(row.get("bot_id") or "") for row in selected_lane_rows[lane]],
            }
            for lane in LANE_ORDER
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split retrain targets into isolated mature, probation, and new-bot lanes.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--new-bot-admission-file", default=str(PROJECT_ROOT / "governance" / "health" / "new_bot_admission_guard_latest.json"))
    parser.add_argument("--probation-guard-file", default=str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"))
    parser.add_argument("--target-bot-ids", default="")
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--new-bot-max-runs", type=int, default=24)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        registry=_load_json(Path(args.registry)),
        walk_forward=_load_json(Path(args.walk_forward_file)),
        new_bot_admission_guard=_load_json(Path(args.new_bot_admission_file)),
        probation_guard=_load_json(Path(args.probation_guard_file)),
        target_bot_ids=[item for item in str(args.target_bot_ids or "").split(",") if str(item).strip()],
        max_targets=int(args.max_targets),
        new_bot_max_runs=int(args.new_bot_max_runs),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary", {})
        print(
            "retrain_lane_scheduler "
            f"ok={str(payload['ok']).lower()} "
            f"candidates={int(summary.get('candidate_count', 0) or 0)} "
            f"selected={int(summary.get('selected_count', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
