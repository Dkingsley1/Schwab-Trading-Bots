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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_snapshot_cache_control_latest.json"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    fresh_minutes: float = 360.0,
    stale_minutes: float = 24.0 * 60.0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    snapshot_path = health_root / "runtime_training_snapshot_latest.json"
    snapshot = load_json(snapshot_path)
    training_runtime = load_json(health_root / "training_runtime_control_latest.json")
    retrain_freshness = load_json(health_root / "retrain_artifact_freshness_latest.json")
    coverage_seed = load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")

    snapshot_age_minutes = payload_age_minutes(snapshot, snapshot_path)
    snapshot_exists = snapshot_path.exists() and bool(snapshot)
    sequence_count = int(snapshot.get("sequence_count", 0) or 0)
    row_count = int(snapshot.get("row_count", 0) or 0)
    coverage_top_sequences = ((snapshot.get("coverage") or {}).get("top_sequences")) if isinstance(snapshot.get("coverage"), dict) else []
    snapshot_intrinsic_ready = bool(snapshot_exists and sequence_count > 0 and row_count > 0)
    training_runtime_snapshot_ready = bool(training_runtime.get("snapshot_ready", False))
    snapshot_ready = bool(training_runtime_snapshot_ready or snapshot_intrinsic_ready)
    freshness_ok = snapshot_age_minutes is not None and float(snapshot_age_minutes) <= float(fresh_minutes)
    stale = snapshot_age_minutes is None or float(snapshot_age_minutes) > float(stale_minutes)
    coverage_shortfall_bots = int(coverage_seed.get("coverage_shortfall_bots", 0) or 0)

    overall_status = "ready"
    if not snapshot_exists or stale or not snapshot_ready:
        overall_status = "blocked"
    elif not freshness_ok:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "./scripts/ops/opsctl.sh runtime-training-snapshot --json" if not snapshot_exists or stale else "",
            "treat the runtime snapshot as a continuously refreshed cache, not an ad hoc batch artifact" if not freshness_ok else "",
            "precompute sequence-heavy retrain targets before asking the live host to retrain again" if coverage_shortfall_bots > 0 else "",
            "keep replay and reconciliation freshness green so cache rebuilds have trustworthy inputs" if retrain_freshness.get("ok") is False else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "cache_health": {
            "snapshot_exists": bool(snapshot_exists),
            "snapshot_ready": bool(snapshot_ready),
            "snapshot_ready_source": "training_runtime_control" if training_runtime_snapshot_ready else "snapshot_payload" if snapshot_intrinsic_ready else "none",
            "training_runtime_snapshot_ready": bool(training_runtime_snapshot_ready),
            "snapshot_intrinsic_ready": bool(snapshot_intrinsic_ready),
            "snapshot_age_minutes": round(float(snapshot_age_minutes), 4) if snapshot_age_minutes is not None else None,
            "fresh_minutes": float(fresh_minutes),
            "stale_minutes": float(stale_minutes),
            "sequence_count": sequence_count,
            "row_count": row_count,
            "top_sequence_count": len(coverage_top_sequences or []),
        },
        "upstream_inputs": {
            "retrain_artifact_freshness_ok": bool(retrain_freshness.get("ok", False)),
            "coverage_shortfall_bots": coverage_shortfall_bots,
            "precompute_target_count": len(training_runtime.get("precompute_targets") or []),
        },
        "infra_bots": ["runtime_snapshot_cache_control", "build_runtime_training_snapshot", "training_runtime_control"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track whether the runtime training snapshot behaves like a continuously refreshed cache.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--fresh-minutes", type=float, default=360.0)
    parser.add_argument("--stale-minutes", type=float, default=24.0 * 60.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        fresh_minutes=float(args.fresh_minutes),
        stale_minutes=float(args.stale_minutes),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_snapshot_cache_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"snapshot_ready={int(bool(((payload.get('cache_health') or {}).get('snapshot_ready', False))))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
