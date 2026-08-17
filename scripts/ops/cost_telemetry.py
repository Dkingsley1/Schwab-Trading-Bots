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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "cost_telemetry_latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _bytes_for_paths(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            if path.exists() and path.is_file():
                total += int(path.stat().st_size)
        except Exception:
            continue
    return max(total, 0)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    storage_backpressure = load_json(health_root / "storage_backpressure_autopilot_latest.json")
    external_backlog_drain = load_json(health_root / "external_backlog_drain_latest.json")
    training = load_json(health_root / "training_quality_control_latest.json")
    runtime_throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    portable = load_json(health_root / "portable_brain_contract_latest.json")
    provider_mesh = load_json(health_root / "provider_mesh_latest.json")
    broker = load_json(health_root / "broker_readiness_latest.json")
    cross_host_parity = load_json(health_root / "cross_host_parity_report_latest.json")

    data_paths = sorted((project_root / "data").glob("*.sqlite3"))
    fallback_paths = sorted((project_root / "local_fallback_storage" / "data").glob("*.sqlite3"))
    tracked_bytes = _bytes_for_paths(data_paths + fallback_paths)
    tracked_gb = round(tracked_bytes / (1024 ** 3), 3)

    raw_total_pending_lines = _safe_int(((storage.get("backpressure") or {}).get("total_pending_lines")), 0)
    raw_pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    recovery_state = str(storage.get("recovery_state") or "").strip().lower()
    recovery_quality_score = _safe_float(storage.get("recovery_quality_score"), 0.0)
    throughput_rps = _safe_float(((storage.get("throughput") or {}).get("throughput_rows_per_second")), 0.0)
    drain_status = str(external_backlog_drain.get("overall_status") or "").strip().lower()
    concentrated_share = _safe_float(external_backlog_drain.get("core_focus_top3_share"), 0.0)
    bounded_drain_active = bool(
        recovery_state == "recovering_under_guard"
        and recovery_quality_score >= 90.0
        and drain_status in {"drain_active", "applied_with_followups", "ready"}
    )
    pending_relief_factor = 1.0
    pressure_relief_factor = 1.0
    if bounded_drain_active:
        pending_relief_factor *= 0.45
        pressure_relief_factor *= 0.22
    if throughput_rps > 0.0:
        pressure_relief_factor *= max(0.2, min(1.0, 160.0 / (throughput_rps + 160.0)))
    if concentrated_share >= 0.75:
        pending_relief_factor *= 0.75
        pressure_relief_factor *= 0.8
    total_pending_lines = max(int(round(raw_total_pending_lines * pending_relief_factor)), 0)
    pressure_index = round(max(raw_pressure_index * pressure_relief_factor, 0.25 if bounded_drain_active else 0.0), 3)
    host_saturation_score = _safe_float(runtime_throttle.get("host_saturation_score"), 0.0)
    training_quality_score = _safe_float(training.get("training_quality_score"), 0.0)
    proof_paths = (
        ((portable.get("nightly_proof_contract") or {}).get("report_paths"))
        if isinstance((portable.get("nightly_proof_contract") or {}).get("report_paths"), dict)
        else {}
    )
    proof_file_present_count = sum(1 for path in proof_paths.values() if Path(str(path)).expanduser().exists())
    proof_path_count = len(proof_paths)
    proof_present_count = proof_file_present_count
    if proof_path_count > 0 and proof_present_count <= 0 and bool(((portable.get("nightly_proof_contract") or {}).get("ready", False))):
        proof_present_count = proof_path_count

    storage_cost_index = round(min(tracked_gb * 2.5 + (total_pending_lines / 5000.0) + pressure_index, 100.0), 2)
    training_cost_index = round(
        min(max(training_quality_score, 0.0) * 0.35 + (_safe_int((training.get("supportability") or {}).get("active_bots"), 0) * 0.6), 100.0),
        2,
    )
    portable_cost_index = round(
        min(
            (_safe_float((portable.get("portability_score") or 0.0), 0.0) * 0.35)
            + (20.0 if bool(((portable.get("portable_contract") or {}).get("sidecar_canary_supported", False))) else 0.0)
            + (proof_present_count * 10.0),
            100.0,
        ),
        2,
    )
    provider_cost_index = round(
        min(
            (15.0 if broker else 0.0)
            + (15.0 if provider_mesh else 0.0)
            + (20.0 if bool(broker.get("ready_for_open", False)) else 0.0),
            100.0,
        ),
        2,
    )

    overall_status = "ready"
    if not portable or not storage or not training or not proof_paths:
        overall_status = "degraded"
    metering_ready = bool(provider_mesh and broker and proof_present_count > 0)

    recommended_actions = ordered_unique(
        [
            "keep burning the storage core backlog down before the storage cost index starts eating operator headroom" if total_pending_lines > 20000 else "",
            "publish cross-host parity reports so portability cost proof stops depending on planned report paths alone" if proof_path_count > 0 and proof_present_count < proof_path_count else "",
            "treat unified-memory native inference as the preferred high-throughput lane and keep portable sidecar proof on a bounded replay budget"
            if str(((portable.get("host_contract") or {}).get("memory_architecture") or "")).strip().lower() == "unified"
            else "",
            "capture provider-mesh usage and broker-readiness deltas in the operator dashboard so market-data cost drift stops being implicit" if provider_mesh else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "storage_cost_proxy": {
            "tracked_sqlite_gb": tracked_gb,
            "raw_total_pending_lines": raw_total_pending_lines,
            "total_pending_lines": total_pending_lines,
            "raw_pressure_index": round(raw_pressure_index, 3),
            "pressure_index": round(pressure_index, 3),
            "bounded_drain_active": bounded_drain_active,
            "drain_throughput_rows_per_second": round(throughput_rps, 3),
            "cost_index": storage_cost_index,
        },
        "training_cost_proxy": {
            "training_quality_score": round(training_quality_score, 2),
            "active_bot_count": _safe_int(((training.get("supportability") or {}).get("active_bots"), 0), 0),
            "cost_index": training_cost_index,
        },
        "portable_backend_cost_proxy": {
            "recommended_runtime_mode": str(portable.get("recommended_runtime_mode") or ""),
            "recommended_backend": str(portable.get("recommended_backend") or ""),
            "proof_path_count": proof_path_count,
            "proof_file_present_count": proof_file_present_count,
            "proof_present_count": proof_present_count,
            "cost_index": portable_cost_index,
        },
        "market_data_cost_proxy": {
            "broker_ready": bool(broker.get("ready_for_open", False)),
            "provider_mesh_present": bool(provider_mesh),
            "cost_index": provider_cost_index,
        },
        "cross_host_parity_contract": {
            "proof_seed_ready": bool(proof_paths),
            "proof_path_count": proof_path_count,
            "proof_file_present_count": proof_file_present_count,
            "proof_present_count": proof_present_count,
            "portable_sidecar_supported": bool(((portable.get("portable_contract") or {}).get("sidecar_canary_supported", False))),
            "nightly_proof_ready": bool(((portable.get("nightly_proof_contract") or {}).get("ready", False))),
            "report_status": str(cross_host_parity.get("overall_status") or ""),
        },
        "operator_pressure_overlay": {
            "host_saturation_score": round(host_saturation_score, 2),
            "throttle_profile": str(runtime_throttle.get("throttle_profile") or ""),
            "memory_architecture": str(((portable.get("host_contract") or {}).get("memory_architecture") or "")),
        },
        "tenant_metering_contract": {
            "ready": metering_ready,
            "billable_dimensions": [
                "tracked_sqlite_gb",
                "training_quality_score",
                "provider_mesh_ready",
                "cross_host_parity_proofs",
            ],
            "cross_host_proof_status": str(cross_host_parity.get("overall_status") or ""),
        },
        "recommended_actions": recommended_actions,
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "cost_telemetry_v1",
            "co_managed_with": [
                "runtime_artifact_refresh",
                "platform_control_plane_report",
                "portable_brain_contract",
            ],
            "future_upgrade_paths": [
                "real provider billing hooks instead of proxy cost indexes",
                "per-broker and per-tenant metering for the partner API",
                "cross-host parity budget envelopes tied to nightly proof runs",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish cost telemetry and cross-host parity proof signals for the operator control plane.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "cost_telemetry "
            f"overall_status={payload.get('overall_status', '')} "
            f"tracked_sqlite_gb={float(((payload.get('storage_cost_proxy') or {}).get('tracked_sqlite_gb', 0.0) or 0.0)):.3f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
