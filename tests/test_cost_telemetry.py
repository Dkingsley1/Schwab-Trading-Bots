import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import cost_telemetry as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_cost_telemetry_reports_cross_host_parity_seed_and_present_counts(tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    present_report = tmp_path / "exports" / "parity" / "backend_parity_report_latest.json"
    missing_report = tmp_path / "exports" / "parity" / "shadow_replay_diff_latest.json"
    present_report.parent.mkdir(parents=True, exist_ok=True)
    present_report.write_text("{}", encoding="utf-8")

    _write_json(health_root / "ingestion_storage_control_latest.json", {"backpressure": {"total_pending_lines": 2500}, "pressure_index": 1.2})
    _write_json(health_root / "training_quality_control_latest.json", {"training_quality_score": 86.0, "supportability": {"active_bots": 19}})
    _write_json(health_root / "runtime_throttle_control_latest.json", {"host_saturation_score": 42.0, "throttle_profile": "sustain"})
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "portability_score": 100.0,
            "recommended_runtime_mode": "native",
            "recommended_backend": "native_default",
            "host_contract": {"memory_architecture": "unified"},
            "portable_contract": {"sidecar_canary_supported": True},
            "nightly_proof_contract": {
                "ready": False,
                "report_paths": {
                    "backend_parity_report": str(present_report),
                    "shadow_replay_diff": str(missing_report),
                },
            },
        },
    )
    _write_json(health_root / "provider_mesh_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health_root / "cross_host_parity_report_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["cross_host_parity_contract"]["proof_seed_ready"] is True
    assert payload["cross_host_parity_contract"]["proof_path_count"] == 2
    assert payload["cross_host_parity_contract"]["proof_present_count"] == 1
    assert payload["portable_backend_cost_proxy"]["proof_present_count"] == 1
    assert payload["tenant_metering_contract"]["ready"] is True
