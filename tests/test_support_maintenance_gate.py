import json
from pathlib import Path

from scripts.ops import support_maintenance_gate as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_support_maintenance_gate_activates_from_runtime_override(tmp_path: Path) -> None:
    override = tmp_path / "config" / ".env.runtime_resource_guard_override"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(
        "OPS_SUPPORT_MAINTENANCE_FREEZE=1\nMAC_FLUIDITY_SUPPORT_PAUSE=1\nSUPPORT_MAINTENANCE_CONCURRENCY=0\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {
            "mac_fluidity_contract": {
                "overall_status": "needs_work",
                "fluidity_band": "strained",
                "fluidity_score": 58.0,
                "support_pause_recommended": True,
            }
        },
    )

    contract = src.support_maintenance_freeze_contract(tmp_path, "resource_guard")

    assert contract["active"] is True
    assert contract["reason"] == "support_maintenance_frozen_for_mac_fluidity"
    assert contract["mac_fluidity"]["support_pause_recommended"] is True


def test_frozen_health_payload_preserves_previous_shape(tmp_path: Path) -> None:
    previous_path = tmp_path / "governance" / "health" / "storage_failback_sync_latest.json"
    _write_json(previous_path, {"mode": "external", "certified_mode": "external", "split_brain_conflicts": 0})

    payload = src.frozen_health_payload(
        previous_path,
        {"active": True, "reason": "support_maintenance_frozen_for_mac_fluidity", "component": "storage_failback_sync"},
    )

    assert payload["mode"] == "external"
    assert payload["certified_mode"] == "external"
    assert payload["support_maintenance_frozen"] is True
    assert payload["skipped_reason"] == "support_maintenance_frozen_for_mac_fluidity"


def test_support_maintenance_gate_cli_writes_status_artifact(tmp_path: Path) -> None:
    out_file = tmp_path / "governance" / "health" / "support_maintenance_gate_latest.json"

    rc = src.main(["--project-root", str(tmp_path), "--out-file", str(out_file), "--json"])

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["memory_pressure_allocation_role"] == "yieldable_support_lane"
    assert payload["lane_policy"]["support_report_media"] == "off_hours_or_nice_20"
