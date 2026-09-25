import gzip
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import emergency_storage_thin as thin

NOW = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)


def setup(tmp_path, monkeypatch, free=24):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    payload = {
        "timestamp_utc": NOW.isoformat(),
        "previews": {
            "raw_training_compaction": {
                "blockers": ["storage_pressure_above_raw_compaction_ceiling"],
                "raw_live": {
                    "total_pending_lines": 10,
                    "core_pending_lines": 5,
                    "oldest_pending_age_seconds": 5,
                },
            }
        },
    }
    (health / "storage_backpressure_autopilot_latest.json").write_text(
        json.dumps(payload)
    )
    monkeypatch.setattr(thin, "resources_clear", lambda *a: True)
    monkeypatch.setattr(thin, "recovery_hold_active", lambda *a: False)
    monkeypatch.setattr(
        thin.shutil, "disk_usage", lambda *a: SimpleNamespace(free=free * thin.GIB)
    )
    return health, payload


@pytest.mark.parametrize(
    "free,allowed", [(8, False), (16, False), (24, True), (32, False), (49, False)]
)
def test_only_emergency_band_admitted(tmp_path, monkeypatch, free, allowed):
    setup(tmp_path, monkeypatch, free)
    blockers, _ = thin.admission(tmp_path, NOW)
    assert (not blockers) is allowed


def test_real_backlog_or_extra_fault_is_not_disk_only(tmp_path, monkeypatch):
    health, payload = setup(tmp_path, monkeypatch)
    preview = payload["previews"]["raw_training_compaction"]
    preview["raw_live"]["core_pending_lines"] = 10001
    preview["blockers"].append("integrity_failure")
    (health / "storage_backpressure_autopilot_latest.json").write_text(
        json.dumps(payload)
    )
    blockers, _ = thin.admission(tmp_path, NOW)
    assert "active_ingestion_backpressure" in blockers
    assert "disk_only_repair_deadlock_not_proven" in blockers


def test_current_day_fallback_and_symlinks_never_selected(tmp_path, monkeypatch):
    health, _ = setup(tmp_path, monkeypatch)
    paths = [
        tmp_path / "archive/2026-09-20.jsonl",
        tmp_path / "archive/2026-09-23.jsonl",
        tmp_path / "local_fallback_storage/governance/old.jsonl",
        tmp_path / "archive/state_latest.jsonl",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"evidence":1}\n')
        os.utime(path, (NOW.timestamp() - 3 * 86400,) * 2)
    link = tmp_path / "archive/link.jsonl"
    link.symlink_to(paths[0])
    paths.append(link)
    (health / "raw_training_compaction_intelligence_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": NOW.isoformat(),
                "top_compaction_candidates": [{"path": str(p)} for p in paths],
            }
        )
    )
    assert [row["path"] for row in thin.candidates(tmp_path, NOW)] == [str(paths[0])]


def test_failed_attempts_consume_hourly_budget(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        thin,
        "candidates",
        lambda *a: [{"path": "unused", "compressed_path": "unused.gz"}],
    )
    calls = []

    def compress(*a, **kw):
        calls.append(a)
        return {"status": "failed", "raw_removed": False}

    monkeypatch.setattr(thin.raw, "_compress_and_clear", compress)
    for _ in range(2):
        assert thin.run(tmp_path, apply=True, now=NOW)["state"] == "incomplete"
    result = thin.run(tmp_path, apply=True, now=NOW)
    assert result["blockers"] == ["hourly_attempt_budget"]
    assert len(calls) == 2


def test_verified_compaction_retains_full_restore(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    path = tmp_path / "old.jsonl"
    data = b'{"evidence":42}\n' * 100
    path.write_bytes(data)
    monkeypatch.setattr(
        thin,
        "candidates",
        lambda *a: [{"path": str(path), "compressed_path": str(path) + ".gz"}],
    )
    monkeypatch.setattr(thin.ThinGuard, "check", lambda self: None)
    monkeypatch.setattr(thin.raw, "_require_idle_recovery_source", lambda *a: None)
    result = thin.run(tmp_path, apply=True, now=NOW)
    assert result["state"] == "completed" and not path.exists()
    assert gzip.decompress((tmp_path / "old.jsonl.gz").read_bytes()) == data
    assert (
        result["records"][0]["verification_basis"]
        == "full_gzip_restore_sha256_and_stable_source_identity"
    )


def test_non_admitted_never_calls_compression(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, free=49)
    monkeypatch.setattr(
        thin.raw, "_compress_and_clear", lambda *a, **k: pytest.fail("must not run")
    )
    assert thin.run(tmp_path, apply=True, now=NOW)["state"] == "deferred"


def test_resource_admission_relaxes_only_disk_component(tmp_path):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    runtime = {
        "timestamp_utc": NOW.isoformat(),
        "host_saturation_score": 30,
        "compute_pressure_level": "normal",
        "mac_fluidity_contract": {"fluidity_band": "comfortable"},
        "runtime_snapshot": {
            "thermal": {
                "thermal_warning_active": False,
                "performance_warning_active": False,
                "cpu_power_warning_active": False,
            }
        },
    }
    resource = {
        "timestamp_utc": NOW.isoformat(),
        "input_evidence_ready": True,
        "memory_available_pct": 70,
        "memory_free_pct": 20,
        "load1_per_core": 0.5,
        "local_disk_free_gb": 24,
        "swap_used_gb": 0,
        "pages_throttled": 0,
    }
    (health / "runtime_throttle_control_latest.json").write_text(json.dumps(runtime))
    path = health / "resource_guard_latest.json"
    path.write_text(json.dumps(resource))
    assert thin.resources_clear(tmp_path, NOW)
    resource["memory_available_pct"] = 20
    path.write_text(json.dumps(resource))
    assert not thin.resources_clear(tmp_path, NOW)
    resource["memory_available_pct"] = 70
    resource["timestamp_utc"] = "2026-09-23T11:00:00+00:00"
    path.write_text(json.dumps(resource))
    assert not thin.resources_clear(tmp_path, NOW)


def test_stop_flag_is_preserved_and_blocks_repair(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    thin.write(tmp_path, "governance/health/SYSTEM_POWER_OFF.flag", {"reason": "off"})
    assert "operator_or_maintenance_hold" in thin.admission(tmp_path, NOW)[0]


def test_candidate_wave_limits_two_files_and_three_gib(tmp_path, monkeypatch):
    health, _ = setup(tmp_path, monkeypatch)
    paths = [tmp_path / f"old_{i}.jsonl" for i in range(4)]
    for path in paths:
        path.write_text("old")
    (health / "raw_training_compaction_intelligence_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": NOW.isoformat(),
                "top_compaction_candidates": [{"path": str(p)} for p in paths],
            }
        )
    )
    monkeypatch.setattr(
        thin.raw,
        "_classify_row",
        lambda path, **kw: {
            "path": str(path),
            "compression_candidate": True,
            "age_hours": 72,
            "size_bytes": int(1.5 * thin.GIB),
        },
    )
    rows = thin.candidates(tmp_path, NOW)
    assert len(rows) == 2 and sum(row["size_bytes"] for row in rows) == 3 * thin.GIB
