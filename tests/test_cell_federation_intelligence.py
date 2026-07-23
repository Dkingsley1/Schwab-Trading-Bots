from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import cell_federation_intelligence as intel


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cell_federation_intelligence_prioritizes_storage_and_protects_training(tmp_path: Path) -> None:
    distributed = {
        "timestamp_utc": intel.iso_now(),
        "grade": "A+",
        "operational_health": {"status": "blocked", "grade": "F", "score": 12.0},
        "cells": [
            {"cell_id": "storage_writer_cell", "overall_status": "blocked", "score": 1.0, "grade": "F", "need_count": 2},
            {"cell_id": "training_cell", "overall_status": "blocked", "score": 0.0, "grade": "F", "need_count": 1},
        ],
        "top_needs": [
            {
                "cell_id": "storage_writer_cell",
                "surface": "ingestion_storage",
                "status": "blocked",
                "risk_level": "high",
                "exact_file": "governance/health/ingestion_storage_control_latest.json",
                "exact_blocker": "backpressure_overload_severe",
                "recommended_command": ["./scripts/ops/opsctl.sh", "training-drain-autopilot", "--apply", "--json"],
            },
            {
                "cell_id": "training_cell",
                "surface": "training_runtime",
                "status": "blocked",
                "risk_level": "high",
                "exact_file": "governance/health/training_runtime_control_latest.json",
                "exact_blocker": "storage_quota_hard_breach",
                "recommended_command": ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
            },
        ],
    }
    distributed_path = tmp_path / "governance" / "health" / "distributed_cell_architecture_latest.json"
    _write_json(distributed_path, distributed)

    payload = intel.build_payload(
        project_root=tmp_path,
        distributed_path=distributed_path,
        cell_root=tmp_path / "governance" / "cells",
        apply=False,
    )

    assert payload["intelligence_score"] >= 80
    assert payload["ranked_needs"][0]["cell_id"] == "storage_writer_cell"
    training_policy = next(row for row in payload["cell_runtime_policy"] if row["cell_id"] == "training_cell")
    assert training_policy["action"] == "pause_until_storage_writer_and_infra_clear"
    assert "batch20" in training_policy["pause_or_throttle"]
    assert training_policy["dependency_blockers"][0]["dependency_cell"] == "storage_writer_cell"
    assert payload["dependency_health"]["training_cell"]["dependency_blocker_count"] >= 1
    assert payload["distributed_mode"] == "drain_or_host_relief_before_training"
    assert payload["resource_arbitration"]["parallel_sqlite_commit_writers_allowed"] is False
    assert any(row["cell_id"] == "training_cell" for row in payload["cell_handshake_packets"])
    assert payload["computer_smoothness_policy"]["writer_policy"].startswith("single SQLite writer")
    assert payload["computer_smoothness_policy"]["protected_volumes"]["VIDEO"] == "never_touched"
