from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.ops import system_drift_registry as src


def test_system_drift_registry_payload_is_json_ready(tmp_path: Path) -> None:
    payload = src.build_payload(tmp_path)

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["overall_status"] == "ready"
    assert decoded["surface_count"] > 0
    assert decoded["repairable_surface_count"] > 0
    assert decoded["family_counts"]
    assert all(isinstance(row["artifact_path"], str) for row in decoded["surfaces"])
    assert {"paper_execution_truth_layer", "paper_profitability_control"} <= {row["name"] for row in decoded["surfaces"]}
    assert decoded["recommended_commands"] == [["./scripts/ops/opsctl.sh", "system-drift-guard", "--json"]]


def test_system_drift_registry_main_writes_artifact(monkeypatch, tmp_path: Path) -> None:
    out_file = tmp_path / "governance" / "health" / "system_drift_registry_latest.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "system_drift_registry.py",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(out_file),
            "--json",
        ],
    )
    rc = src.main()

    assert rc == 0
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written["overall_status"] == "ready"
    assert written["surface_count"] > 0
