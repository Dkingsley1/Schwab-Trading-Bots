from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import distributed_cell_architecture as cells


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_distributed_cell_architecture_builds_seven_cells(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(health / "whole_system_intelligence_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "ingestion_storage_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "training_runtime_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "macro_event_intelligence_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})
    _write_json(health / "paper_profitability_control_latest.json", {"timestamp_utc": cells.iso_now(), "overall_status": "ready"})

    payload = cells.build_payload(project_root=tmp_path, apply=False, cell_root=tmp_path / "governance" / "cells")

    assert payload["cell_count"] == 7
    assert {row["cell_id"] for row in payload["cells"]} == {
        "control_plane",
        "sleeve_cells",
        "storage_writer_cell",
        "training_cell",
        "market_data_cell",
        "execution_paper_cell",
        "infra_cell",
    }
    assert payload["federation_contract"]["single_writer_authority"] == "storage_writer_cell"
    assert payload["intercell_bus"]["single_writer_authority"] == "storage_writer_cell"
    assert payload["distributed_runtime_arbitration"]["parallel_sqlite_commit_writers_allowed"] is False
    assert "storage_writer_cell" in payload["cell_dependency_graph"]["training_cell"]["depends_on"]
    market_contract = payload["cell_resource_contracts"]["market_data_cell"]
    assert market_contract["primary_budget"] == "required_context_first_optional_news_bounded"
    market_state = next(row for row in cells.CELL_DEFINITIONS if row["cell_id"] == "market_data_cell")
    assert "ticker_news_context" in {row["name"] for row in market_state["surfaces"]}
    assert payload["protected_volumes"]["VIDEO"] == "never_touched"
    assert "/Volumes/VIDEO" in payload["integration_contract"]["never_touch_protected_volumes"]
