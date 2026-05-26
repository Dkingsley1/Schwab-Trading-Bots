from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import quant_strategy_storage_backlog_accommodation as accommodation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_quant_strategy_storage_backlog_accommodation_refreshes_from_current_storage_truth(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "backpressure": {
                "core_pending_lines": 68,
                "deferred_pending_lines": 9,
                "cold_pending_lines": 0,
                "support_pending_lines": 7,
                "total_pending_lines": 77,
                "oldest_pending_age_seconds": 0.0,
                "pending_lines_threshold": 15000,
                "oldest_age_threshold_seconds": 240.0,
                "estimated_total_drain_minutes": 15.0,
            },
            "backlog_truth": {
                "raw_live": {
                    "grade": "A++",
                    "pressure_ratio": 0.005,
                    "core_pending_lines": 68,
                    "total_pending_lines": 77,
                    "oldest_pending_age_seconds": 0.0,
                },
                "stale_pending_locator": {"stale_source_count": 0},
            },
        },
    )

    payload = accommodation.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["storage_snapshot"]["backlog_letter_grade"] == "A++"
    assert payload["storage_snapshot"]["total_pending_lines"] == 77
    assert payload["grade"]["letter_grade"] == "A+"
    assert payload["grade"]["target_met"] is True
    assert payload["stale_artifact_repair"]["cleared"] is True
    assert payload["artifact_contract"]["never_touch_video_volume"] is True
