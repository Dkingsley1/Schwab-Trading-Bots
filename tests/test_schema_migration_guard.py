import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.schema_migration_guard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_schema_migration_guard_counts_missing_and_legacy_contracts(tmp_path: Path) -> None:
    _write_json(tmp_path / "governance" / "health" / "paper_performance_latest.json", {"schema_version": 1, "ok": True, "sleeve_latest": []})
    _write_json(tmp_path / "governance" / "health" / "point_in_time_event_store_latest.json", {"ok": True, "event_count": 3, "events": []})
    _write_json(tmp_path / "governance" / "health" / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 90.0, "improvements": []})
    _write_json(tmp_path / "governance" / "health" / "platform_control_plane_latest.json", {"institutional_readiness": {}, "institutional_domains_by_slug": {}})
    _write_json(tmp_path / "governance" / "feature_store" / "latest.json", {"schema_version": 1, "dataset_contract": {}, "point_in_time_contract": {}})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["summary"]["missing_contracts"] == 1
    assert payload["summary"]["legacy_unversioned_contracts"] >= 2

