import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.schema_migration_guard as src


def _write_valid_contracts(root: Path) -> None:
    for spec in src.CONTRACT_SPECS:
        payload = {key: {} for key in spec["required_keys"]}
        payload["schema_version"] = 1
        _write_json(root / spec["path"], payload)


def test_missing_required_fields_fail_closed(tmp_path: Path) -> None:
    _write_valid_contracts(tmp_path)
    _write_json(
        tmp_path / src.CONTRACT_SPECS[0]["path"],
        {"schema_version": 1, "ok": True},
    )

    payload = src.build_payload(tmp_path)

    assert payload["ok"] is False
    assert payload["overall_status"] == "needs_work"
    assert payload["summary"]["needs_work_contracts"] == 1
    assert payload["contracts"][0]["missing_keys"] == ["sleeve_latest"]


def test_all_incomplete_contracts_cannot_report_success(tmp_path: Path) -> None:
    for spec in src.CONTRACT_SPECS:
        _write_json(tmp_path / spec["path"], {"schema_version": 1})
    payload = src.build_payload(tmp_path)
    assert not payload["ok"]
    assert payload["overall_status"] == "needs_work"
    assert payload["summary"]["needs_work_contracts"] == len(src.CONTRACT_SPECS)


def test_complete_contracts_are_ready(tmp_path: Path) -> None:
    _write_valid_contracts(tmp_path)
    payload = src.build_payload(tmp_path)
    assert payload["ok"]
    assert payload["overall_status"] == "ready"


def test_cli_returns_nonzero_for_incomplete_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    _write_valid_contracts(tmp_path)
    _write_json(tmp_path / src.CONTRACT_SPECS[0]["path"], {"schema_version": 1})
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "schema_migration_guard.py",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(output),
        ],
    )
    assert src.main() == 2
    assert json.loads(output.read_text())["ok"] is False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_schema_migration_guard_counts_missing_and_legacy_contracts(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {"schema_version": 1, "ok": True, "sleeve_latest": []},
    )
    _write_json(
        tmp_path / "governance" / "health" / "point_in_time_event_store_latest.json",
        {"ok": True, "event_count": 3, "events": []},
    )
    _write_json(
        tmp_path / "governance" / "health" / "training_quality_control_latest.json",
        {"overall_status": "ready", "training_quality_score": 90.0, "improvements": []},
    )
    _write_json(
        tmp_path / "governance" / "health" / "platform_control_plane_latest.json",
        {"institutional_readiness": {}, "institutional_domains_by_slug": {}},
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {"schema_version": 1, "dataset_contract": {}, "point_in_time_contract": {}},
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["summary"]["missing_contracts"] == 1
    assert payload["summary"]["legacy_unversioned_contracts"] >= 2
