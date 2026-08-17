import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collector_contracts
import run_cached_collector
import storage_tier_policy


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collector_mesh_adds_ten_observation_only_streams_with_fail_closed_central_bank_context() -> None:
    specs = collector_contracts.ORGANIC_EVIDENCE_COLLECTOR_SPECS
    names = [str(spec["name"]) for spec in specs]

    assert len(specs) == 10
    assert len(set(names)) == 10
    assert {str(spec["collector_class"]) for spec in specs} == {
        "source_context",
        "decision_critical_source_context",
        "evidence_accrual",
    }
    central_bank = next(spec for spec in specs if spec["name"] == "central_bank_liquidity_context")
    optional_specs = [spec for spec in specs if spec is not central_bank]
    assert central_bank["required"] is True
    assert central_bank["safe_to_degrade"] is False
    assert central_bank["data_plane_key"] == "official_macro_context"
    assert all(spec["required"] is False for spec in optional_specs)
    assert all(spec["safe_to_degrade"] is True for spec in optional_specs)
    assert all(spec.get("evidence_domains") for spec in specs)
    assert all(spec.get("owner_command") for spec in specs)
    assert not any("start-live" in " ".join(spec["owner_command"]) for spec in specs)


def test_organic_collector_progress_requires_real_evidence_counts() -> None:
    spec = {"collector_class": "evidence_accrual", "organic_minimums": {"capture_count": 100}}

    accumulating = collector_contracts._organic_readiness(
        spec,
        fresh=True,
        health_ok=True,
        payload_present=True,
        payload_nonempty=True,
        health_payload={"capture_count": 25},
        payload_body={},
    )
    ready = collector_contracts._organic_readiness(
        spec,
        fresh=True,
        health_ok=True,
        payload_present=True,
        payload_nonempty=True,
        health_payload={"capture_count": 100},
        payload_body={},
    )

    assert accumulating["ready"] is False
    assert accumulating["progress"] == 0.25
    assert accumulating["blockers"] == ["minimum_not_met:capture_count:25/100"]
    assert ready["ready"] is True
    assert ready["progress"] == 1.0


def test_organic_collector_reports_partial_lineage_progress_without_clearing_gate() -> None:
    spec = {
        "collector_class": "evidence_accrual",
        "organic_truthy_paths": ["strict_ok"],
        "organic_ratio_targets": {"point_in_time.snapshot_coverage_ratio": 0.75},
    }

    payload = collector_contracts._organic_readiness(
        spec,
        fresh=True,
        health_ok=False,
        payload_present=True,
        payload_nonempty=True,
        health_payload={"strict_ok": False, "point_in_time": {"snapshot_coverage_ratio": 0.375}},
        payload_body={},
    )

    assert payload["ready"] is False
    assert payload["progress"] == 0.5
    assert "collector_health_not_ok" in payload["blockers"]
    assert "truthy_requirement_not_met:strict_ok" in payload["blockers"]
    assert "ratio_target_not_met:point_in_time.snapshot_coverage_ratio:0.375/0.75" in payload["blockers"]


def test_run_cached_collector_skips_when_expected_artifact_is_fresh(tmp_path, monkeypatch, capsys) -> None:
    expected = tmp_path / "governance" / "health" / "collector.json"
    _write_json(expected, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": True})
    cache_root = tmp_path / "cache"
    ops_db = tmp_path / "governance" / "ops_data_plane.sqlite3"
    monkeypatch.setenv("BOT_OPS_CONTROL_DB", str(ops_db))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cached_collector.py",
            "--key",
            "demo",
            "--max-age-minutes",
            "60",
            "--cache-root",
            str(cache_root),
            "--expect-path",
            str(expected),
            "--json",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(5)",
        ],
    )

    rc = run_cached_collector.main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert rc == 0
    assert payload["skipped"] is True
    assert payload["reason"] == "fresh_artifacts_reused"
    assert payload["run_uid"]
    with sqlite3.connect(str(ops_db)) as conn:
        row = conn.execute(
            "SELECT collector_key, skipped FROM collector_provenance_runs ORDER BY finished_utc DESC LIMIT 1"
        ).fetchone()
    assert row == ("demo", 1)


def test_run_cached_collector_runs_when_any_expected_artifact_is_missing(tmp_path, monkeypatch, capsys) -> None:
    expected = tmp_path / "governance" / "health" / "collector.json"
    missing_payload = tmp_path / "exports" / "external_context" / "collector_payload.json"
    _write_json(expected, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "ok": True})
    cache_root = tmp_path / "cache"
    ops_db = tmp_path / "governance" / "ops_data_plane.sqlite3"
    monkeypatch.setenv("BOT_OPS_CONTROL_DB", str(ops_db))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cached_collector.py",
            "--key",
            "demo",
            "--max-age-minutes",
            "60",
            "--cache-root",
            str(cache_root),
            "--expect-path",
            str(expected),
            "--expect-path",
            str(missing_payload),
            "--json",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path(r'{missing_payload}').parent.mkdir(parents=True, exist_ok=True); Path(r'{missing_payload}').write_text('{{}}', encoding='utf-8')",
        ],
    )

    rc = run_cached_collector.main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert rc == 0
    assert payload["skipped"] is False
    assert payload["ran"] is True
    assert payload["all_expected_present"] is False
    assert missing_payload.exists()


def test_collector_contracts_reports_required_failures(tmp_path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    spec = {
        "name": "demo_required",
        "health_path": tmp_path / "governance" / "health" / "demo_required.json",
        "payload_path": tmp_path / "exports" / "external_context" / "demo_required.json",
        "freshness_minutes": 30,
        "required": True,
        "safe_to_degrade": False,
    }
    _write_json(Path(spec["health_path"]), {"timestamp_utc": now.isoformat(), "ok": False})

    monkeypatch.setattr(collector_contracts, "COLLECTOR_SPECS", [spec])
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector_contracts.py", "--project-root", str(tmp_path)],
    )

    rc = collector_contracts.main()
    payload = json.loads((tmp_path / "governance" / "health" / "collector_contracts_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["required_failures"] == ["demo_required"]
    assert payload["rows"][0]["quality_score"] < 1.0
    assert payload["rows"][0]["error_budget"]["run_count"] == 0
    assert "intake_score_components" in payload["rows"][0]
    assert "source_status" in payload["rows"][0]


def test_storage_tier_policy_summarizes_hot_and_warm_files(tmp_path, monkeypatch) -> None:
    (tmp_path / "decisions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decision_explanations" / "shadow_default").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions" / "trade_decisions_20260101.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "decision_explanations" / "shadow_default" / "decision_explanations_20260101.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["storage_tier_policy.py", "--project-root", str(tmp_path)])
    rc = storage_tier_policy.main()
    payload = json.loads((tmp_path / "governance" / "health" / "storage_tier_policy_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["by_temperature"]["hot"]["files"] == 1
    assert payload["by_temperature"]["warm"]["files"] == 1
    assert payload["by_economic_value"]["critical"]["files"] == 1
