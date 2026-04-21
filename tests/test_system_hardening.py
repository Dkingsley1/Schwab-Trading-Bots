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
