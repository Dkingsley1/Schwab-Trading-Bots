import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import readiness_evidence_refresh as refresh


NOW = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(artifact: str, *, allowed=(0,), max_age=15) -> dict:
    return {
        "name": "test_step",
        "script": "scripts/test_step.py",
        "artifact": artifact,
        "args": ["--json"],
        "max_age_minutes": max_age,
        "allowed_returncodes": list(allowed),
        "depends_on": [],
    }


def test_fresh_artifact_is_not_recomputed(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "test_latest.json"
    _write(artifact, {"timestamp_utc": (NOW - timedelta(minutes=2)).isoformat()})

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("fresh step should not execute")

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/test_latest.json")],
        runner=should_not_run,
        now=NOW,
    )

    assert payload["ok"] is True
    assert payload["fresh_step_count"] == 1
    assert payload["refreshed_step_count"] == 0


def test_due_step_accepts_evidence_pending_return_code(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "test_latest.json"

    def runner(*_args, **_kwargs):
        _write(artifact, {"timestamp_utc": NOW.isoformat(), "overall_status": "evidence_pending", "ok": False})
        return {"rc": 2, "stdout": json.dumps({"overall_status": "evidence_pending", "ok": False}), "stderr": "", "timed_out": False}

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/test_latest.json", allowed=(0, 2))],
        runner=runner,
        now=NOW,
    )

    assert payload["overall_status"] == "ready"
    assert payload["refreshed_step_count"] == 1
    assert payload["operational_failures"] == []
    assert payload["steps"][0]["published_status"] == "evidence_pending"


def test_timeout_is_an_operational_failure(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        return {"rc": 124, "stdout": "", "stderr": "timeout", "timed_out": True}

    payload = refresh.refresh(
        tmp_path,
        steps=[_spec("governance/health/missing.json")],
        runner=runner,
        now=NOW,
    )

    assert payload["ok"] is False
    assert payload["operational_failures"] == ["test_step"]


def test_refresh_report_cooldown_returns_without_rewriting(tmp_path: Path) -> None:
    out = tmp_path / "governance" / "health" / "readiness_evidence_refresh_latest.json"
    _write(out, {"timestamp_utc": (NOW - timedelta(minutes=2)).isoformat(), "overall_status": "ready", "ok": True})

    payload = refresh.refresh(tmp_path, steps=[], now=NOW)

    assert payload["refresh_skipped"] is True
    assert payload["write_latest"] is False
    assert payload["refresh_skip_reason"] == "cooldown_active"


def test_unattended_soak_runs_after_all_freshness_dependencies() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}

    assert set(steps["unattended_soak_readiness"]["depends_on"]) == {
        "storage_retention_unison",
        "notification_escalation_ladder",
        "livefeed_refresh_guard",
        "storage_resilience_control",
    }
    for name in steps["unattended_soak_readiness"]["depends_on"]:
        assert steps[name]["max_age_minutes"] < 180
    assert "--apply" not in steps["storage_retention_unison"]["args"]


def test_profitability_firewall_runs_after_all_hardening_evidence_producers() -> None:
    steps = {row["name"]: row for row in refresh.default_steps()}
    dependencies = set(steps["profitability_evidence_firewall"]["depends_on"])

    assert {
        "paper_execution_calibration",
        "paper_profitability_control",
        "execution_queue_stress",
        "multiple_testing_guard",
        "decay_monitor",
        "profitability_independent_validator",
        "profitability_holdout_vault",
        "profitability_benchmark_capture",
        "profitability_benchmark_hurdle",
    }.issubset(dependencies)
    assert "profitability_evidence_firewall" in steps["production_excellence"]["depends_on"]
