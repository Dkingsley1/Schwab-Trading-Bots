import json
from pathlib import Path

from scripts.ops import system_cleanliness_autopilot as autopilot_src
from scripts.ops import system_cleanliness_infrabot as infrabot_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_system_cleanliness_autopilot_builds_five_layer_repair_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    champion = project_root / "governance" / "champion_challenger"

    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "pressure_index": 9.4, "storage": {"retention_debt_gb": 18.0}},
    )
    _write_json(
        health / "collector_contracts_latest.json",
        {
            "required_failures": ["market_micro_context"],
            "soft_failures": ["sec_edgar_context", "extended_quant_context", "options_flow_context"],
        },
    )
    _write_json(
        health / "source_verification_latest.json",
        {"overall": {"unverified_sources": ["market_micro_context", "sec_edgar_context"]}},
    )
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "blocked",
            "training_quality_score": 68.0,
            "rollout": {"considered_bots": 1, "min_considered_bots": 4},
            "targeted_actions": {"weak_sleeves": [{"profile": "default"}]},
        },
    )
    _write_json(health / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(health / "golden_replay_regression_latest.json", {"ok": False})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"promotion_ready": False})

    payload = autopilot_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert payload["overall_status"] == "blocked"
    assert payload["layer_statuses"]["storage_backpressure"] == "blocked"
    assert payload["layer_statuses"]["collectors_sources"] == "blocked"
    assert payload["layer_statuses"]["training_eligibility"] == "blocked"
    assert "storage_pressure_clearance" in names
    assert "bounded_market_micro_sync" in names
    assert "sec_edgar_sync" in names
    assert "extended_quant_sync" in names
    assert "health_gates_recheck" in names
    assert "bot_quality_autopilot" in names
    assert "replay_hash_registry" in names
    assert payload["assigned_infrabot"] == "system_cleanliness_infrabot"


def test_system_cleanliness_autopilot_skips_downstream_apply_when_storage_blocked(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "blocked", "pressure_index": 3.0, "storage": {"retention_debt_gb": 4.0}},
    )
    _write_json(health / "collector_contracts_latest.json", {"required_failures": [], "soft_failures": []})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "blocked"})

    calls: list[list[str]] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        calls.append(list(cmd))
        return {"cmd": list(cmd), "rc": 0, "timed_out": False, "payload": {}}

    monkeypatch.setattr(autopilot_src, "_run_json", _fake_run_json)

    payload = autopilot_src.build_payload(project_root, apply=True, timeout_sec=10)

    skipped = [row for row in payload["attempts"] if row.get("skipped")]
    assert any(row["name"] == "runtime_training_snapshot" for row in skipped)
    assert not any("runtime-training-snapshot" in " ".join(cmd) for cmd in calls)
    assert any("storage-pressure-clearance" in " ".join(cmd) for cmd in calls)


def test_system_cleanliness_infrabot_wraps_autopilot_status(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "pressure_index": 0.1, "storage": {"retention_debt_gb": 0.0}},
    )
    _write_json(project_root / "governance" / "health" / "collector_contracts_latest.json", {"required_failures": [], "soft_failures": []})
    _write_json(project_root / "governance" / "health" / "source_verification_latest.json", {"overall": {"unverified_sources": []}})
    _write_json(
        project_root / "governance" / "health" / "training_quality_control_latest.json",
        {"overall_status": "ready", "rollout": {"considered_bots": 4, "min_considered_bots": 4}},
    )
    _write_json(project_root / "governance" / "health" / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "health" / "golden_replay_regression_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json", {"promotion_ready": True})

    payload = infrabot_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["supervision_contract"]["owner_bot"] == "system_cleanliness_infrabot"
    assert "collectors_sources" in payload["assigned_scope"]


def test_system_cleanliness_treats_contained_weak_sleeves_as_managed_soak_watch(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    champion = project_root / "governance" / "champion_challenger"

    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "pressure_index": 0.05, "storage": {"retention_debt_gb": 0.0}})
    _write_json(health / "collector_contracts_latest.json", {"required_failures": [], "soft_failures": []})
    _write_json(health / "source_verification_latest.json", {"overall_status": "ready", "overall": {"unverified_sources": [], "stale_sources": []}})
    _write_json(
        health / "training_quality_control_latest.json",
        {
            "overall_status": "ready",
            "training_quality_score": 100.0,
            "rollout": {"considered_bots": 1, "min_considered_bots": 1},
            "targeted_actions": {"weak_sleeves": [{"profile": "bond"}]},
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "ok": True,
            "profitability_grade": "A+",
            "controlled_profitability_grade": "A+",
            "low_grade_layer_summary": {"control_posture_grade": "A+", "active_blocker_count": 0},
        },
    )
    _write_json(health / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health / "golden_replay_regression_latest.json", {"ok": True})
    _write_json(health / "live_canary_readiness_contract_latest.json", {"live_canary_money_ready": False, "blockers": ["raw_profitability_posture_blocked"]})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"promotion_ready": False})

    payload = autopilot_src.build_payload(project_root, apply=False)

    assert payload["overall_status"] == "ready"
    assert payload["layer_statuses"]["paper_feedback"] == "ready"
    assert payload["layer_statuses"]["promotion_replay"] == "ready"
    assert payload["metrics"]["weak_sleeves_managed_by_profitability_controls"] is True
    assert payload["metrics"]["promotion_live_money_watch"] is True
    assert payload["repair_plan"] == []
