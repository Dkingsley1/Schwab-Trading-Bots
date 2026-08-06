import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import production_excellence_control as control


NOW = datetime(2026, 8, 4, 16, 0, tzinfo=timezone.utc)


def _write_config(project_root: Path) -> Path:
    config = {
        "policy_id": "test-production-excellence",
        "candidate": {
            "state_path": "governance/runtime/production_candidate_state.json",
            "event_log_path": "governance/evidence/production_candidate_events.jsonl",
            "minimum_change_reason_chars": 12,
            "scope_globs": {
                "operations": ["ops/**/*.py"],
                "strategy": ["strategy/**/*.py"],
            },
            "soak_scopes": ["operations", "strategy"],
            "profitability_scopes": ["strategy"],
        },
        "soak": {"artifact": "governance/health/soak.json", "required_hours": 720, "checkpoint_hours": 168},
        "recovery": {"artifact": "governance/health/recovery.json", "required_drills": []},
        "live_execution": {"required_source_paths": [], "allowed_asset_types": ["EQUITY"], "allowed_instructions": ["BUY", "SELL"]},
        "fill_evidence": {"artifact": "governance/health/fills.json"},
        "promotion": {"artifact": "governance/health/promotion.json", "packet_artifact": "governance/health/packet.json"},
        "profitability": {"performance_artifact": "governance/health/performance.json", "control_artifact": "governance/health/profitability.json"},
        "canary": {"control_artifact": "governance/health/canary.json", "rollout_artifact": "governance/health/rollout.json"},
        "grading_integrity": {"a_plus_requires_all_checks": True, "missing_evidence_score": 0},
        "institutional_operations": {},
    }
    path = project_root / "config" / "production_excellence_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _seed_sources(project_root: Path) -> None:
    for rel in ("ops/health.py", "strategy/model.py"):
        path = project_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")


def test_missing_candidate_and_evidence_can_never_report_a_plus(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _seed_sources(tmp_path)

    payload = control.build_payload(tmp_path, config_path=config_path, now=NOW)

    assert payload["ten_out_of_ten_ready"] is False
    assert payload["live_money_consideration_ready"] is False
    assert payload["overall_grade"] != "A+"
    assert payload["candidate"]["candidate_ready"] is False
    assert "p01_frozen_candidate" in payload["blocked_pillars"]


def test_candidate_drift_requires_reasoned_acceptance_and_resets_only_affected_scope(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _seed_sources(tmp_path)
    initialized = control.build_payload(
        tmp_path,
        config_path=config_path,
        initialize_candidate=True,
        now=NOW,
    )
    state = initialized["candidate"]
    assert state["candidate_ready"] is True
    assert state["generation"] == 1
    initial_windows = dict(state["scope_windows_started_utc"])

    (tmp_path / "strategy" / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    drifted = control.build_payload(tmp_path, config_path=config_path, now=NOW + timedelta(hours=1))
    assert drifted["candidate"]["candidate_drift"] is True
    assert drifted["candidate"]["changed_scopes"] == ["strategy"]

    refused = control.build_payload(
        tmp_path,
        config_path=config_path,
        accept_candidate_change=True,
        change_reason="too short",
        now=NOW + timedelta(hours=1),
    )
    assert "change_reason_shorter" in refused["candidate"]["operation_error"]
    assert refused["candidate"]["generation"] == 1

    accepted = control.build_payload(
        tmp_path,
        config_path=config_path,
        accept_candidate_change=True,
        change_reason="Recalibrated strategy threshold after review",
        now=NOW + timedelta(hours=2),
    )
    accepted_candidate = accepted["candidate"]
    assert accepted_candidate["candidate_ready"] is True
    assert accepted_candidate["generation"] == 2
    assert accepted_candidate["candidate_drift"] is False
    assert accepted_candidate["scope_windows_started_utc"]["operations"] == initial_windows["operations"]
    assert accepted_candidate["scope_windows_started_utc"]["strategy"] == (NOW + timedelta(hours=2)).isoformat()
    assert accepted_candidate["event_chain"]["event_count"] == 2


def test_candidate_event_log_tampering_blocks_candidate(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _seed_sources(tmp_path)
    initialized = control.build_payload(
        tmp_path,
        config_path=config_path,
        initialize_candidate=True,
        now=NOW,
    )
    event_path = Path(initialized["candidate"]["event_path"])
    row = json.loads(event_path.read_text(encoding="utf-8").splitlines()[0])
    row["change_reason"] = "tampered"
    event_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = control.build_payload(tmp_path, config_path=config_path, now=NOW + timedelta(minutes=5))

    assert payload["candidate"]["candidate_ready"] is False
    assert payload["candidate"]["event_chain"]["ok"] is False
    pillar = next(item for item in payload["pillars"] if item["pillar_id"] == "p01_frozen_candidate")
    assert "candidate_event_chain_valid" in pillar["failed_checks"]


def test_missing_event_log_cannot_be_accepted_as_ordinary_drift(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _seed_sources(tmp_path)
    initialized = control.build_payload(
        tmp_path,
        config_path=config_path,
        initialize_candidate=True,
        now=NOW,
    )
    Path(initialized["candidate"]["event_path"]).unlink()
    (tmp_path / "strategy" / "model.py").write_text("VALUE = 2\n", encoding="utf-8")

    refused = control.build_payload(
        tmp_path,
        config_path=config_path,
        accept_candidate_change=True,
        change_reason="Accept strategy adjustment after review",
        now=NOW + timedelta(hours=1),
    )

    assert refused["candidate"]["candidate_ready"] is False
    assert refused["candidate"]["operation_error"] == "candidate_state_event_chain_head_mismatch"


def test_explicit_event_chain_recovery_resets_every_window(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _seed_sources(tmp_path)
    initialized = control.build_payload(
        tmp_path,
        config_path=config_path,
        initialize_candidate=True,
        now=NOW,
    )
    Path(initialized["candidate"]["event_path"]).unlink()
    recovery_time = NOW + timedelta(hours=4)

    recovered = control.build_payload(
        tmp_path,
        config_path=config_path,
        recover_candidate_event_chain=True,
        change_reason="Recover missing candidate chain and restart all evidence clocks",
        now=recovery_time,
    )

    candidate = recovered["candidate"]
    assert candidate["candidate_ready"] is True
    assert candidate["generation"] == 2
    assert candidate["event_chain"]["event_count"] == 1
    assert set(candidate["scope_windows_started_utc"].values()) == {recovery_time.isoformat()}
    event = json.loads(Path(candidate["event_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "candidate_chain_recovery_anchor"
    assert event["recovery_evidence"]["all_evidence_windows_reset"] is True
    assert event["recovery_evidence"]["prior_state_event_chain_head"]


def test_profitability_grade_integrity_allows_honest_equal_grades() -> None:
    assert control._profitability_grade_labels_honest(
        {
            "raw_profitability_grade": "A",
            "controlled_profitability_grade": "A",
            "profitability_display_grade": "A",
        }
    ) is True
    assert control._profitability_grade_labels_honest(
        {
            "raw_profitability_grade": "C",
            "controlled_profitability_grade": "A+",
            "profitability_display_grade": "A+ controlled / C raw",
        }
    ) is True
    assert control._profitability_grade_labels_honest(
        {
            "raw_profitability_grade": "C",
            "controlled_profitability_grade": "A+",
            "profitability_display_grade": "A+",
        }
    ) is False


def test_profitability_source_match_requires_exact_hash(tmp_path: Path) -> None:
    performance = tmp_path / "performance.json"
    performance.write_text('{"executions": 10}', encoding="utf-8")
    artifact = {"path": str(performance), "fresh": True}
    payload = {
        "paper_performance_input_contract": {
            "usable_for_profitability_grade": True,
            "sha256": control._file_sha256(performance),
        }
    }

    assert control._profitability_source_matches(artifact, payload) is True
    performance.write_text('{"executions": 11}', encoding="utf-8")
    assert control._profitability_source_matches(artifact, payload) is False


def test_repository_candidate_scopes_cover_collectors_and_profitability_evidence() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads((project_root / "config" / "production_excellence_v1.json").read_text(encoding="utf-8"))
    scope_globs = config["candidate"]["scope_globs"]

    data_files = set(control._scope_files(project_root, scope_globs["data"]))
    execution_files = set(control._scope_files(project_root, scope_globs["execution"]))
    promotion_files = set(control._scope_files(project_root, scope_globs["promotion"]))

    assert project_root / "scripts" / "collect_public_policy_context.py" in data_files
    assert project_root / "scripts" / "paper_performance_report.py" in promotion_files
    assert project_root / "scripts" / "canary_rollout_guard.py" in promotion_files
    assert project_root / "scripts" / "ops" / "independent_fill_evidence_acquisition.py" in execution_files
    assert project_root / "scripts" / "ops" / "independent_fill_evidence_acquisition.py" in promotion_files
    assert project_root / "scripts" / "multiple_testing_guard.py" in promotion_files
    assert project_root / "config" / "profitability_evidence_firewall_v1.json" in promotion_files
