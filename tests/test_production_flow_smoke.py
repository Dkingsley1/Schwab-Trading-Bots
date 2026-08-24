from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import production_flow_smoke
from scripts.ops import production_excellence_control
from scripts.ops import source_mutation_guard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_flow_smoke_passes_current_contract() -> None:
    payload = production_flow_smoke.build_payload(PROJECT_ROOT)

    assert payload["ok"] is True
    names = {item["name"] for item in payload["checks"]}
    assert "registry_source_write_guard" in names
    assert "showcase_generated_artifact_flow" in names
    assert "stale_latest_ticker_universe_contract" in names
    assert "ci_production_smoke_coverage" in names
    policy_check = next(
        item
        for item in payload["checks"]
        if item["name"] == "deployment_healing_credential_promotion_policies"
    )
    assert (
        policy_check["conditions"]["use_mode_has_operator_grade_personal_autonomy"]
        is True
    )
    ci_check = next(
        item
        for item in payload["checks"]
        if item["name"] == "ci_production_smoke_coverage"
    )
    assert ci_check["command_validity_bot_in_ci"] is True
    assert ci_check["commands_hygiene_bot_in_ci"] is True
    assert ci_check["use_mode_compliance_guard_in_ci"] is True
    assert ci_check["production_hardening_watch_in_ci"] is True
    assert ci_check["infrabot_library_self_awareness_control_in_ci"] is True
    assert ci_check["paper_400_ramp_control_in_ci"] is True
    assert ci_check["runtime_throttle_control_in_ci"] is True
    assert ci_check["production_level_upgrade_hardener_control_in_ci"] is True
    assert ci_check["production_quality_control_in_ci"] is True
    assert ci_check["production_quality_slo_guard_in_ci"] is True
    assert ci_check["uniform_hardening_contract_in_ci"] is True


def test_uniform_contract_sources_are_protected_from_runtime_mutation() -> None:
    assert (
        "scripts/ops/uniform_hardening_contract.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "config/production_uniform_hardening_v1.json"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/ops/production_resilience_control.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "config/production_resilience_v1.json"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/observability_exporter.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/ops/soak_reliability_sentinel.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/ops/readiness_evidence_refresh.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/risk_service_boundary.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/ops/market_replay_fill_capture.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        "scripts/ops/storage_switch_orchestrator.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )


def test_ticker_contract_ignores_runtime_universe_env(monkeypatch) -> None:
    monkeypatch.setenv("TICKER_UNIVERSE_SLOW_TIER_DEFER_ON_STORAGE_PRESSURE", "1")
    monkeypatch.setenv(
        "TICKER_UNIVERSE_STANDARD_SYMBOLS", ",".join(f"TST{i}" for i in range(501))
    )

    payload = production_flow_smoke.check_ticker_universe_contract()

    assert payload["ok"] is True
    assert payload["pressure_symbol_count"] == 500


def test_source_mutation_guard_reports_clean_tmp_repo(tmp_path) -> None:
    protected = ("master_bot_registry.json", "README.md")
    for rel_path in protected:
        (tmp_path / rel_path).write_text("clean\n", encoding="utf-8")

    import subprocess

    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, text=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", *protected],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = source_mutation_guard.build_payload(tmp_path, protected_paths=protected)

    assert payload["ok"] is True
    assert payload["dirty_count"] == 0


def test_source_mutation_guard_reports_dirty_tmp_repo(tmp_path) -> None:
    protected = ("master_bot_registry.json",)
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps({"before": True}), encoding="utf-8"
    )

    import subprocess

    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, text=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", *protected],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    (tmp_path / "master_bot_registry.json").write_text(
        json.dumps({"after": True}), encoding="utf-8"
    )

    payload = source_mutation_guard.build_payload(tmp_path, protected_paths=protected)

    assert payload["ok"] is False
    assert payload["dirty_count"] == 1
    assert "master_bot_registry.json" in payload["dirty_entries"][0]


def _candidate_config(tmp_path: Path) -> tuple[dict, Path]:
    config = {
        "candidate": {
            "state_path": "governance/runtime/production_candidate_state.json",
            "event_log_path": "governance/evidence/production_candidate_events.jsonl",
            "minimum_change_reason_chars": 12,
            "scope_globs": {"operations": ["ops/**/*.py"]},
        }
    }
    config_path = tmp_path / "config" / "production_excellence_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path


def _committed_guard_repo(tmp_path: Path) -> Path:
    source = tmp_path / "ops" / "guard.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    import subprocess

    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, text=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", "ops/guard.py"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    return source


def test_source_mutation_guard_accepts_matching_reviewed_candidate(tmp_path) -> None:
    source = _committed_guard_repo(tmp_path)
    config, config_path = _candidate_config(tmp_path)
    production_excellence_control.manage_candidate(tmp_path, config, initialize=True)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    production_excellence_control.manage_candidate(
        tmp_path,
        config,
        accept_change=True,
        change_reason="Reviewed source guard update",
    )

    payload = source_mutation_guard.build_payload(
        tmp_path,
        protected_paths=("ops/guard.py",),
        candidate_config_path=config_path,
    )

    assert payload["ok"] is True
    assert payload["dirty_count"] == 0
    assert payload["observed_dirty_count"] == 1
    assert payload["accepted_candidate_dirty_count"] == 1
    assert (
        payload["candidate_acceptance"]["reason"]
        == "accepted_candidate_matches_worktree"
    )


def test_source_mutation_guard_blocks_post_acceptance_drift(tmp_path) -> None:
    source = _committed_guard_repo(tmp_path)
    config, config_path = _candidate_config(tmp_path)
    production_excellence_control.manage_candidate(tmp_path, config, initialize=True)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    production_excellence_control.manage_candidate(
        tmp_path,
        config,
        accept_change=True,
        change_reason="Reviewed source guard update",
    )
    source.write_text("VALUE = 3\n", encoding="utf-8")

    payload = source_mutation_guard.build_payload(
        tmp_path,
        protected_paths=("ops/guard.py",),
        candidate_config_path=config_path,
    )

    assert payload["ok"] is False
    assert payload["dirty_count"] == 1
    assert payload["accepted_candidate_dirty_count"] == 0
    assert payload["candidate_acceptance"]["changed_scopes"] == ["operations"]


def test_source_mutation_guard_blocks_tampered_candidate_event_chain(tmp_path) -> None:
    source = _committed_guard_repo(tmp_path)
    config, config_path = _candidate_config(tmp_path)
    initialized = production_excellence_control.manage_candidate(
        tmp_path, config, initialize=True
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    accepted = production_excellence_control.manage_candidate(
        tmp_path,
        config,
        accept_change=True,
        change_reason="Reviewed source guard update",
    )
    event_path = Path(accepted["event_path"])
    rows = event_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[-1])
    event["change_reason"] = "tampered after acceptance"
    rows[-1] = json.dumps(event)
    event_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    payload = source_mutation_guard.build_payload(
        tmp_path,
        protected_paths=("ops/guard.py",),
        candidate_config_path=config_path,
    )

    assert initialized["event_chain"]["ok"] is True
    assert payload["ok"] is False
    assert payload["dirty_count"] == 1
    assert payload["candidate_acceptance"]["event_chain_valid"] is False
    assert payload["candidate_acceptance"]["reason"] == "candidate_event_chain_invalid"


def test_source_mutation_guard_blocks_unscoped_dirty_source(tmp_path) -> None:
    source = _committed_guard_repo(tmp_path)
    config, config_path = _candidate_config(tmp_path)
    production_excellence_control.manage_candidate(tmp_path, config, initialize=True)
    readme = tmp_path / "README.md"
    readme.write_text("before\n", encoding="utf-8")

    import subprocess

    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "readme",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    readme.write_text("after\n", encoding="utf-8")

    payload = source_mutation_guard.build_payload(
        tmp_path,
        protected_paths=("README.md",),
        candidate_config_path=config_path,
    )

    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert payload["ok"] is False
    assert payload["dirty_count"] == 1
    assert payload["candidate_acceptance"]["ready"] is True
    assert payload["candidate_acceptance"]["entry_scope_coverage"] == {
        " M README.md": []
    }
