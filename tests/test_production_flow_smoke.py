from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import production_flow_smoke
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
    policy_check = next(item for item in payload["checks"] if item["name"] == "deployment_healing_credential_promotion_policies")
    assert policy_check["conditions"]["use_mode_has_operator_grade_personal_autonomy"] is True
    ci_check = next(item for item in payload["checks"] if item["name"] == "ci_production_smoke_coverage")
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
    assert "scripts/ops/uniform_hardening_contract.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "config/production_uniform_hardening_v1.json" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/ops/production_resilience_control.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "config/production_resilience_v1.json" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/observability_exporter.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/ops/soak_reliability_sentinel.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/ops/readiness_evidence_refresh.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/risk_service_boundary.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    assert "scripts/ops/market_replay_fill_capture.py" in source_mutation_guard.DEFAULT_PROTECTED_PATHS


def test_ticker_contract_ignores_runtime_universe_env(monkeypatch) -> None:
    monkeypatch.setenv("TICKER_UNIVERSE_SLOW_TIER_DEFER_ON_STORAGE_PRESSURE", "1")
    monkeypatch.setenv("TICKER_UNIVERSE_STANDARD_SYMBOLS", ",".join(f"TST{i}" for i in range(501)))

    payload = production_flow_smoke.check_ticker_universe_contract()

    assert payload["ok"] is True
    assert payload["pressure_symbol_count"] == 500


def test_source_mutation_guard_reports_clean_tmp_repo(tmp_path) -> None:
    protected = ("master_bot_registry.json", "README.md")
    for rel_path in protected:
        (tmp_path / rel_path).write_text("clean\n", encoding="utf-8")

    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, text=True, capture_output=True)
    subprocess.run(["git", "add", *protected], cwd=tmp_path, check=True, text=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
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
    (tmp_path / "master_bot_registry.json").write_text(json.dumps({"before": True}), encoding="utf-8")

    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, text=True, capture_output=True)
    subprocess.run(["git", "add", *protected], cwd=tmp_path, check=True, text=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    (tmp_path / "master_bot_registry.json").write_text(json.dumps({"after": True}), encoding="utf-8")

    payload = source_mutation_guard.build_payload(tmp_path, protected_paths=protected)

    assert payload["ok"] is False
    assert payload["dirty_count"] == 1
    assert "master_bot_registry.json" in payload["dirty_entries"][0]
