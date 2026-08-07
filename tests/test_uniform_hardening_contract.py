import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import system_needs_intelligence
from scripts.ops import uniform_hardening_contract as contract


NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _domain(domain_id: str, *, critical: bool) -> dict:
    return {
        "domain_id": domain_id,
        "title": domain_id.replace("_", " ").title(),
        "candidate_scopes": ["operations"],
        "failure_policy": "fail_closed",
        "live_execution_authority": "none",
        "bounded_automation": True,
        "atomic_publication_required": True,
        "evidence_grade_separate": True,
        "owner_command": ["./scripts/ops/opsctl.sh", f"{domain_id}-owner", "--json"],
        "recovery_command": ["./scripts/ops/opsctl.sh", f"{domain_id}-repair", "--json"],
        "regression_tests": [f"tests/test_{domain_id}.py"],
        "artifacts": [
            {
                "artifact_id": domain_id,
                "path": f"governance/health/{domain_id}_latest.json",
                "max_age_minutes": 15,
                "ready_statuses": ["ready"],
                "truthy_paths": ["ok"],
                "required": True,
            }
        ],
        "critical": critical,
    }


def _seed(tmp_path: Path) -> Path:
    critical = _domain("critical_runtime", critical=True)
    evidence = _domain("evidence_lane", critical=False)
    opsctl = tmp_path / "scripts" / "ops" / "opsctl.sh"
    opsctl.parent.mkdir(parents=True, exist_ok=True)
    opsctl.write_text(
        "critical_runtime-owner critical_runtime-repair evidence_lane-owner evidence_lane-repair\n",
        encoding="utf-8",
    )
    for domain_id in ("critical_runtime", "evidence_lane"):
        path = tmp_path / "tests" / f"test_{domain_id}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_contract():\n    assert True\n", encoding="utf-8")
    _write(
        tmp_path / "governance" / "health" / "critical_runtime_latest.json",
        {"timestamp_utc": NOW.isoformat(), "overall_status": "ready", "ok": True},
    )
    _write(
        tmp_path / "governance" / "health" / "evidence_lane_latest.json",
        {"timestamp_utc": NOW.isoformat(), "overall_status": "blocked", "ok": False},
    )
    config = {
        "schema_version": 1,
        "policy_id": "test_uniform_hardening",
        "minimum_structural_grade": "A+",
        "critical_runtime_domains": ["critical_runtime"],
        "evidence_domains": ["evidence_lane"],
        "required_common_controls": list(contract.COMMON_CONTROL_IDS),
        "domains": [critical, evidence],
        "control_contract": {"automatic_live_execution_authority": False},
    }
    config_path = tmp_path / "config" / "production_uniform_hardening_v1.json"
    _write(config_path, config)
    return config_path


def test_uniform_floor_can_be_ready_while_economic_evidence_remains_pending(tmp_path: Path) -> None:
    config_path = _seed(tmp_path)

    payload = contract.build_payload(tmp_path, config_path=config_path, now=NOW)

    assert payload["overall_status"] == "ready_with_evidence_debt"
    assert payload["ok"] is True
    assert payload["uniform_floor_ready"] is True
    assert payload["uniform_structural_grade"] == "A+"
    assert payload["critical_runtime_ready"] is True
    assert payload["evidence_debt_domains"] == ["evidence_lane"]
    assert payload["live_execution_authority"] is False


def test_stale_critical_artifact_fails_closed(tmp_path: Path) -> None:
    config_path = _seed(tmp_path)
    _write(
        tmp_path / "governance" / "health" / "critical_runtime_latest.json",
        {"timestamp_utc": "2026-08-07T10:00:00+00:00", "overall_status": "ready", "ok": True},
    )

    payload = contract.build_payload(tmp_path, config_path=config_path, now=NOW)

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["uniform_floor_ready"] is True
    assert payload["critical_runtime_ready"] is False
    assert "critical_runtime:critical_runtime_stale" in payload["critical_runtime_blockers"]


def test_structural_ci_exit_never_relabels_missing_runtime_evidence() -> None:
    payload = {
        "uniform_floor_ready": True,
        "critical_runtime_ready": False,
        "ok": False,
        "overall_status": "blocked",
    }

    assert contract.evaluation_exit_code(payload, structural_only=True) == 0
    assert contract.evaluation_exit_code(payload, structural_only=False) == 2
    assert payload["overall_status"] == "blocked"


def test_missing_regression_test_breaks_the_uniform_structural_floor(tmp_path: Path) -> None:
    config_path = _seed(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["domains"][0]["regression_tests"] = ["tests/test_missing_contract.py"]
    _write(config_path, config)

    payload = contract.build_payload(tmp_path, config_path=config_path, now=NOW)

    assert payload["uniform_floor_ready"] is False
    assert payload["overall_status"] == "blocked"
    assert "critical_runtime:regression_tests" in payload["structural_blockers"]


def test_repository_manifest_covers_twelve_structurally_hardened_domains() -> None:
    payload = contract.build_payload(contract.PROJECT_ROOT, config_path=contract.DEFAULT_CONFIG)

    assert payload["domain_count"] == 12
    assert payload["structurally_ready_domain_count"] == 12
    assert payload["uniform_floor_ready"] is True
    assert payload["uniform_structural_grade"] == "A+"
    assert set(payload["domain_statuses"]) == {
        "execution_boundary",
        "broker_auth",
        "market_data_sources",
        "paper_execution",
        "data_ingestion",
        "storage_recovery",
        "runtime_resources",
        "training_models",
        "profitability_research",
        "promotion_release",
        "observability_incident",
        "security_governance",
    }


def test_repository_manifest_separates_runtime_truth_from_economic_and_context_evidence() -> None:
    config = json.loads(contract.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    domains = {row["domain_id"]: row for row in config["domains"]}
    source_spec = domains["market_data_sources"]["artifacts"][0]
    paper_spec = domains["paper_execution"]["artifacts"][0]
    runtime_specs = {
        row["artifact_id"]: row for row in domains["runtime_resources"]["artifacts"]
    }

    assert source_spec["truthy_paths"] == ["source_runtime_contract.decision_critical_sources_ready"]
    assert source_spec["grade_requirements"] == {"source_control_grade": "A+"}
    assert "ready_statuses" not in source_spec
    assert "ok" not in paper_spec["truthy_paths"]
    assert "gates.paper_broker_truth_reconciliation.ok" in paper_spec["truthy_paths"]
    assert "gates.decision_replay_harness.ok" not in paper_spec["truthy_paths"]
    assert runtime_specs["runtime_throttle"]["ready_statuses"] == ["ready", "advisory"]
    assert runtime_specs["runtime_throttle"]["truthy_paths"] == ["ok"]


def test_system_needs_surfaces_critical_uniform_failure_but_not_evidence_only_debt() -> None:
    critical = system_needs_intelligence._need_from_uniform_hardening(
        {
            "uniform_floor_ready": True,
            "critical_runtime_ready": False,
            "structural_blockers": [],
            "critical_runtime_blockers": ["paper_execution:runtime_paper_regression_stale"],
            "recommended_recovery_commands": [
                ["./scripts/ops/opsctl.sh", "runtime-paper-regression-guard", "--json"]
            ],
        }
    )
    evidence_only = system_needs_intelligence._need_from_uniform_hardening(
        {
            "uniform_floor_ready": True,
            "critical_runtime_ready": True,
            "structural_blockers": [],
            "critical_runtime_blockers": [],
            "evidence_debt_domains": ["profitability_research"],
        }
    )

    assert critical[0]["blocker"] == "uniform_critical_runtime_not_ready"
    assert critical[0]["command"][1] == "runtime-paper-regression-guard"
    assert evidence_only == []
