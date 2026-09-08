import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.live_canary_preflight import (
    REQUIRED_OPERATOR_CONFIRMATIONS,
    RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS,
    evaluate_live_canary_preflight,
)
from scripts.ops import live_canary_preflight as preflight_cli


def _write(path: Path, payload: dict, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_ready_preflight(
    project_root: Path, *, now: datetime
) -> tuple[dict[str, str], Path]:
    account_reference = "candidate-account-hash-ref"
    policy = {
        "live_execution_risk_firewall": {
            "account_reference_env": "SCHWAB_ACCOUNT_HASH",
            "canary_plan_path": "config/live_canary_micro_policy_v1.json",
            "production_candidate_state_path": "governance/runtime/production_candidate_state.json",
            "account_policy_registry_path": "config/account_policy_registry.json",
            "account_position_study_path": "governance/health/account_position_study_latest.json",
            "live_canary_operator_attestation_path": "governance/runtime/live_canary_operator_attestation.json",
            "risk_service_boundary_path": "governance/risk/risk_service_boundary_latest.json",
            "live_order_ledger_control_path": "governance/health/live_order_ledger_control_latest.json",
            "release_freeze_guard_path": "governance/health/release_freeze_guard_latest.json",
            "trading_tax_ledger_path": "governance/tax/trading_tax_ledger_{year}_latest.json",
            "max_account_preflight_age_seconds": 120,
            "max_risk_boundary_age_seconds": 900,
            "max_live_order_ledger_age_seconds": 120,
            "max_release_guard_age_seconds": 900,
            "max_tax_ledger_age_seconds": 86400,
            "require_primary_equity_session_open": False,
        }
    }
    _write(project_root / "config" / "production_readiness_control_v1.json", policy)
    _write(
        project_root / "config" / "live_canary_micro_policy_v1.json",
        {
            "account_capital_usd": 200,
            "account_policy_key": "schwab_cash_account_1",
            "execution_route_id": "dividend_liquid_etf_candidate_v1",
            "activation_contract": {
                "max_operator_attestation_hours": 4,
                "max_allowlist_duration_hours": 4,
            },
            "stages": [{"stage": 1, "symbols": ["SCHD"]}],
        },
    )
    _write(
        project_root / "config" / "account_policy_registry.json",
        {
            "schema_version": 2,
            "account_slots": [
                {
                    "account_policy_key": "schwab_cash_account_1",
                    "canary_candidate": True,
                    "canary_cap_usd": 200,
                    "borrowing_allowed": False,
                    "allowed_live_routes": ["dividend_liquid_etf_candidate_v1"],
                    "env_names": ["SCHWAB_CASH_ACCOUNT_1_HASH"],
                }
            ],
        },
        mode=0o600,
    )
    _write(
        project_root / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "pc-test"},
    )
    account_study_path = (
        project_root / "governance" / "health" / "account_position_study_latest.json"
    )
    _write(
        account_study_path,
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "accounts": [
                {
                    "account_policy_key": "schwab_cash_account_1",
                    "borrowing_allowed": False,
                    "flags": {"closing_only": False},
                    "canary_preflight": {"blockers": []},
                    "account_capability_truth": {
                        "balance_truth": {
                            "cash_balance": 200,
                            "cash_available_for_trading": 200,
                            "pending_deposits": 0,
                        },
                        "debit_truth": {"requires_broker_ui_confirmation": True},
                        "broker_call_truth": {"in_call": False},
                        "position_collateral_truth": {
                            "uncovered_short_option_count": 0
                        },
                    },
                }
            ],
        },
    )
    _write(
        project_root / "governance" / "risk" / "risk_service_boundary_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "input_health": {"sources_ready": True},
        },
    )
    _write(
        project_root
        / "governance"
        / "health"
        / "live_order_ledger_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "overall_status": "ready",
            "unresolved_intent_count": 0,
        },
    )
    manifest_path = (
        project_root
        / "governance"
        / "releases"
        / "immutable_release_manifest_latest.json"
    )
    manifest_base = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "release_identity": {
            "branch": "test",
            "commit": "a" * 40,
            "tracked_tree_receipt_sha256": "b" * 64,
            "tags_at_head": [],
        },
        "rollback": {"reference": "a" * 40, "command": "rollback"},
        "freeze_window": {
            "active": True,
            "started_at_utc": now.isoformat(),
            "ends_at_utc": (now + timedelta(days=1)).isoformat(),
            "reason": "test",
        },
        "live_execution_authority": False,
    }
    manifest = {
        **manifest_base,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_base, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    _write(manifest_path, manifest)
    _write(
        project_root / "governance" / "health" / "release_freeze_guard_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "immutable_release_boundary": {
                "ready": True,
                "manifest_path": str(manifest_path),
            },
            "git_integrity": {
                "ready": True,
                "commit": "a" * 40,
                "tracked_tree_receipt_sha256": "b" * 64,
            },
        },
    )
    _write(
        project_root
        / "governance"
        / "tax"
        / f"trading_tax_ledger_{now.year}_latest.json",
        {"timestamp_utc": now.isoformat(), "events": []},
    )
    tax_path = (
        project_root
        / "governance"
        / "tax"
        / f"trading_tax_ledger_{now.year}_latest.json"
    )
    attestation = {
        "schema_version": 1,
        "candidate_id": "pc-test",
        "account_policy_key": "schwab_cash_account_1",
        "execution_route_id": "dividend_liquid_etf_candidate_v1",
        "account_reference_sha256": hashlib.sha256(
            account_reference.encode("utf-8")
        ).hexdigest(),
        "account_study_sha256": _sha(account_study_path),
        "trading_tax_ledger_sha256": _sha(tax_path),
        "issued_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
        "settled_cash_usd": 200,
        "live_execution_authority": False,
    }
    for field in REQUIRED_OPERATOR_CONFIRMATIONS:
        attestation[field] = True
    attestation_path = (
        project_root
        / "governance"
        / "runtime"
        / "live_canary_operator_attestation.json"
    )
    _write(attestation_path, attestation, mode=0o600)
    return {
        "SCHWAB_ACCOUNT_HASH": account_reference,
        "SCHWAB_CASH_ACCOUNT_1_HASH": account_reference,
    }, attestation_path


def _convert_ready_seed_to_roth(
    project_root: Path,
    *,
    env: dict[str, str],
    attestation_path: Path,
) -> dict[str, str]:
    policy_key = "schwab_roth_ira_primary"
    plan_path = project_root / "config" / "live_canary_micro_policy_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["account_policy_key"] = policy_key
    plan["account_constraints"] = {
        "required_account_kind": "roth_ira",
        "required_tax_wrapper": "roth_ira",
        "cash_only": True,
        "existing_positions_authority": "observe_only",
        "new_contribution_assumed": False,
        "cross_account_wash_sale_review_required": True,
        "retirement_account_loss_capacity_review_required": True,
    }
    _write(plan_path, plan)

    registry_path = project_root / "config" / "account_policy_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    slot = registry["account_slots"][0]
    slot.update(
        {
            "account_policy_key": policy_key,
            "account_type": "roth",
            "cash_only_live_budget": True,
            "existing_positions_authority": "observe_only",
            "env_names": ["SCHWAB_ROTH_ACCOUNT_HASH"],
        }
    )
    _write(registry_path, registry, mode=0o600)

    study_path = (
        project_root / "governance" / "health" / "account_position_study_latest.json"
    )
    study = json.loads(study_path.read_text(encoding="utf-8"))
    account = study["accounts"][0]
    account["account_policy_key"] = policy_key
    account["operator_account_kind"] = "roth_ira"
    account["tax_wrapper"] = "roth_ira"
    account["account_capability_truth"]["operator_classification"] = {
        "account_kind": "roth_ira",
        "tax_wrapper": "roth_ira",
    }
    _write(study_path, study)

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["account_policy_key"] = policy_key
    attestation["account_study_sha256"] = _sha(study_path)
    for field in RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS:
        attestation.pop(field, None)
    _write(attestation_path, attestation, mode=0o600)

    account_reference = env["SCHWAB_ACCOUNT_HASH"]
    return {
        "SCHWAB_ACCOUNT_HASH": account_reference,
        "SCHWAB_ROTH_ACCOUNT_HASH": account_reference,
    }


def test_ready_preflight_binds_candidate_account_cash_and_release(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)

    result = evaluate_live_canary_preflight(
        tmp_path,
        symbol="SCHD",
        action="BUY",
        env=env,
        now=now,
    )

    assert result["ready"] is True
    assert result["account_reference_matches"] is True
    assert result["risk_boundary_ready"] is True
    assert result["immutable_release_manifest_ready"] is True
    assert result["live_execution_authority"] is False


def test_wrong_live_account_fails_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    env["SCHWAB_ACCOUNT_HASH"] = "different-account-hash-ref"

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert "live_account_not_designated_canary_account" in result["blockers"]
    assert "operator_attestation_account_reference_mismatch" in result["blockers"]


def test_missing_operator_attestation_fails_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, attestation_path = _seed_ready_preflight(tmp_path, now=now)
    attestation_path.unlink()

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert "live_canary_operator_attestation_missing_or_invalid" in result["blockers"]
    assert "operator_attestation_candidate_mismatch" not in result["blockers"]
    assert result["operator_attestation_ready"] is False


def test_roth_canary_requires_retirement_specific_operator_confirmations(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, attestation_path = _seed_ready_preflight(tmp_path, now=now)
    env = _convert_ready_seed_to_roth(
        tmp_path,
        env=env,
        attestation_path=attestation_path,
    )

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert result["retirement_account"] is True
    assert result["tax_wrapper"] == "roth_ira"
    assert "operator_attestation_confirmations_incomplete" in result["blockers"]
    assert set(RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS).issubset(
        result["missing_operator_confirmations"]
    )


def test_operator_cli_requires_explicit_roth_risk_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    now = datetime.now(timezone.utc)
    env, attestation_path = _seed_ready_preflight(tmp_path, now=now)
    env = _convert_ready_seed_to_roth(
        tmp_path,
        env=env,
        attestation_path=attestation_path,
    )
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    blocked = preflight_cli._issue_attestation(
        tmp_path,
        settled_cash_usd=200.0,
        duration_minutes=60,
        confirmation=preflight_cli.CONFIRMATION_PHRASE,
        confirm_all=True,
    )
    assert blocked["ok"] is False
    assert blocked["error"] == "explicit_retirement_account_risk_confirmation_required"

    issued = preflight_cli._issue_attestation(
        tmp_path,
        settled_cash_usd=200.0,
        duration_minutes=60,
        confirmation=preflight_cli.CONFIRMATION_PHRASE,
        confirm_all=True,
        confirm_retirement_account_risk=True,
    )
    assert issued["ok"] is True
    attestation = json.loads(
        Path(issued["attestation_path"]).read_text(encoding="utf-8")
    )
    assert attestation["retirement_account"] is True
    assert attestation["tax_wrapper"] == "roth_ira"
    assert all(
        attestation[field] is True
        for field in RETIREMENT_ACCOUNT_OPERATOR_CONFIRMATIONS
    )


def test_stale_risk_boundary_fails_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    risk_path = tmp_path / "governance" / "risk" / "risk_service_boundary_latest.json"
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    risk["timestamp_utc"] = (now - timedelta(hours=1)).isoformat()
    _write(risk_path, risk)

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert "risk_service_boundary_not_ready" in result["blockers"]


def test_recent_unreviewed_same_symbol_disposition_fails_closed(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    tax_path = (
        tmp_path / "governance" / "tax" / f"trading_tax_ledger_{now.year}_latest.json"
    )
    _write(
        tax_path,
        {
            "timestamp_utc": now.isoformat(),
            "events": [
                {
                    "event_id": "sale-1",
                    "symbol": "SCHD",
                    "action": "SELL",
                    "tax_event_kind": "disposition",
                    "transaction_date": (now - timedelta(days=5)).isoformat(),
                    "tax_treatment": "taxable",
                    "wash_sale_status": "unknown",
                    "realized_gain_loss_usd": -1.0,
                }
            ],
        },
    )

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert "same_symbol_wash_sale_review_required" in result["blockers"]


def test_tampered_release_manifest_fails_closed(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 15, tzinfo=timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    manifest_path = (
        tmp_path / "governance" / "releases" / "immutable_release_manifest_latest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_identity"]["commit"] = "c" * 40
    _write(manifest_path, manifest)

    result = evaluate_live_canary_preflight(
        tmp_path, symbol="SCHD", action="BUY", env=env, now=now
    )

    assert result["ready"] is False
    assert "immutable_release_boundary_not_ready" in result["blockers"]


def test_operator_cli_issues_private_bound_attestation_and_allowlist(
    tmp_path: Path, monkeypatch
) -> None:
    now = datetime.now(timezone.utc)
    env, _ = _seed_ready_preflight(tmp_path, now=now)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    attestation_result = preflight_cli._issue_attestation(
        tmp_path,
        settled_cash_usd=200.0,
        duration_minutes=60,
        confirmation=preflight_cli.CONFIRMATION_PHRASE,
        confirm_all=True,
    )

    assert attestation_result["ok"] is True
    attestation_path = Path(attestation_result["attestation_path"])
    assert attestation_path.stat().st_mode & 0o077 == 0
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["account_policy_key"] == "schwab_cash_account_1"
    assert attestation["trading_tax_ledger_sha256"]
    assert attestation["live_execution_authority"] is False

    allowlist_result = preflight_cli._issue_allowlist(
        tmp_path,
        stage=1,
        duration_minutes=60,
        confirmation=preflight_cli.CONFIRMATION_PHRASE,
        confirm_all=True,
    )

    assert allowlist_result["ok"] is True
    allowlist_path = Path(allowlist_result["allowlist_path"])
    assert allowlist_path.stat().st_mode & 0o077 == 0
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    assert allowlist["account_policy_key"] == "schwab_cash_account_1"
    assert allowlist["execution_route_id"] == "dividend_liquid_etf_candidate_v1"
    assert allowlist["account_reference_sha256"]
    assert allowlist["operator_attestation_sha256"] == _sha(attestation_path)
    assert allowlist["live_execution_authority"] is False
