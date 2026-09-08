import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from core.live_canary_allowlist import evaluate_live_canary_allowlist
from core.live_order_ledger import LiveOrderLedger


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_contract(
    project_root: Path,
    *,
    candidate_id: str = "pc-current",
    symbols: list[str] | None = None
) -> Path:
    symbols = symbols or ["SCHD"]
    _write(
        project_root / "config" / "production_readiness_control_v1.json",
        {
            "live_execution_risk_firewall": {
                "canary_allowlist_path": "governance/runtime/live_canary_allowlist.json",
                "canary_plan_path": "config/live_canary_micro_policy_v1.json",
                "production_candidate_state_path": "governance/runtime/production_candidate_state.json",
                "symbol_lifecycle_path": "config/symbol_lifecycle_v1.json",
            }
        },
    )
    _write(
        project_root / "config" / "live_canary_micro_policy_v1.json",
        {
            "status": "advisory_only",
            "hard_limits": {"max_order_notional_usd": 100, "max_order_quantity": 1},
            "stages": [{"stage": 1, "symbols": ["SCHD"]}],
            "activation_contract": {"max_allowlist_duration_hours": 4},
        },
    )
    _write(
        project_root / "config" / "symbol_lifecycle_v1.json",
        {"renamed_symbols": {"SPLG": "SPYM"}},
    )
    _write(
        project_root / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": candidate_id, "accepted_at_utc": "2026-08-17T11:00:00+00:00"},
    )
    path = project_root / "governance" / "runtime" / "live_canary_allowlist.json"
    _write(
        path,
        {
            "schema_version": 1,
            "enabled": True,
            "candidate_id": candidate_id,
            "stage": 1,
            "symbols": symbols,
            "issued_at_utc": "2026-08-17T12:00:00+00:00",
            "expires_at_utc": "2026-08-17T16:00:00+00:00",
        },
    )
    return path


def test_valid_allowlist_is_candidate_bound_and_stage_bounded(tmp_path: Path) -> None:
    _write_contract(tmp_path)

    payload = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
    )

    assert payload["ready"] is True
    assert payload["candidate_matches"] is True
    assert payload["symbols"] == ["SCHD"]
    assert payload["hard_limits"]["max_order_quantity"] == 1


def test_expired_or_deprecated_allowlist_fails_closed(tmp_path: Path) -> None:
    path = _write_contract(tmp_path, symbols=["SPLG"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expires_at_utc"] = "2026-08-17T12:30:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
    )

    assert result["ready"] is False
    assert "canary_allowlist_expired_or_invalid" in result["blockers"]
    assert "canary_allowlist_symbol_not_in_stage" in result["blockers"]
    assert "canary_allowlist_contains_deprecated_symbol" in result["blockers"]


def test_overlong_or_pre_candidate_allowlist_fails_closed(tmp_path: Path) -> None:
    path = _write_contract(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issued_at_utc"] = "2026-08-17T10:00:00+00:00"
    payload["expires_at_utc"] = "2026-08-17T20:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
    )

    assert result["ready"] is False
    assert "canary_allowlist_predates_candidate_acceptance" in result["blockers"]
    assert "canary_allowlist_duration_exceeds_policy" in result["blockers"]


def test_missing_allowlist_never_inherits_live_authority(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    (tmp_path / "governance" / "runtime" / "live_canary_allowlist.json").unlink()

    payload = evaluate_live_canary_allowlist(tmp_path)

    assert payload["ready"] is False
    assert "canary_allowlist_missing" in payload["blockers"]
    assert payload["live_execution_authority"] is False


def test_required_account_and_attestation_bindings_fail_closed(tmp_path: Path) -> None:
    path = _write_contract(tmp_path)
    plan_path = tmp_path / "config" / "live_canary_micro_policy_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "account_policy_key": "schwab_cash_account_1",
            "execution_route_id": "dividend_liquid_etf_candidate_v1",
            "required_allowlist_fields": [
                "schema_version",
                "enabled",
                "candidate_id",
                "account_policy_key",
                "execution_route_id",
                "account_reference_sha256",
                "operator_attestation_sha256",
                "stage",
                "symbols",
                "issued_at_utc",
                "expires_at_utc",
            ],
        }
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    firewall_path = tmp_path / "config" / "production_readiness_control_v1.json"
    firewall = json.loads(firewall_path.read_text(encoding="utf-8"))
    firewall["live_execution_risk_firewall"][
        "account_reference_env"
    ] = "SCHWAB_ACCOUNT_HASH"
    firewall["live_execution_risk_firewall"][
        "live_canary_operator_attestation_path"
    ] = "governance/runtime/live_canary_operator_attestation.json"
    firewall_path.write_text(json.dumps(firewall), encoding="utf-8")
    account_reference = "candidate-account-hash"
    attestation_path = (
        tmp_path / "governance" / "runtime" / "live_canary_operator_attestation.json"
    )
    attestation_path.write_text("{}", encoding="utf-8")

    missing = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
        env={"SCHWAB_ACCOUNT_HASH": account_reference},
    )
    allowlist = json.loads(path.read_text(encoding="utf-8"))
    allowlist.update(
        {
            "account_policy_key": "schwab_cash_account_1",
            "execution_route_id": "dividend_liquid_etf_candidate_v1",
            "account_reference_sha256": hashlib.sha256(
                account_reference.encode("utf-8")
            ).hexdigest(),
            "operator_attestation_sha256": hashlib.sha256(
                attestation_path.read_bytes()
            ).hexdigest(),
        }
    )
    path.write_text(json.dumps(allowlist), encoding="utf-8")
    bound = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
        env={"SCHWAB_ACCOUNT_HASH": account_reference},
    )

    assert missing["ready"] is False
    assert "canary_allowlist_required_fields_missing" in missing["blockers"]
    assert bound["ready"] is True
    assert bound["account_reference_bound"] is True
    assert bound["operator_attestation_bound"] is True


def test_allowlist_spends_one_new_entry_budget_and_preserves_exits(
    tmp_path: Path,
) -> None:
    _write_contract(tmp_path)
    plan_path = tmp_path / "config" / "live_canary_micro_policy_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        {
            "account_policy_key": "schwab_roth_ira_primary",
            "execution_route_id": "dividend_liquid_etf_candidate_v1",
        }
    )
    plan["activation_contract"].update(
        {
            "require_entry_budget_enforcement": True,
            "max_new_entry_intents_per_allowlist": 1,
            "live_order_ledger_path": "governance/runtime/live_order_ledger.sqlite3",
        }
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    ledger = LiveOrderLedger(
        tmp_path / "governance" / "runtime" / "live_order_ledger.sqlite3"
    )
    ledger.reserve(
        intent_id="canary-entry-one",
        requested_quantity=1.0,
        payload={
            "symbol": "SCHD",
            "action": "BUY",
            "live_execution_envelope": {
                "candidate_id": "pc-current",
                "account_snapshot_evidence": {
                    "live_canary_preflight_receipt": {
                        "account_policy_key": "schwab_roth_ira_primary",
                        "execution_route_id": "dividend_liquid_etf_candidate_v1",
                    }
                },
            },
        },
    )

    payload = evaluate_live_canary_allowlist(
        tmp_path,
        now=datetime(2026, 8, 17, 13, tzinfo=timezone.utc),
    )

    assert payload["ready"] is False
    assert "canary_allowlist_entry_budget_exhausted" in payload["blockers"]
    assert payload["entry_budget"]["consumed_new_entry_intents"] == 1
    assert payload["entry_budget"]["remaining_new_entry_intents"] == 0
    assert payload["entry_budget"]["reduce_only_exits_preserved"] is True
