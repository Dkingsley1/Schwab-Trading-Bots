from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.schwab_broker_boundary_control import (
    build_boundary_signature,
    build_payload,
    compare_boundary_signatures,
    snapshot_complete,
)


def _inputs(access: str = "limited_margin") -> tuple[dict, dict, dict]:
    study = {
        "ok": True,
        "accounts": [
            {
                "account_policy_key": "schwab_cash_account_1",
                "account_type": "MARGIN",
                "account_capability_truth": {
                    "operator_classification": {
                        "account_policy_key": "schwab_cash_account_1",
                        "account_kind": "cash",
                        "tax_wrapper": "taxable",
                        "tax_treatment": "taxable",
                        "trading_access": access,
                        "borrowing_allowed": False,
                        "margin_interest_possible": False,
                        "option_access": "covered_only",
                    },
                    "provider_account": {
                        "provider_account_type": "MARGIN",
                        "fields": {
                            "type": "MARGIN",
                            "isIntradayMargin": True,
                        },
                    },
                    "provider_field_inventory": {
                        "account": ["type", "isIntradayMargin"],
                        "current": ["cashBalance"],
                    },
                },
            }
        ],
    }
    policy = {
        "ok": True,
        "account_policy_context": {
            "configured_account_slots": [
                {
                    "account_policy_key": "schwab_cash_account_1",
                    "account_type": "cash",
                    "trading_access": access,
                }
            ]
        },
    }
    snapshot = {
        "ok": True,
        "account_count": 1,
        "discovered_account_count": 1,
        "failed_account_count": 0,
        "account_snapshot_partial": False,
        "account_snapshot_mode": "connected_account_aggregate",
    }
    return study, policy, snapshot


def _write_inputs(root: Path, access: str = "limited_margin") -> None:
    study, policy, snapshot = _inputs(access)
    health = root / "governance" / "health"
    runtime = root / "governance" / "runtime"
    health.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    (health / "account_position_study_latest.json").write_text(
        json.dumps(study), encoding="utf-8"
    )
    (health / "account_policy_context_latest.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    (health / "schwab_account_snapshot_refresh_latest.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    (runtime / "production_candidate_state.json").write_text(
        json.dumps({"live_execution_authority": False}), encoding="utf-8"
    )


def test_partial_snapshot_is_not_complete() -> None:
    _, _, snapshot = _inputs()
    assert snapshot_complete(snapshot) is True
    snapshot["failed_account_count"] = 1
    snapshot["account_snapshot_partial"] = True
    assert snapshot_complete(snapshot) is False


def test_signature_change_identifies_capability_path() -> None:
    study, policy, snapshot = _inputs("limited_margin")
    baseline = build_boundary_signature(study, policy, snapshot)
    study2, policy2, snapshot2 = _inputs("full_margin")
    current = build_boundary_signature(study2, policy2, snapshot2)

    changes = compare_boundary_signatures(baseline, current)

    assert any("trading_access" in row["path"] for row in changes)


def test_control_bootstraps_then_quarantines_unreviewed_drift(tmp_path: Path) -> None:
    _write_inputs(tmp_path, "limited_margin")
    first = build_payload(tmp_path, apply=True)

    assert first["ok"] is True
    assert first["baseline_bootstrapped"] is True

    _write_inputs(tmp_path, "full_margin")
    second = build_payload(tmp_path, apply=True)
    quarantine = json.loads(
        (
            tmp_path
            / "governance"
            / "health"
            / "SCHWAB_BROKER_BOUNDARY_QUARANTINE.json"
        ).read_text(encoding="utf-8")
    )

    assert second["ok"] is False
    assert second["quarantine_active"] is True
    assert quarantine["active"] is True
    assert second["live_execution_authority"] is False


def test_explicit_review_accepts_drift_only_with_reason(tmp_path: Path) -> None:
    _write_inputs(tmp_path, "limited_margin")
    build_payload(tmp_path, apply=True)
    _write_inputs(tmp_path, "full_margin")

    accepted = build_payload(
        tmp_path,
        apply=True,
        accept_baseline=True,
        reason="operator reviewed limited-to-full margin test fixture",
    )

    assert accepted["ok"] is True
    assert accepted["baseline_accepted"] is True
    assert accepted["change_count"] == 0
