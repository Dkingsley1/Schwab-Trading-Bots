import json
from pathlib import Path

from scripts.ops import live_execution_rehearsal_control


def _seed_policy(project_root: Path) -> None:
    config = project_root / "config" / "production_readiness_control_v1.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "live_execution_risk_firewall": {
                    "max_quote_age_seconds": 15.0,
                    "max_account_snapshot_age_seconds": 30.0,
                    "max_spread_bps": 75.0,
                    "max_future_clock_skew_seconds": 2.0,
                    "live_execution_envelope_ttl_seconds": 15.0,
                    "require_sealed_live_execution_envelope": True,
                    "mutating_broker_retries_after_dispatch_allowed": False,
                    "allow_live_order_replace": False,
                }
            }
        ),
        encoding="utf-8",
    )


def test_live_execution_rehearsal_is_a_plus_and_validate_only(tmp_path: Path) -> None:
    _seed_policy(tmp_path)
    payload = live_execution_rehearsal_control.build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["control_grade"] == "A+"
    assert payload["implemented_control_count"] == payload["control_count"] == 14
    assert payload["authority"] == {
        "validate_only": True,
        "network_access": False,
        "broker_client_created": False,
        "paper_order_authority": False,
        "live_execution_authority": False,
        "live_orders_must_remain_disabled": True,
    }
    assert all(row["passed"] for row in payload["negative_scenarios"].values())
    assert len(payload["negative_scenarios"]) == 10
    assert len(payload["source_influences"]) == 5


def test_live_execution_rehearsal_fails_when_mutation_policy_is_relaxed(
    tmp_path: Path,
) -> None:
    _seed_policy(tmp_path)
    config_path = tmp_path / "config" / "production_readiness_control_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["live_execution_risk_firewall"][
        "mutating_broker_retries_after_dispatch_allowed"
    ] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    payload = live_execution_rehearsal_control.build_payload(tmp_path)

    assert payload["ok"] is False
    control = next(
        row
        for row in payload["controls"]
        if row["control_id"] == "13_fail_closed_policy"
    )
    assert control["implemented"] is False
    assert payload["authority"]["live_execution_authority"] is False
