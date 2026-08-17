import json
from datetime import datetime, timezone
from pathlib import Path

from core.capability_materialization import (
    build_materialized_capabilities,
    validate_materialization_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "capability_materialization_v1.json"
DERIVATIVE_PATH = PROJECT_ROOT / "config" / "derivatives_contract_master_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repository_materialization_is_source_backed_and_ready() -> None:
    policy = _load(POLICY_PATH)
    payload = build_materialized_capabilities(
        PROJECT_ROOT,
        policy,
        _load(DERIVATIVE_PATH),
        derivative_master_path=DERIVATIVE_PATH,
        now=datetime(2026, 8, 13, 17, 0, tzinfo=timezone.utc),
    )

    assert validate_materialization_policy(policy) == []
    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["live_promotion_ready"] is True
    assert payload["ready_capability_count"] == 4
    assert set(payload["ready_capability_ids"]) == {
        "trading_calendars",
        "market_session_state",
        "derivatives_contract_master",
        "stress_scenarios",
    }
    assert set(payload["authority_contract"].values()) == {False}
    assert all(row["proof_receipt_sha256"] for row in payload["capabilities"])
    assert all(row["source_receipts"] for row in payload["capabilities"])


def test_exchange_calendar_materializes_closed_holiday_state() -> None:
    payload = build_materialized_capabilities(
        PROJECT_ROOT,
        _load(POLICY_PATH),
        _load(DERIVATIVE_PATH),
        derivative_master_path=DERIVATIVE_PATH,
        now=datetime(2026, 7, 3, 16, 0, tzinfo=timezone.utc),
    )
    rows = {
        row["calendar_id"]: row
        for row in payload["calendar_materialization"]["calendars"]
    }

    assert rows["XNYS"]["session_state"] == "closed"
    assert rows["XNAS"]["session_state"] == "closed"
    assert rows["24/7"]["session_state"] == "open"


def test_derivative_contract_master_fails_closed_on_tick_mismatch() -> None:
    master = _load(DERIVATIVE_PATH)
    master["contracts"][0]["tick_value"] = 999.0
    payload = build_materialized_capabilities(
        PROJECT_ROOT,
        _load(POLICY_PATH),
        master,
        derivative_master_path=DERIVATIVE_PATH,
        now=datetime(2026, 8, 13, 17, 0, tzinfo=timezone.utc),
    )
    rows = {row["capability_id"]: row for row in payload["capabilities"]}

    assert rows["derivatives_contract_master"]["usable"] is False
    assert payload["live_promotion_ready"] is False
    assert any("tick_value_mismatch" in item for item in payload["errors"])
