import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ops import independent_fill_evidence_acquisition as acquisition


NOW = datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)


def _candidate(project_root: Path) -> None:
    path = project_root / "governance" / "runtime" / "production_candidate_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_id": "pc-test-g1",
                "generation": 1,
                "scope_windows_started_utc": {
                    "execution": "2026-08-06T16:00:00+00:00",
                    "data": "2026-08-06T16:00:00+00:00",
                    "dependencies": "2026-08-06T16:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )


def _valid_fill(*, fill_price: float = 100.1) -> dict:
    return {
        "timestamp_utc": "2026-08-06T17:00:00+00:00",
        "observed_at_utc": "2026-08-06T17:00:01+00:00",
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 2,
        "reference_price": 100.0,
        "expected_fill_price": 100.05,
        "fill_price": fill_price,
        "expected_slippage_bps": 5.0,
        "paper_fill_source": "broker_paper_fill",
        "broker": "test_broker",
        "account_mode": "paper",
        "external_fill_id": "fill-001",
        "metadata": {"source_profile": "intraday_aggressive"},
    }


def test_valid_fill_is_content_addressed_and_materialized_idempotently(tmp_path: Path) -> None:
    _candidate(tmp_path)
    inbox = tmp_path / "exports" / "independent_fill_inbox"
    inbox.mkdir(parents=True)
    (inbox / "broker.jsonl").write_text(json.dumps(_valid_fill()) + "\n", encoding="utf-8")

    first = acquisition.build_payload(tmp_path, apply=True, now=NOW)
    second = acquisition.build_payload(tmp_path, apply=True, now=NOW)

    assert first["new_ledger_records"] == 1
    assert first["candidate_eligible_ledger_records"] == 1
    assert second["new_ledger_records"] == 0
    assert second["accepted_ledger_records"] == 1
    trade_log = tmp_path / "exports" / "trade_logs" / "independent_fills" / "paper_trades_20260806.jsonl"
    rows = [json.loads(line) for line in trade_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["paper_fill_source"] == "broker_paper_fill"
    assert rows[0]["promotion_evidence_eligible"] is True


def test_model_fill_and_pre_candidate_fill_are_rejected(tmp_path: Path) -> None:
    _candidate(tmp_path)
    inbox = tmp_path / "exports" / "independent_fill_inbox"
    inbox.mkdir(parents=True)
    model_fill = _valid_fill()
    model_fill["paper_fill_source"] = "expected_fill_model"
    old_fill = _valid_fill()
    old_fill["external_fill_id"] = "fill-old"
    old_fill["timestamp_utc"] = "2026-08-06T15:00:00+00:00"
    old_fill["observed_at_utc"] = "2026-08-06T15:00:01+00:00"
    (inbox / "invalid.jsonl").write_text(
        "\n".join((json.dumps(model_fill), json.dumps(old_fill))) + "\n",
        encoding="utf-8",
    )

    payload = acquisition.build_payload(tmp_path, apply=True, now=NOW)

    assert payload["accepted_ledger_records"] == 0
    assert payload["rejected_count"] == 2
    reasons = {reason for row in payload["rejected_tail"] for reason in row["reasons"]}
    assert "model_derived_source_not_independent" in reasons
    assert "before_candidate_evidence_cutoff" in reasons


def test_reused_source_record_id_with_changed_content_fails_closed(tmp_path: Path) -> None:
    _candidate(tmp_path)
    inbox = tmp_path / "exports" / "independent_fill_inbox"
    inbox.mkdir(parents=True)
    path = inbox / "broker.jsonl"
    path.write_text(json.dumps(_valid_fill()) + "\n", encoding="utf-8")
    acquisition.build_payload(tmp_path, apply=True, now=NOW)
    path.write_text(json.dumps(_valid_fill(fill_price=101.0)) + "\n", encoding="utf-8")

    payload = acquisition.build_payload(tmp_path, apply=True, now=NOW)

    assert payload["overall_status"] == "conflict"
    assert payload["ok"] is False
    assert payload["conflict_count"] == 1
    assert payload["accepted_ledger_records"] == 1
