import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import base_trader
from scripts import paper_performance_report as report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def setup_report(tmp_path, monkeypatch, rows):
    now = datetime(2026, 9, 12, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(report, "_utc_now", lambda: now)
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    write_json(
        tmp_path / "governance/runtime/production_candidate_state.json",
        {
            "candidate_id": "current",
            "generation": 2,
            "accepted_at_utc": "2026-09-10T12:00:00+00:00",
            "scope_windows_started_utc": {"strategy": "2026-09-10T12:00:00+00:00"},
        },
    )
    path = tmp_path / "exports/trade_logs/paper/paper_trades_paper.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def outcome(**overrides):
    return {
        "timestamp_utc": "2026-09-12T15:00:00+00:00",
        "action": "SELL",
        "symbol": "SPY",
        "strategy": "test",
        "paper_pnl_schema_version": 3,
        "post_cost_pnl_delta": -1.5,
        "post_cost_return_bps": -2.0,
        "metadata": {"production_candidate_id": "current", "source_profile": "default"},
        **overrides,
    }


def test_timestamp_alias_is_read_without_relabeling_candidate(tmp_path, monkeypatch):
    row = outcome(timestamp="2026-09-12T15:00:00+00:00")
    row.pop("timestamp_utc")
    setup_report(tmp_path, monkeypatch, [row])
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    assert payload["accounting_views"]["candidate_forward_flow"]["sample_count"] == 1
    assert (
        payload["accounting_views"]["candidate_forward_flow"][
            "post_cost_pnl_delta_total"
        ]
        == -1.5
    )
    assert (
        payload["outcome_evidence_diagnostics"]["status"]
        == "candidate_outcomes_observed"
    )


def test_report_separates_old_missing_and_mismatched_candidate_rows(
    tmp_path, monkeypatch
):
    setup_report(
        tmp_path,
        monkeypatch,
        [
            outcome(timestamp_utc="2026-09-09T15:00:00+00:00"),
            outcome(metadata={}),
            outcome(metadata={"production_candidate_id": "older"}),
        ],
    )
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    diagnostic = payload["outcome_evidence_diagnostics"]
    assert diagnostic["source_scan_complete"] is True
    assert diagnostic["counts"]["rows_before_candidate_cutoff"] == 1
    assert diagnostic["counts"]["candidate_identity_missing_rows"] == 1
    assert diagnostic["counts"]["candidate_identity_mismatch_rows"] == 1
    assert diagnostic["candidate_bound_post_cost_rows"] == 0
    assert diagnostic["history_relabeling_authority"] is False


@pytest.mark.parametrize(
    "bad_line,key",
    [
        ("not JSON", "invalid_json_rows"),
        ("[]", "non_object_rows"),
        ('{"timestamp_utc":"bad"}', "invalid_timestamp_rows"),
    ],
)
def test_bad_source_row_is_visible_and_cannot_certify_complete_evidence(
    tmp_path, monkeypatch, bad_line, key
):
    path = setup_report(tmp_path, monkeypatch, [outcome()])
    with path.open("a") as handle:
        handle.write(bad_line + "\n")
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    assert payload["ok"] is False
    assert payload["execution_deduplication"][key] == 1
    assert payload["outcome_evidence_diagnostics"]["status"] == "source_scan_incomplete"
    assert payload["post_cost_expectancy"]["promotion_evidence_sufficient"] is False
    assert (
        "source_scan_incomplete"
        in payload["post_cost_expectancy"]["promotion_blockers"]
    )


def test_unreadable_compressed_source_is_not_silent(tmp_path, monkeypatch):
    path = setup_report(tmp_path, monkeypatch, [outcome()])
    path.with_name("paper_trades_broken.jsonl.gz").write_bytes(b"broken gzip")
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    assert payload["execution_deduplication"]["source_read_error_count"] == 1
    assert payload["outcome_evidence_diagnostics"]["source_scan_complete"] is False


@pytest.mark.parametrize("alias", [False, True])
def test_discovery_never_probes_protected_target(tmp_path, monkeypatch, alias):
    target = Path("/Volumes/VIDEO/private_trade_history")
    if alias:
        target = tmp_path / "archive_alias"
        target.symlink_to(
            "/Volumes/VIDEO/private_trade_history", target_is_directory=True
        )
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(target))
    original = Path.lstat

    def checked(path, *args, **kwargs):
        assert not str(path).casefold().startswith("/volumes/video")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked)
    audit = {}
    paths, _, _ = report._paper_source_files(tmp_path, audit=audit)
    assert paths == []
    assert audit["discovery_error_count"] == 1
    assert audit["discovery_errors"][0]["reason"] == "protected_path"


def test_runtime_holds_explain_absence_but_do_not_create_outcomes(
    tmp_path, monkeypatch
):
    setup_report(
        tmp_path, monkeypatch, [outcome(timestamp_utc="2026-09-09T15:00:00+00:00")]
    )
    health = tmp_path / "governance/health"
    write_json(
        health / "execution_lane_paper_latest.json",
        {
            "timestamp_utc": report._utc_now().isoformat(),
            "pending_rows_unknown": True,
            "execution_safety_hold": {
                "active": True,
                "reason": "paper_execution_paused_for_runtime_pressure",
            },
        },
    )
    write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": report._utc_now().isoformat(),
            "safe_hold_active": True,
            "safe_hold_reason": "paper_performance_input_not_gradeable",
        },
    )
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    diagnostic = payload["outcome_evidence_diagnostics"]
    assert diagnostic["status"] == "no_outcomes_in_candidate_time_window"
    assert diagnostic["no_outcomes_hold_requires_review"] is True
    assert diagnostic["candidate_bound_post_cost_rows"] == 0
    assert (
        diagnostic["execution_context"]["sources"]["execution"]["pending_rows_known"]
        is False
    )


def test_stale_runtime_hold_is_not_presented_as_current(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch, [])
    write_json(
        tmp_path / "governance/health/execution_lane_paper_latest.json",
        {
            "timestamp_utc": (report._utc_now() - timedelta(minutes=3)).isoformat(),
            "execution_safety_hold": {"active": True, "reason": "old_hold"},
        },
    )
    context = report._paper_execution_context(tmp_path)
    assert context["fresh_reported_holds"] == []
    assert context["sources"]["execution"]["fresh"] is False


def test_profitability_evidence_hold_does_not_claim_execution_deadlock(
    tmp_path, monkeypatch
):
    setup_report(tmp_path, monkeypatch, [])
    write_json(
        tmp_path / "governance/health/paper_runtime_profitability_controls_latest.json",
        {
            "timestamp_utc": report._utc_now().isoformat(),
            "safe_hold_active": True,
            "safe_hold_reason": "paper_performance_input_not_gradeable",
            "safe_hold_scope": "profitability_evidence_and_control_authority",
            "execution_hold_observed": False,
        },
    )
    payload = report.build_paper_performance_report(tmp_path, day="20260912")
    diagnostic = payload["outcome_evidence_diagnostics"]
    context = diagnostic["execution_context"]
    assert context["fresh_reported_holds"] == []
    assert context["fresh_reported_evidence_holds"] == [
        "paper_performance_input_not_gradeable"
    ]
    assert not diagnostic["no_outcomes_hold_requires_review"]
    assert not diagnostic["execution_authority"]


def test_paper_journal_append_failure_is_not_returned_as_success(monkeypatch):
    trader = object.__new__(base_trader.BaseTrader)
    trader.project_root = "/tmp/test_paper_evidence"
    trader.paper_log_path = "/tmp/test_paper_evidence/paper_trades_paper.jsonl"
    monkeypatch.setattr(base_trader, "safe_append_jsonl", lambda *a, **k: False)
    with pytest.raises(OSError, match="paper_trade_evidence_persistence_failed"):
        trader._record_jsonl(trader.paper_log_path, outcome())
    # Noncritical telemetry retains its existing best-effort behavior.
    assert trader._record_jsonl(
        "/tmp/test_paper_evidence/telemetry.jsonl", {"event": "observation"}
    ) == {"event": "observation"}
