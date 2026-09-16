import json

import pytest

from scripts import decay_monitor, paper_performance_report as report
from scripts.ops import codex_operator_bridge as bridge
from scripts.ops import paper_execution_truth_layer as truth
from scripts.ops import paper_trading_summary_pdf as summary_pdf
from scripts.ops import sendout_pdf_refresh as sendout


def build_history(tmp_path, previous_net):
    log_dir = tmp_path / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True)
    row = {
        "timestamp_utc": "2026-08-16T20:00:00Z",
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "test",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": previous_net,
        "unrealized_pnl_total": 0.0,
    }
    (log_dir / "paper_bridge_orders_20260816.jsonl").write_text(json.dumps(row) + "\n")
    return report.build_paper_performance_report(tmp_path, day="20260908")


@pytest.mark.parametrize("previous_net", [-32.44229, 18.5, 0.0])
def test_empty_current_day_never_creates_measured_profit_or_loss(
    tmp_path, previous_net
):
    payload = build_history(tmp_path, previous_net)
    assert payload["day"]["available"] is False
    assert payload["day"]["ending_net_pnl_total"] is None
    assert payload["day"]["change_vs_previous_day"] is None
    assert payload["week"]["week_to_date_change"] is None
    assert payload["week"]["rolling_change"] is None
    assert payload["week"]["week_to_date_realized_change"] is None
    assert all(
        row["change"] is None and row["available"] is False
        for row in payload["period_change_series"]
    )
    assert payload["history_daily_series"][0]["ending_net_pnl_total"] == previous_net
    assert (
        payload["accounting_views"]["active_book_snapshot"]["ending_net_pnl_total"]
        == previous_net
    )
    markdown = report.render_paper_performance_markdown(payload)
    html = report.render_paper_performance_html(
        payload,
        source_path=tmp_path / "paper.json",
        generated_utc="2026-09-08T21:00:00Z",
    )
    for text in (markdown, html):
        assert "week_to_date_change: unavailable" in text
        assert "ending_net_pnl_total: unavailable" in text


def test_measured_zero_is_distinct_from_missing(tmp_path):
    build_history(tmp_path, 0.0)
    payload = report.build_paper_performance_report(tmp_path, day="20260816")
    assert payload["day"]["available"] is True
    assert payload["week"]["week_to_date_change"] == 0.0
    assert all(
        row["available"] and row["change"] == 0.0
        for row in payload["period_change_series"]
    )
    assert report._format_pnl(0) == "0.000000"
    assert sendout._fmt_amount(0) == "+0.00"
    assert summary_pdf._money(0) != "n/a"


@pytest.mark.parametrize(
    "week",
    [
        {},
        {"available": True, "rolling_change": "bad"},
        {"available": True, "rolling_change": "NaN"},
        {"available": True, "rolling_change": float("inf")},
        {"available": False, "rolling_change": 32.44229},
        {"available": False, "rolling_change": None},
    ],
)
def test_truth_ledger_rejects_unavailable_week_even_from_legacy_report(week):
    payload = truth._build_haircut_ledger({"week": week}, [])
    assert payload["available"] is False
    assert payload["raw_week_pnl"] is None
    assert payload["realism_adjusted_week_pnl"] is None


def test_consumers_preserve_missing_and_measured_periods(tmp_path):
    paper = {
        "day": {"available": False, "ending_net_pnl_total": None},
        "week": {
            "available": False,
            "week_to_date_change": None,
            "rolling_change": None,
        },
        "period_change_series": [
            {"label": "7D", "available": False, "change": None},
            {"label": "30D", "available": True, "change": 0.0},
        ],
    }
    path = tmp_path / "governance" / "health" / "paper_performance_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(paper))
    snapshot = bridge._paper_trade_snapshot(paper)
    assert snapshot["day"]["ending_net_pnl_total"] is None
    assert snapshot["week"]["week_to_date_change"] is None
    trailing = decay_monitor.build_payload(tmp_path)["trailing_periods"]
    assert trailing[0]["change"] is None
    assert trailing[1]["change"] == 0.0
    assert summary_pdf._periods(paper)["7D"] is None
    assert summary_pdf._periods(paper)["30D"] == 0.0
    assert sendout._fmt_amount(None) == "n/a"
    assert summary_pdf._money(None) == "n/a"


def test_sendout_pdf_omits_missing_period_bars(tmp_path, monkeypatch):
    paper = build_history(tmp_path, -32.44229)
    source = tmp_path / "paper.json"
    source.write_text(json.dumps(paper))
    calls = []
    original = sendout._plot_bars

    def capture(ax, labels, values, **kwargs):
        calls.append((kwargs["title"], labels, values))
        return original(ax, labels, values, **kwargs)

    monkeypatch.setattr(sendout, "_plot_bars", capture)
    output = tmp_path / "paper.pdf"
    sendout.render_paper_performance_ready_pdf(source, output)
    assert ("Window Change Comparison", [], []) in calls
    assert output.read_bytes().startswith(b"%PDF")
