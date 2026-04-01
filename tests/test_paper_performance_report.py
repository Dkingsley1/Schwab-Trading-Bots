import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.paper_performance_report as report


def test_paper_performance_report_builds_day_and_week_changes(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_performance_latest.json"
    md_file = project_root / "exports" / "reports" / "paper_performance_latest.md"
    html_file = project_root / "exports" / "reports" / "paper_performance_latest.html"
    pdf_file = project_root / "exports" / "reports" / "paper_performance_latest.pdf"
    weekly_chart = project_root / "exports" / "reports" / "paper_performance_weekly_latest.png"
    monthly_chart = project_root / "exports" / "reports" / "paper_performance_monthly_latest.png"
    quarterly_chart = project_root / "exports" / "reports" / "paper_performance_quarterly_latest.png"
    sleeves_chart = project_root / "exports" / "reports" / "paper_performance_sleeves_latest.png"

    day1 = {
        "timestamp_utc": "2026-03-18T20:00:00+00:00",
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::brain_refinery_v56_meta_ranker",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": 10.0,
        "unrealized_pnl_total": 5.0,
    }
    day2 = {
        "timestamp_utc": "2026-03-19T20:00:00+00:00",
        "symbol": "QQQ",
        "action": "SELL",
        "strategy": "paper_mirror::brain_refinery_v43_intraday_ultrafast_proxy",
        "metadata": {"source_profile": "intraday_aggressive"},
        "realized_pnl_total": 14.0,
        "unrealized_pnl_total": 8.0,
    }
    day3 = {
        "timestamp_utc": "2026-03-20T20:00:00+00:00",
        "symbol": "IWM",
        "action": "BUY",
        "strategy": "paper_mirror::brain_refinery_v10_seasonal",
        "metadata": {"source_profile": "swing_aggressive"},
        "realized_pnl_total": 21.0,
        "unrealized_pnl_total": 6.0,
    }
    rows = "\n".join(json.dumps(row) for row in (day1, day2, day3)) + "\n"
    (log_dir / "paper_bridge_orders_20260320.jsonl").write_text(rows, encoding="utf-8")

    def _fake_render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
        pdf_path.write_bytes(b"%PDF-1.4\n%dummy paper report\n")
        return True, "ok"

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "_render_pdf_from_html", _fake_render_pdf_from_html)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_performance_report.py",
            "--day",
            "20260320",
            "--week-days",
            "2",
            "--out-file",
            str(out_file),
            "--md-out-file",
            str(md_file),
            "--html-out-file",
            str(html_file),
            "--pdf-out-file",
            str(pdf_file),
            "--weekly-chart-file",
            str(weekly_chart),
            "--monthly-chart-file",
            str(monthly_chart),
            "--quarterly-chart-file",
            str(quarterly_chart),
            "--sleeves-chart-file",
            str(sleeves_chart),
            "--allow-gui-pdf-renderer",
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    markdown = md_file.read_text(encoding="utf-8")

    assert rc == 0
    assert payload["day"]["ending_net_pnl_total"] == 27.0
    assert payload["day"]["change_vs_previous_day"] == 5.0
    assert payload["week"]["week_to_date_change"] == 27.0
    assert payload["week"]["rolling_change"] == 12.0
    assert payload["week"]["executions"] == 3
    assert payload["weekly_history_series"][0]["week_key"] == "20260316"
    assert payload["weekly_history_series"][0]["change_vs_previous_period"] == 27.0
    assert payload["monthly_history_series"][0]["month_key"] == "202603"
    assert payload["quarterly_history_series"][0]["quarter_key"] == "2026Q1"
    assert payload["graphs"]["weekly_png"] == str(weekly_chart)
    assert payload["graphs"]["monthly_png"] == str(monthly_chart)
    assert payload["graphs"]["quarterly_png"] == str(quarterly_chart)
    assert payload["graphs"]["sleeves_png"] == str(sleeves_chart)
    assert payload["pdf"]["available"] is True
    assert payload["pdf"]["pdf_path"] == str(pdf_file)
    assert any(row["profile"] == "swing_aggressive" for row in payload["sleeve_latest"])
    assert html_file.exists()
    assert pdf_file.exists()
    assert weekly_chart.exists()
    assert monthly_chart.exists()
    assert quarterly_chart.exists()
    assert sleeves_chart.exists()
    assert pdf_file.stat().st_size > 0
    assert weekly_chart.stat().st_size > 0
    assert monthly_chart.stat().st_size > 0
    assert quarterly_chart.stat().st_size > 0
    assert sleeves_chart.stat().st_size > 0
    assert "End Of Day" in markdown
    assert "Week" in markdown
    assert "Graphs" in markdown
    assert "Sleeve Progression" in markdown


def test_paper_performance_report_json_only_skips_render_bundle(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_performance_latest.json"
    md_file = project_root / "exports" / "reports" / "paper_performance_latest.md"
    html_file = project_root / "exports" / "reports" / "paper_performance_latest.html"
    pdf_file = project_root / "exports" / "reports" / "paper_performance_latest.pdf"
    weekly_chart = project_root / "exports" / "reports" / "paper_performance_weekly_latest.png"
    monthly_chart = project_root / "exports" / "reports" / "paper_performance_monthly_latest.png"
    quarterly_chart = project_root / "exports" / "reports" / "paper_performance_quarterly_latest.png"
    sleeves_chart = project_root / "exports" / "reports" / "paper_performance_sleeves_latest.png"

    row = {
        "timestamp_utc": "2026-03-31T20:00:00+00:00",
        "symbol": "NVDA",
        "action": "BUY",
        "strategy": "grand_master_bot",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": 3.0,
        "unrealized_pnl_total": 1.5,
    }
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("render bundle should be skipped in json-only mode")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "render_paper_performance_graphs", _raise_if_called)
    monkeypatch.setattr(report, "_render_pdf_from_html", _raise_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_performance_report.py",
            "--day",
            "20260331",
            "--out-file",
            str(out_file),
            "--md-out-file",
            str(md_file),
            "--html-out-file",
            str(html_file),
            "--pdf-out-file",
            str(pdf_file),
            "--weekly-chart-file",
            str(weekly_chart),
            "--monthly-chart-file",
            str(monthly_chart),
            "--quarterly-chart-file",
            str(quarterly_chart),
            "--sleeves-chart-file",
            str(sleeves_chart),
            "--json-only",
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["day"]["ending_net_pnl_total"] == 4.5
    assert payload["graphs"]["mode"] == "json_only"
    assert payload["pdf"]["available"] is False
    assert payload["pdf"]["detail"] == "skipped_json_only"
    assert not md_file.exists()
    assert not html_file.exists()
    assert not pdf_file.exists()
    assert not weekly_chart.exists()
    assert not monthly_chart.exists()
    assert not quarterly_chart.exists()
    assert not sleeves_chart.exists()


def test_sleeve_chart_profiles_keeps_all_unique_profiles() -> None:
    rows = [
        {"profile": "default"},
        {"profile": "conservative"},
        {"profile": "aggressive"},
        {"profile": "intraday_aggressive"},
        {"profile": "swing_aggressive"},
        {"profile": "dividend"},
        {"profile": "bond"},
        {"profile": "fx"},
        {"profile": "schwab_futures"},
        {"profile": "crypto_futures"},
        {"profile": "default"},
        {"profile": ""},
        {},
    ]

    assert report._sleeve_chart_profiles(rows) == [
        "default",
        "conservative",
        "aggressive",
        "intraday_aggressive",
        "swing_aggressive",
        "dividend",
        "bond",
        "fx",
        "schwab_futures",
        "crypto_futures",
    ]


def test_paper_performance_report_aggregates_multiple_strategies_within_profile(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_performance_latest.json"

    rows = [
        {
            "timestamp_utc": "2026-03-31T20:00:00+00:00",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": 1.0,
            "unrealized_pnl_total": 2.5,
        },
        {
            "timestamp_utc": "2026-03-31T20:01:00+00:00",
            "symbol": "QQQ",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": -0.5,
            "unrealized_pnl_total": 4.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)

    sleeve = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")
    assert sleeve["executions"] == 2
    assert sleeve["ending_realized_pnl_total"] == 0.5
    assert sleeve["ending_unrealized_pnl_total"] == 6.5
    assert sleeve["ending_net_pnl_total"] == 7.0


def test_paper_performance_report_includes_win_rate_by_non_flat_strategy(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp_utc": "2026-03-31T20:00:00+00:00",
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": 1.0,
            "unrealized_pnl_total": 2.5,
        },
        {
            "timestamp_utc": "2026-03-31T20:01:00+00:00",
            "symbol": "QQQ",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": -0.5,
            "unrealized_pnl_total": -4.0,
        },
        {
            "timestamp_utc": "2026-03-31T20:02:00+00:00",
            "symbol": "IWM",
            "action": "BUY",
            "strategy": "paper_mirror::gamma",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": 0.0,
            "unrealized_pnl_total": 0.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)

    sleeve = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")
    assert sleeve["strategy_count"] == 3
    assert sleeve["winning_strategy_count"] == 1
    assert sleeve["losing_strategy_count"] == 1
    assert sleeve["flat_strategy_count"] == 1
    assert sleeve["non_flat_strategy_count"] == 2
    assert sleeve["win_rate"] == 0.5
    assert sleeve["top_winning_strategies"][0]["strategy"] == "paper_mirror::alpha"
    assert sleeve["top_losing_strategies"][0]["strategy"] == "paper_mirror::beta"
