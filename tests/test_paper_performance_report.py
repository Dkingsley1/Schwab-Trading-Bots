import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
    daily_chart = project_root / "exports" / "reports" / "paper_performance_daily_latest.png"
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
            "--daily-chart-file",
            str(daily_chart),
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
    assert payload["graphs"]["daily_png"] == str(daily_chart)
    assert payload["graphs"]["weekly_png"] == str(weekly_chart)
    assert payload["graphs"]["monthly_png"] == str(monthly_chart)
    assert payload["graphs"]["quarterly_png"] == str(quarterly_chart)
    assert payload["graphs"]["sleeves_png"] == str(sleeves_chart)
    assert payload["pdf"]["available"] is True
    assert payload["pdf"]["pdf_path"] == str(pdf_file)
    assert any(row["profile"] == "swing_aggressive" for row in payload["sleeve_latest"])
    assert html_file.exists()
    assert pdf_file.exists()
    assert daily_chart.exists()
    assert weekly_chart.exists()
    assert monthly_chart.exists()
    assert quarterly_chart.exists()
    assert sleeves_chart.exists()
    assert pdf_file.stat().st_size > 0
    assert daily_chart.stat().st_size > 0
    assert weekly_chart.stat().st_size > 0
    assert monthly_chart.stat().st_size > 0
    assert quarterly_chart.stat().st_size > 0
    assert sleeves_chart.stat().st_size > 0
    assert "End Of Day" in markdown
    assert "Week" in markdown
    assert "Graphs" in markdown
    assert "daily_png" in markdown
    assert "Sleeve Scoreboard" in markdown


def test_paper_performance_report_json_only_skips_render_bundle(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = project_root / "governance" / "health" / "paper_performance_latest.json"
    md_file = project_root / "exports" / "reports" / "paper_performance_latest.md"
    html_file = project_root / "exports" / "reports" / "paper_performance_latest.html"
    pdf_file = project_root / "exports" / "reports" / "paper_performance_latest.pdf"
    daily_chart = project_root / "exports" / "reports" / "paper_performance_daily_latest.png"
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

    sync_calls = []

    def _fake_sync(project_root_arg, performance_path, *, enabled):
        sync_calls.append((project_root_arg, performance_path, enabled))
        return {"ok": True, "attempted": True, "reason": "hash_bound"}

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "render_paper_performance_graphs", _raise_if_called)
    monkeypatch.setattr(report, "_render_pdf_from_html", _raise_if_called)
    monkeypatch.setattr(report, "_sync_profitability_control", _fake_sync)
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
            "--daily-chart-file",
            str(daily_chart),
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
    assert payload["graphs"]["daily_png"] == ""
    assert payload["pdf"]["available"] is False
    assert payload["pdf"]["detail"] == "skipped_json_only"
    assert not md_file.exists()
    assert not html_file.exists()
    assert not pdf_file.exists()
    assert not daily_chart.exists()
    assert not weekly_chart.exists()
    assert not monthly_chart.exists()
    assert not quarterly_chart.exists()
    assert not sleeves_chart.exists()
    assert sync_calls == [(project_root, out_file, True)]


def test_profitability_generation_sync_requires_matching_hash(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    performance_path = project_root / "governance" / "health" / "paper_performance_latest.json"
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.write_text(json.dumps({"timestamp_utc": "2026-08-05T20:00:00+00:00"}), encoding="utf-8")
    control_script = project_root / "scripts" / "ops" / "paper_profitability_control.py"
    control_script.parent.mkdir(parents=True, exist_ok=True)
    control_script.write_text("# test placeholder\n", encoding="utf-8")
    expected_hash = report._file_sha256(performance_path)

    def _fake_run(cmd, **kwargs):
        assert cmd[-1] == "--apply"
        assert kwargs["env"][report.PAPER_PROFITABILITY_LOCK_ENV] == "1"
        control_path = project_root / "governance" / "health" / "paper_profitability_control_latest.json"
        control_path.write_text(
            json.dumps(
                {
                    "paper_performance_input_contract": {
                        "sha256": expected_hash,
                        "usable_for_profitability_grade": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(report.subprocess, "run", _fake_run)

    result = report._sync_profitability_control(project_root, performance_path, enabled=True)
    sync_payload = json.loads(
        (project_root / "governance" / "health" / "paper_profitability_generation_sync_latest.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["ok"] is True
    assert result["reason"] == "hash_bound"
    assert result["paper_performance_sha256"] == expected_hash
    assert result["profitability_source_sha256"] == expected_hash
    assert sync_payload["generation_id"] == expected_hash[:16]


def test_sleeve_chart_profiles_keeps_all_unique_profiles() -> None:
    rows = [
        {"profile": "default"},
        {"profile": "conservative"},
        {"profile": "aggressive"},
        {"profile": "intraday_aggressive"},
        {"profile": "swing_aggressive"},
        {"profile": "dividend"},
        {"profile": "dividend_capture"},
        {"profile": "dividend_compound"},
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
        "dividend_capture",
        "dividend_compound",
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


def test_schema_v2_counts_each_profile_book_once_and_keeps_strategy_books(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "metadata": {"source_profile": "intraday_aggressive"},
        "paper_pnl_schema_version": 2,
        "paper_profile": "intraday_aggressive",
        "action": "BUY",
        "execution_notional": 1000.0,
        "expected_execution_cost_amount": 1.0,
    }
    rows = [
        {
            **base,
            "timestamp_utc": "2026-03-31T20:00:00+00:00",
            "symbol": "SPY",
            "strategy": "paper_mirror::alpha",
            "paper_book_id": "book-a",
            "paper_profile_realized_pnl_total": 10.0,
            "paper_profile_unrealized_pnl_total": 2.0,
            "paper_strategy_net_pnl_total": 3.0,
            "post_cost_pnl_delta": 1.0,
            "post_cost_return_bps": 10.0,
        },
        {
            **base,
            "timestamp_utc": "2026-03-31T20:01:00+00:00",
            "symbol": "QQQ",
            "strategy": "paper_mirror::beta",
            "paper_book_id": "book-a",
            "paper_profile_realized_pnl_total": 12.0,
            "paper_profile_unrealized_pnl_total": 3.0,
            "paper_strategy_net_pnl_total": -1.0,
            "post_cost_pnl_delta": -0.5,
            "post_cost_return_bps": -5.0,
        },
        {
            **base,
            "timestamp_utc": "2026-03-31T20:02:00+00:00",
            "symbol": "IWM",
            "strategy": "paper_mirror::alpha",
            "paper_book_id": "book-b",
            "paper_profile_realized_pnl_total": 2.0,
            "paper_profile_unrealized_pnl_total": 1.0,
            "paper_strategy_net_pnl_total": 2.0,
            "post_cost_pnl_delta": 2.0,
            "post_cost_return_bps": 20.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)

    sleeve = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")
    assert sleeve["ending_realized_pnl_total"] == 14.0
    assert sleeve["ending_unrealized_pnl_total"] == 4.0
    assert sleeve["ending_net_pnl_total"] == 18.0
    assert sleeve["accounting_scope"] == "persistent_profile_book"
    assert sleeve["paper_book_count"] == 2
    assert sleeve["strategy_count"] == 3
    assert sleeve["top_winning_strategies"][0]["strategy"] == "paper_mirror::alpha"
    assert sleeve["top_losing_strategies"][0]["strategy"] == "paper_mirror::beta"
    assert payload["post_cost_expectancy"]["sample_count"] == 3
    assert payload["post_cost_expectancy"]["mean_post_cost_pnl_delta"] == 0.833333
    assert payload["post_cost_expectancy"]["status"] == "insufficient_evidence"


def test_post_cost_expectancy_requires_positive_confidence_bound() -> None:
    rows = [
        {
            "timestamp_utc": f"2026-03-31T20:{idx:02d}:00+00:00",
            "paper_pnl_schema_version": 2,
            "post_cost_pnl_delta": 1.0,
            "post_cost_return_bps": 10.0,
            "execution_notional": 1000.0,
            "expected_execution_cost_amount": 1.0,
        }
        for idx in range(30)
    ]

    expectancy = report._post_cost_expectancy(rows)

    assert expectancy["sample_count"] == 30
    assert expectancy["evidence_sufficient"] is True
    assert expectancy["positive_lower_confidence_bound_95"] is True
    assert expectancy["status"] == "positive_with_95pct_confidence"


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


def test_paper_performance_report_uses_sleeve_specific_risk_metrics(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day, aggressive_net, conservative_net in (
        ("20260329", 1.0, 0.5),
        ("20260330", -1.0, 1.0),
        ("20260331", 3.0, 0.75),
    ):
        rows.extend(
            [
                {
                    "timestamp_utc": f"{day[:4]}-{day[4:6]}-{day[6:]}T20:00:00+00:00",
                    "symbol": "SPY",
                    "action": "BUY",
                    "strategy": "paper_mirror::alpha",
                    "metadata": {"source_profile": "intraday_aggressive"},
                    "realized_pnl_total": aggressive_net,
                    "unrealized_pnl_total": 0.0,
                },
                {
                    "timestamp_utc": f"{day[:4]}-{day[4:6]}-{day[6:]}T20:01:00+00:00",
                    "symbol": "TLT",
                    "action": "BUY",
                    "strategy": "paper_mirror::bond",
                    "metadata": {"source_profile": "conservative"},
                    "realized_pnl_total": conservative_net,
                    "unrealized_pnl_total": 0.0,
                },
            ]
        )
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)

    aggressive = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")
    conservative = next(row for row in payload["sleeve_latest"] if row["profile"] == "conservative")
    assert aggressive["risk_adjusted_metric"] == "sortino_ratio"
    assert aggressive["sortino_ratio"] is not None
    assert conservative["risk_adjusted_metric"] == "sharpe_ratio"
    assert conservative["sharpe_ratio"] is not None


def test_paper_performance_report_builds_loss_causes_and_tca_summary(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp_utc": "2026-03-31T14:31:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": -2.0,
            "unrealized_pnl_total": -1.0,
            "spread_regime": "wide",
            "tradeability_score": 0.22,
            "source_quality_norm": 0.88,
            "event_proximity_norm": 0.71,
            "allocation_conflict_norm": 0.60,
            "expected_fill_quality_bucket": "poor",
            "expected_slippage_bps": 14.0,
            "realized_slippage_bps": 22.0,
            "slippage_gap_bps": 8.0,
            "expected_partial_fill_ratio": 0.5,
        },
        {
            "timestamp_utc": "2026-03-31T19:31:00+00:00",
            "symbol": "MSFT",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": 1.0,
            "unrealized_pnl_total": 0.5,
            "spread_regime": "tight",
            "tradeability_score": 0.82,
            "source_quality_norm": 0.90,
            "event_proximity_norm": 0.05,
            "allocation_conflict_norm": 0.10,
            "expected_fill_quality_bucket": "good",
            "expected_slippage_bps": 6.0,
            "realized_slippage_bps": 4.0,
            "slippage_gap_bps": -2.0,
            "expected_partial_fill_ratio": 1.0,
        },
    ]
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)

    sleeve = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")
    assert sleeve["top_loss_causes"]
    assert any(
        str(row["cause"]).startswith(
            (
                "spread_regime:",
                "tradeability:",
                "event_proximity:",
                "time_of_day:",
                "fill_quality:",
                "source_quality:",
                "conflict_control:",
            )
        )
        for row in sleeve["top_loss_causes"]
    )
    assert sleeve["tca_summary"]["mean_slippage_gap_bps"] == 3.0


def test_paper_performance_report_surfaces_advanced_feature_telemetry(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    log_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp_utc": "2026-03-31T14:31:00+00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "strategy": "paper_mirror::alpha",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": -1.0,
            "unrealized_pnl_total": 0.0,
            "core_cross_sectional_rank_norm": 0.90,
            "day_failed_breakout_risk_norm": 0.80,
            "dividend_payout_stress_gate_norm": 0.10,
        },
        {
            "timestamp_utc": "2026-03-31T19:31:00+00:00",
            "symbol": "MSFT",
            "action": "BUY",
            "strategy": "paper_mirror::beta",
            "metadata": {"source_profile": "intraday_aggressive"},
            "realized_pnl_total": 0.5,
            "unrealized_pnl_total": 1.5,
            "core_cross_sectional_rank_norm": 0.50,
            "day_failed_breakout_risk_norm": 0.20,
            "long_term_factor_exposure_control_norm": 0.75,
        },
    ]
    (log_dir / "paper_bridge_orders_20260331.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260331", week_days=7)
    sleeve = next(row for row in payload["sleeve_latest"] if row["profile"] == "intraday_aggressive")

    telemetry = sleeve["advanced_feature_telemetry"]
    assert telemetry["core_cross_sectional_rank_norm"]["mean_norm"] == 0.7
    assert telemetry["day_failed_breakout_risk_norm"]["high_count"] == 1
    assert telemetry["long_term_factor_exposure_control_norm"]["bucket"] == "high"
    assert "cross_sectional_rank" in sleeve["advanced_feature_summary"]


def test_paper_performance_report_ingests_mixed_and_gz_sources(tmp_path, monkeypatch) -> None:
    import gzip

    project_root = tmp_path / "project"
    bridge_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    trade_logs_dir = project_root / "exports" / "trade_logs" / "session_a"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    trade_logs_dir.mkdir(parents=True, exist_ok=True)

    bridge_row = {
        "timestamp_utc": "2026-04-01T15:00:00+00:00",
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::alpha",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": 1.0,
        "unrealized_pnl_total": 0.5,
    }
    trade_log_row = {
        "timestamp_utc": "2026-04-01T16:00:00+00:00",
        "symbol": "QQQ",
        "action": "SELL",
        "strategy": "paper_mirror::beta",
        "metadata": {"source_profile": "aggressive"},
        "realized_pnl_total": 2.0,
        "unrealized_pnl_total": 1.0,
    }
    root_row = {
        "timestamp_utc": "2026-04-01T17:00:00+00:00",
        "symbol": "IWM",
        "action": "BUY",
        "strategy": "paper_mirror::gamma",
        "metadata": {"source_profile": "dividend"},
        "realized_pnl_total": 0.5,
        "unrealized_pnl_total": 0.25,
    }

    (bridge_dir / "paper_bridge_orders_20260401.jsonl").write_text(json.dumps(bridge_row) + "\n", encoding="utf-8")
    with gzip.open(trade_logs_dir / "paper_trades_20260401.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(trade_log_row) + "\n")
    with gzip.open(project_root / "paper_trades_20260401.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(root_row) + "\n")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260401", week_days=7)

    assert payload["source_kind"] == "paper_broker_bridge,trade_logs,root_paper_trades"
    assert payload["source_files_scanned"] == 3
    profiles = {row["profile"] for row in payload["sleeve_latest"] if row["data_status"] != "no_data"}
    assert profiles == {"default", "aggressive", "dividend"}


def test_paper_performance_report_counts_bridge_mirror_once(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    bridge_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
    trade_dir = project_root / "exports" / "trade_logs" / "paper"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::alpha",
        "decision_id": "decision-123",
        "paper_book_id": "book-1",
        "paper_pnl_schema_version": 2,
        "metadata": {"source_profile": "default", "decision_id": "decision-123"},
        "post_cost_pnl_delta": 2.5,
        "post_cost_return_bps": 25.0,
        "execution_notional": 1_000.0,
        "expected_execution_cost_amount": 0.5,
        "realized_pnl_total": 2.5,
        "unrealized_pnl_total": 0.0,
    }
    canonical = {
        **common,
        "timestamp_utc": "2026-04-01T15:00:00.000000+00:00",
        "message_id": "canonical-message",
        "routing_lane": "schwab_equities",
    }
    bridge = {
        **common,
        "timestamp_utc": "2026-04-01T15:00:00.001000+00:00",
        "message_id": "bridge-message",
        "bridge_source": "local_paper_mirror",
        "routing_lane": "paper_broker_bridge",
    }
    (trade_dir / "paper_trades_paper.jsonl").write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    (bridge_dir / "paper_bridge_orders_20260401.jsonl").write_text(json.dumps(bridge) + "\n", encoding="utf-8")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260401", week_days=7)

    assert payload["post_cost_expectancy"]["sample_count"] == 1
    assert payload["post_cost_expectancy"]["total_post_cost_pnl_delta"] == 2.5
    assert payload["day"]["executions"] == 1
    assert payload["execution_deduplication"]["mirrored_records_suppressed"] == 1
    assert payload["execution_deduplication"]["records_emitted"] == 1


def test_paper_performance_report_excludes_independent_and_replay_fill_calibration(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    paper_dir = project_root / "exports" / "trade_logs" / "paper"
    independent_dir = project_root / "exports" / "trade_logs" / "independent_fills"
    replay_dir = project_root / "exports" / "trade_logs" / "session_replay"
    paper_dir.mkdir(parents=True, exist_ok=True)
    independent_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    canonical = {
        "timestamp_utc": "2026-04-01T15:00:00+00:00",
        "decision_id": "paper-execution",
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::alpha",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": 1.0,
        "unrealized_pnl_total": 0.0,
    }
    calibration = {
        "timestamp_utc": "2026-04-01T15:01:00+00:00",
        "external_fill_id": "calibration-fill",
        "symbol": "SPY",
        "action": "BUY",
        "paper_fill_source": "market_replay_fill",
        "metadata": {
            "source_profile": "default",
            "account_mode": "replay",
            "independent_fill_evidence": True,
        },
    }
    (paper_dir / "paper_trades_paper.jsonl").write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    (independent_dir / "paper_trades_20260401.jsonl").write_text(
        json.dumps(calibration) + "\n",
        encoding="utf-8",
    )
    (replay_dir / "paper_trades_20260401.jsonl").write_text(
        json.dumps({**calibration, "external_fill_id": "misrouted-calibration"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260401", week_days=7)

    assert payload["day"]["executions"] == 1
    assert payload["day"]["top_strategies"][0]["name"] == "paper_mirror::alpha"
    assert payload["source_files_scanned"] == 2
    assert payload["execution_deduplication"]["calibration_source_files_excluded"] == 1
    assert payload["execution_deduplication"]["calibration_records_excluded"] == 1
    assert payload["execution_deduplication"]["records_emitted"] == 1


def test_paper_performance_report_reads_external_reconciliation_snapshot(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    paper_dir = external_root / "exports" / "trade_logs" / "paper"
    paper_dir.mkdir(parents=True)
    row = {
        "timestamp_utc": "2026-04-01T15:00:00+00:00",
        "decision_id": "reconciled-paper-execution",
        "paper_book_id": "book-1",
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::alpha",
        "metadata": {"source_profile": "default"},
        "realized_pnl_total": 1.0,
        "unrealized_pnl_total": 0.0,
    }
    (paper_dir / "paper_trades_paper.jsonl.local_fallback.1").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    payload = report.build_paper_performance_report(project_root, day="20260401", week_days=7)

    assert payload["day"]["executions"] == 1
    assert "reconciled_trade_logs" in payload["source_kind"]
    assert payload["execution_deduplication"]["reconciliation_source_files_included"] == 1


def test_paper_performance_report_publishes_and_enforces_scan_watermark(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    trade_dir = project_root / "exports" / "trade_logs" / "paper"
    trade_dir.mkdir(parents=True, exist_ok=True)
    watermark = report.datetime(2026, 4, 1, 15, 0, tzinfo=report.timezone.utc)
    common = {
        "symbol": "SPY",
        "action": "BUY",
        "strategy": "paper_mirror::alpha",
        "paper_pnl_schema_version": 2,
        "metadata": {"source_profile": "default"},
        "post_cost_return_bps": 25.0,
        "execution_notional": 1_000.0,
        "realized_pnl_total": 2.5,
        "unrealized_pnl_total": 0.0,
    }
    rows = [
        {
            **common,
            "timestamp_utc": "2026-04-01T14:59:00+00:00",
            "decision_id": "included",
            "post_cost_pnl_delta": 2.5,
        },
        {
            **common,
            "timestamp_utc": "2026-04-01T15:01:00+00:00",
            "decision_id": "next-refresh",
            "post_cost_pnl_delta": 3.0,
        },
    ]
    (trade_dir / "paper_trades_paper.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "_utc_now", lambda: watermark)
    payload = report.build_paper_performance_report(project_root, day="20260401", week_days=7)

    assert payload["profitability_evidence_window"]["evidence_through_utc"] == watermark.isoformat()
    assert payload["profitability_evidence_window"]["snapshot_watermark_active"] is True
    assert payload["post_cost_expectancy"]["sample_count"] == 1
    assert payload["post_cost_expectancy"]["total_post_cost_pnl_delta"] == 2.5
    assert payload["execution_deduplication"]["post_snapshot_records_deferred"] == 1


def test_paper_performance_report_surfaces_active_heartbeat_only_profiles(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health_dir = project_root / "governance" / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    (health_dir / "shadow_loop_fx_equities_schwab_101.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-20T14:00:00+00:00",
                "pid": 101,
                "broker": "schwab",
                "profile": "fx",
                "domain": "equities",
                "state": "running",
                "symbols_total": 10,
                "context_total": 10,
            }
        ),
        encoding="utf-8",
    )
    (health_dir / "shadow_loop_schwab_futures_equities_schwab_102.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-20T14:05:00+00:00",
                "pid": 102,
                "broker": "schwab",
                "profile": "schwab_futures",
                "domain": "equities",
                "state": "running",
                "symbols_total": 7,
                "context_total": 3,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    payload = report.build_paper_performance_report(project_root, day="20260420", week_days=7)

    assert payload["ok"] is True
    assert payload["active_paper_profile_count_today"] == 2
    assert [row["profile"] for row in payload["active_paper_profiles_today"]] == ["fx", "schwab_futures"]
    latest = {row["profile"]: row for row in payload["sleeve_latest"]}
    assert latest["fx"]["data_status"] == "current_live_no_fills"
    assert latest["fx"]["current_day_available"] is False
    assert latest["fx"]["financial_grade_eligible"] is False
    assert latest["fx"]["activity_note"] == "live heartbeat active; no paper fills yet today"


def test_candidate_forward_accounting_requires_current_candidate_identity(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    state_path = project_root / "governance" / "runtime" / "production_candidate_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-current",
                "generation": 54,
                "overall_sha256": "receipt-54",
                "scope_windows_started_utc": {
                    "strategy": "2026-08-16T12:00:00+00:00",
                    "execution": "2026-08-16T12:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )
    log_path = project_root / "exports" / "trade_logs" / "paper" / "paper_trades_20260816.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def row(timestamp: str, candidate_id: str, pnl: float) -> dict:
        return {
            "timestamp_utc": timestamp,
            "symbol": "SPY",
            "action": "BUY",
            "strategy": "paper_portfolio_consensus::default::core",
            "paper_pnl_schema_version": 3,
            "post_cost_pnl_delta": pnl,
            "post_cost_return_bps": pnl,
            "realized_pnl_delta": pnl,
            "realized_pnl_total": pnl,
            "unrealized_pnl_total": 0.0,
            "metadata": {
                "source_profile": "default",
                "production_candidate_id": candidate_id,
            },
        }

    log_path.write_text(
        "".join(
            json.dumps(item) + "\n"
            for item in (
                row("2026-08-16T11:00:00+00:00", "candidate-current", -5.0),
                row("2026-08-16T13:00:00+00:00", "candidate-old", -2.0),
                row("2026-08-16T14:00:00+00:00", "candidate-current", 3.0),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)

    payload = report.build_paper_performance_report(project_root, day="20260816", week_days=7)
    views = payload["accounting_views"]

    assert views["lifetime_flow"]["sample_count"] == 3
    assert views["candidate_forward_flow"]["sample_count"] == 1
    assert views["candidate_forward_flow"]["candidate_ids"] == ["candidate-current"]
    assert views["candidate_forward_flow"]["candidate_binding_mismatch_rows_excluded"] == 1
    assert payload["post_cost_expectancy"]["sample_count"] == 1
