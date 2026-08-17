import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import sentiment_report as report


def test_sentiment_report_renderer_timeout_returns_clean_failure() -> None:
    rc, _out, err = report._run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_sec=0.1,
        process_group=True,
    )

    assert rc == 124
    assert "timeout_after" in err


def test_sentiment_report_builds_daily_weekly_monthly_yearly_bundle(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    events_dir = project_root / "governance" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    latest_macro = project_root / "data" / "external_context" / "live_macro_latest.json"
    latest_macro.parent.mkdir(parents=True, exist_ok=True)
    latest_macro.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-09T15:30:00+00:00",
                "headline": "Latest macro snapshot",
                "summary": "Cooling inflation with steady labor backdrop",
                "speaker": "Powell",
                "source": "fed",
                "sentiment_hint": 0.35,
                "shock_hint": 0.42,
                "stance": "dovish",
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "timestamp_utc": "2025-12-27T14:00:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Year-end risk reset",
                "source": "bloomberg",
                "speaker": "Analyst",
                "sentiment_hint": -0.4,
                "shock_hint": 0.55,
                "stance_confidence": 0.7,
            },
        },
        {
            "timestamp_utc": "2026-02-27T14:00:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Risk-off labor wobble",
                "source": "bloomberg",
                "speaker": "Analyst",
                "sentiment_hint": -0.6,
                "shock_hint": 0.7,
                "stance_confidence": 0.8,
            },
        },
        {
            "timestamp_utc": "2026-03-05T14:00:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Growth stabilizes",
                "source": "cnbc",
                "speaker": "Economist",
                "sentiment_hint": 0.25,
                "shock_hint": 0.45,
                "stance_confidence": 0.7,
            },
        },
        {
            "timestamp_utc": "2026-03-28T15:00:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Earnings breadth improves",
                "source": "fed",
                "speaker": "Chair",
                "stance": "dovish",
                "shock_hint": 0.55,
                "stance_confidence": 0.9,
            },
        },
        {
            "timestamp_utc": "2026-04-08T14:00:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Soft landing odds climb",
                "source": "fed",
                "speaker": "Powell",
                "sentiment_hint": 0.5,
                "shock_hint": 0.35,
                "stance_confidence": 0.85,
            },
        },
        {
            "timestamp_utc": "2026-04-09T13:30:00+00:00",
            "event_type": "macro_media",
            "payload": {
                "headline": "Inflation cools again",
                "source": "fed",
                "speaker": "Powell",
                "sentiment_hint": 0.8,
                "shock_hint": 0.5,
                "stance_confidence": 0.9,
            },
        },
    ]
    (events_dir / "live_macro_events_20260409.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    out_file = project_root / "governance" / "health" / "sentiment_report_latest.json"
    md_file = project_root / "exports" / "reports" / "sentiment_report_latest.md"
    html_file = project_root / "exports" / "reports" / "sentiment_report_latest.html"
    pdf_file = project_root / "exports" / "reports" / "sentiment_report_latest.pdf"
    daily_chart = project_root / "exports" / "reports" / "sentiment_report_daily_latest.png"
    weekly_chart = project_root / "exports" / "reports" / "sentiment_report_weekly_latest.png"
    monthly_chart = project_root / "exports" / "reports" / "sentiment_report_monthly_latest.png"
    yearly_chart = project_root / "exports" / "reports" / "sentiment_report_yearly_latest.png"

    def _fake_render_pdf_from_html(_html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
        assert allow_gui_renderer is True
        pdf_path.write_bytes(b"%PDF-1.4\n%dummy sentiment report\n")
        return True, "ok"

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "_render_pdf_from_html", _fake_render_pdf_from_html)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_report.py",
            "--day",
            "20260409",
            "--lookback-days",
            "500",
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
            "--yearly-chart-file",
            str(yearly_chart),
            "--allow-gui-pdf-renderer",
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    markdown = md_file.read_text(encoding="utf-8")
    html = html_file.read_text(encoding="utf-8")

    assert rc == 0
    assert payload["ok"] is True
    assert payload["schema_version"] == 2
    assert payload["event_count"] == 6
    assert payload["day"]["day_utc"] == "20260409"
    assert payload["week"]["week_key"] == "20260406"
    assert payload["month"]["month_key"] == "202604"
    assert payload["year"]["year_key"] == "2026"
    assert len(payload["daily_sentiment_series"]) >= 5
    assert len(payload["weekly_sentiment_series"]) >= 4
    assert len(payload["monthly_sentiment_series"]) >= 4
    assert len(payload["yearly_sentiment_series"]) == 2
    assert payload["source_breakdown"]["event_log_points"] == 6
    assert payload["source_breakdown"]["media_summary_points"] == 0
    assert payload["graphs"]["daily_png"] == str(daily_chart)
    assert payload["graphs"]["weekly_png"] == str(weekly_chart)
    assert payload["graphs"]["monthly_png"] == str(monthly_chart)
    assert payload["graphs"]["yearly_png"] == str(yearly_chart)
    assert payload["pdf"]["available"] is True
    assert payload["latest_live_macro_snapshot"]["headline"] == "Latest macro snapshot"
    assert payload["recent_events"][0]["headline"] == "Inflation cools again"
    assert daily_chart.exists()
    assert weekly_chart.exists()
    assert monthly_chart.exists()
    assert yearly_chart.exists()
    assert pdf_file.exists()
    assert daily_chart.stat().st_size > 0
    assert weekly_chart.stat().st_size > 0
    assert monthly_chart.stat().st_size > 0
    assert yearly_chart.stat().st_size > 0
    assert pdf_file.stat().st_size > 0
    assert "Sentiment Report" in markdown
    assert "## How It Works" in markdown
    assert "The report reads event history" in markdown
    assert "## Year" in markdown
    assert "yearly_png" in markdown
    assert "How Stance Is Generated" in html
    assert "Daily, weekly, monthly, and yearly lines are all kept separately" in html
    assert "Daily Sentiment Trend" in html
    assert "Yearly Sentiment Trend" in html
    assert "Recent Sentiment Events" in html


def test_sentiment_report_json_only_skips_render_bundle(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    events_dir = project_root / "governance" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    (events_dir / "live_macro_events_20260409.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-09T13:30:00+00:00",
                "event_type": "macro_media",
                "payload": {
                    "headline": "Single event",
                    "source": "fed",
                    "speaker": "Powell",
                    "sentiment_hint": 0.4,
                    "shock_hint": 0.5,
                    "stance_confidence": 0.9,
                },
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_file = project_root / "governance" / "health" / "sentiment_report_latest.json"
    md_file = project_root / "exports" / "reports" / "sentiment_report_latest.md"
    html_file = project_root / "exports" / "reports" / "sentiment_report_latest.html"
    pdf_file = project_root / "exports" / "reports" / "sentiment_report_latest.pdf"
    daily_chart = project_root / "exports" / "reports" / "sentiment_report_daily_latest.png"
    weekly_chart = project_root / "exports" / "reports" / "sentiment_report_weekly_latest.png"
    monthly_chart = project_root / "exports" / "reports" / "sentiment_report_monthly_latest.png"
    yearly_chart = project_root / "exports" / "reports" / "sentiment_report_yearly_latest.png"

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("render bundle should be skipped in json-only mode")

    monkeypatch.setattr(report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(report, "render_sentiment_graphs", _raise_if_called)
    monkeypatch.setattr(report, "_render_pdf_from_html", _raise_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_report.py",
            "--day",
            "20260409",
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
            "--yearly-chart-file",
            str(yearly_chart),
            "--json-only",
        ],
    )

    rc = report.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["ok"] is True
    assert payload["graphs"]["mode"] == "json_only"
    assert payload["graphs"]["daily_png"] == ""
    assert payload["graphs"]["yearly_png"] == ""
    assert payload["pdf"]["available"] is False
    assert payload["pdf"]["detail"] == "skipped_json_only"
    assert not md_file.exists()
    assert not html_file.exists()
    assert not pdf_file.exists()
    assert not daily_chart.exists()
    assert not weekly_chart.exists()
    assert not monthly_chart.exists()
    assert not yearly_chart.exists()


def test_sentiment_report_backfills_from_media_summaries_when_event_logs_missing(tmp_path) -> None:
    project_root = tmp_path / "project"
    latest_macro = project_root / "data" / "external_context" / "live_macro_latest.json"
    latest_macro.parent.mkdir(parents=True, exist_ok=True)
    latest_macro.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-09T15:30:00+00:00",
                "headline": "Live snapshot",
                "summary": "Still volatile",
                "speaker": "White House",
                "source": "White House",
                "sentiment_hint": -0.4,
                "shock_hint": 0.8,
                "stance": "neutral",
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    summary_rows = {
        "alpha123": {
            "timestamp_utc": "2025-07-18T14:00:00+00:00",
            "video_id": "alpha123",
            "title": "Inflation relief takes hold",
            "speaker": "Fed official",
            "source": "Fed",
            "market_sentiment_hint": 0.42,
            "market_shock_hint": 0.35,
            "source_priority_norm": 0.92,
            "official_source_norm": 1.0,
            "transcript_quality_norm": 0.8,
            "market_high_conviction": True,
            "event_resolution_join": {"join_key": "live_macro:alpha123"},
        },
        "beta456": {
            "timestamp_utc": "2026-03-02T14:00:00+00:00",
            "video_id": "beta456",
            "title": "Tariff escalation dents risk appetite",
            "speaker": "White House",
            "source": "White House",
            "market_sentiment_hint": -0.88,
            "market_shock_hint": 0.95,
            "source_priority_norm": 0.97,
            "official_source_norm": 1.0,
            "transcript_quality_norm": 0.58,
            "market_confirmation": {"confirmed": True},
            "event_resolution_join": {"join_key": "live_macro:beta456"},
        },
    }
    media_root = project_root / "data" / "external_context" / "live_macro_media"
    for video_id, payload in summary_rows.items():
        summary_path = media_root / video_id / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    payload = report.build_sentiment_report(project_root, day="20260409", lookback_days=500)

    assert payload["ok"] is True
    assert payload["event_count"] == 2
    assert payload["source_breakdown"]["event_log_points"] == 0
    assert payload["source_breakdown"]["media_summary_points"] == 2
    assert payload["source_breakdown"]["snapshot_fallback_used"] is False
    assert payload["recent_events"][0]["headline"] == "Tariff escalation dents risk appetite"
    assert payload["recent_events"][0]["event_type"] == "live_macro_media_summary"
    assert payload["year"]["year_key"] == "2026"
    assert len(payload["yearly_sentiment_series"]) == 2
    assert payload["latest_event"]["source_kind"] == "media_summary"
