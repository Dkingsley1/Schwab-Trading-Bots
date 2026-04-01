import fcntl
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/project_timeline_report.py")
spec = importlib.util.spec_from_file_location("project_timeline_report_test", MODULE_PATH)
project_timeline_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = project_timeline_report
spec.loader.exec_module(project_timeline_report)


def test_default_allow_gui_pdf_renderer_detects_installed_browser(monkeypatch, tmp_path):
    browser = tmp_path / "Google Chrome"
    browser.write_text("", encoding="utf-8")
    monkeypatch.setattr(project_timeline_report, "APP_BROWSER_CANDIDATES", (browser,))

    assert project_timeline_report._default_allow_gui_pdf_renderer() is True


def test_pdf_renderer_binary_uses_gui_browser_candidate(monkeypatch, tmp_path):
    browser = tmp_path / "Google Chrome"
    browser.write_text("", encoding="utf-8")
    monkeypatch.setattr(project_timeline_report, "APP_BROWSER_CANDIDATES", (browser,))

    renderer, renderer_kind = project_timeline_report._pdf_renderer_binary(allow_gui_renderer=True)

    assert renderer == str(browser)
    assert renderer_kind == "browser"


def test_build_project_milestone_timeline_filters_out_runtime_heartbeat_noise():
    context = {
        "git": {
            "commits": [
                {"date": "2026-02-10T12:00:00+00:00", "sha": "aaa1111", "subject": "Create main control room for trading bot"},
                {"date": "2026-02-25T12:00:00+00:00", "sha": "bbb2222", "subject": "feat: harden guardrails, promotion gates, and retrain automation"},
                {"date": "2026-03-24T12:00:00+00:00", "sha": "ccc3333", "subject": "feat: add actual dividend drip plumbing and runtime integration"},
            ],
            "recent_working_tree_changes": [
                {
                    "path": "scripts/collect_dividend_drip_state.py",
                    "status": "M",
                    "mtime_local": "2026-03-24T12:31:00-04:00",
                    "mtime_epoch": 3.0,
                    "is_dir": False,
                },
            ],
        },
        "ops": {
            "recent_project_activity": [
                {
                    "path": "governance/health/shadow_loop_default_crypto_coinbase_123.json",
                    "mtime_local": "2026-03-24T12:30:00-04:00",
                    "mtime_epoch": 1.0,
                },
                {
                    "path": "governance/health/storage_failback_sync_latest.json",
                    "mtime_local": "2026-03-24T12:35:00-04:00",
                    "mtime_epoch": 2.0,
                },
            ],
        },
    }

    rows = project_timeline_report._build_project_milestone_timeline(context, limit=10)
    titles = [str(row.get("title", "")) for row in rows]
    refs = [str(row.get("reference", "")) for row in rows]

    assert "Bot loop heartbeat" not in titles
    assert any(str(row.get("reference", "")).startswith("aaa1111") or str(row.get("reference", "")) == "aaa1111" for row in rows)
    assert any(str(row.get("reference", "")).startswith("ccc3333") or str(row.get("reference", "")) == "ccc3333" for row in rows)
    assert "scripts/collect_dividend_drip_state.py" in refs


def test_build_current_phase_changes_keeps_high_signal_items_only():
    context = {
        "git": {
            "commits": [
                {"date": "2026-03-24T12:00:00+00:00", "sha": "ccc3333", "subject": "feat: add actual dividend drip plumbing and runtime integration"},
            ],
            "recent_working_tree_changes": [
                {
                    "path": "scripts/ops/project_timeline_report.py",
                    "status": "M",
                    "mtime_local": "2026-03-24T12:31:00-04:00",
                    "mtime_epoch": 3.0,
                    "is_dir": False,
                },
            ],
        },
        "ops": {
            "recent_project_activity_hours": 72,
            "recent_project_activity": [
                {
                    "path": "governance/health/shadow_loop_default_crypto_coinbase_123.json",
                    "mtime_local": "2026-03-24T12:30:00-04:00",
                    "mtime_epoch": 1.0,
                },
                {
                    "path": "governance/health/storage_failback_sync_latest.json",
                    "mtime_local": "2026-03-24T12:35:00-04:00",
                    "mtime_epoch": 2.0,
                },
            ],
        },
    }

    rows = project_timeline_report._build_current_phase_changes(context, limit=10)
    titles = [str(row.get("title", "")) for row in rows]

    assert "Storage failback sync" in titles
    assert "Timeline report generator" in titles
    assert "Bot loop heartbeat" not in titles


def test_build_buildout_summary_groups_milestones_into_themes():
    git_data = {
        "commits": [
            {"date": "2026-02-10T12:00:00+00:00", "sha": "aaa1111", "subject": "Create main control room for trading bot"},
            {"date": "2026-02-24T18:56:54+00:00", "sha": "bbb2222", "subject": "Add dividend and bond sleeves with watchdog + reporting integration"},
            {"date": "2026-02-25T20:17:34+00:00", "sha": "ccc3333", "subject": "Add ops automation stack and sql writer watchdogs"},
        ]
    }
    milestone_timeline = [
        {
            "title": "Milestone commit",
            "detail": "Create main control room for trading bot",
            "reference": "aaa1111",
            "area": "Git",
            "sort_epoch": 1.0,
            "date_local": "2026-02-10T07:00:00-05:00",
        },
        {
            "title": "Milestone commit",
            "detail": "Add dividend and bond sleeves with watchdog + reporting integration",
            "reference": "bbb2222",
            "area": "Git",
            "sort_epoch": 2.0,
            "date_local": "2026-02-24T13:56:54-05:00",
        },
    ]
    current_phase = [
        {
            "title": "Storage failback sync",
            "detail": "latest storage route status refreshed",
            "reference": "governance/health/storage_failback_sync_latest.json",
            "area": "Operations",
            "sort_epoch": 3.0,
            "date_local": "2026-03-24T12:35:00-04:00",
        },
        {
            "title": "Cross-sleeve correlation context",
            "detail": "market/crypto overlap and correlation snapshot refreshed",
            "reference": "governance/health/market_crypto_correlation_sync_latest.json",
            "area": "Intelligence",
            "sort_epoch": 4.0,
            "date_local": "2026-03-24T12:40:00-04:00",
        }
    ]

    summary = project_timeline_report._build_buildout_summary(milestone_timeline, current_phase, git_data)

    assert any("Overall buildout" in line for line in summary)
    assert any("Control room and orchestration" in line for line in summary)
    assert any("Sleeve expansion" in line for line in summary)
    assert any("Cross-sleeve intelligence" in line for line in summary)
    assert any("Reliability and storage" in line for line in summary)


def test_render_markdown_includes_cross_sleeve_intelligence_section():
    context = {
        "generated_utc": "2026-03-25T00:30:00+00:00",
        "generated_local": "2026-03-24T20:30:00-04:00",
        "include_detailed_timeline": False,
        "git": {
            "available": True,
            "branch": "codex/test",
            "head": "abc1234",
            "commit_count": 1,
            "commits": [
                {"date": "2026-03-24T12:00:00+00:00", "sha": "abc1234", "subject": "feat: add cross-sleeve context layer"},
            ],
            "status_lines": [],
            "status_branch_line": "## codex/test",
            "recent_working_tree_changes": [],
        },
        "ops": {
            "promotion": {},
            "graduation": {},
            "retrain": {},
            "leak": {},
            "preflight": {},
            "storage": {},
            "latest_all_sleeves_log": "all_sleeves_20260324_203000.log",
            "latest_coinbase_log": "coinbase_live_20260324_203000.log",
            "recent_project_activity": [],
            "recent_project_activity_hours": 72,
            "preflight_events": [],
            "market_crypto_correlation_sync": {
                "ok": True,
                "mode": "exact",
                "exact_aligned_pairs": 6,
                "aligned_pairs": 6,
                "rows_scanned": 173708,
                "timestamp_utc": "2026-03-25T00:07:08.369139+00:00",
            },
            "market_crypto_correlation": {
                "derived": {
                    "pair_metrics": [
                        {"left": "stock_risk_basket", "right": "crypto_basket", "corr": 0.18459604299787197, "points": 328, "mode": "exact"},
                        {"left": "SPY", "right": "BTC-USD", "corr": 0.1268258100796286, "points": 327, "mode": "exact"},
                        {"left": "QQQ", "right": "BTC-USD", "corr": 0.13111560533095637, "points": 317, "mode": "exact"},
                    ]
                }
            },
            "fx_market_context_sync": {
                "ok_source_count": 5,
                "source_count": 5,
                "official_pairs": 6,
                "proxy_symbols_observed": 13,
                "warning_count": 1,
                "timestamp_utc": "2026-03-25T00:24:13.369046+00:00",
                "sources": {
                    "twelve_data": {"pairs_ok": 6},
                },
            },
        },
    }

    markdown = project_timeline_report._render_markdown(context)

    assert "## Cross-Sleeve Intelligence" in markdown
    assert "Correlation layer:" in markdown
    assert "FX context layer:" in markdown
    assert "SPY_vs_BTC=" in markdown


def test_main_auto_mode_skips_when_lock_busy(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "exports" / "reports" / "project_timeline"
    output_dir.mkdir(parents=True)
    lock_path = tmp_path / "governance" / "locks" / "project_timeline_report.lock"
    lock_path.parent.mkdir(parents=True)

    lock_fh = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_fh.write("pid=123 started=2026-03-25T00:00:00+00:00 cmd=test")
    lock_fh.flush()

    monkeypatch.setenv("PROJECT_TIMELINE_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_timeline_report.py",
            "--auto",
            "--json",
            "--output-dir",
            str(output_dir),
            "--state-file",
            str(tmp_path / "governance" / "health" / "project_timeline_state.json"),
        ],
    )

    try:
        rc = project_timeline_report.main()
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()

    payload = capsys.readouterr().out.strip()
    assert rc == 0
    assert '"busy": true' in payload.lower()
