import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/report_pdf_bundle.py')
spec = importlib.util.spec_from_file_location('report_pdf_bundle', MODULE_PATH)
report_pdf_bundle = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(report_pdf_bundle)


def test_pdf_bundle_renderer_uses_app_bundle_for_headless_policy(monkeypatch, tmp_path):
    browser_bin = tmp_path / 'Google Chrome.app' / 'Contents' / 'MacOS' / 'Google Chrome'
    browser_bin.parent.mkdir(parents=True, exist_ok=True)
    browser_bin.write_text('', encoding='utf-8')

    monkeypatch.setattr(report_pdf_bundle, 'APP_BROWSER_CANDIDATES', (browser_bin,))
    monkeypatch.setattr(report_pdf_bundle.shutil, 'which', lambda _name: None)

    renderer, kind = report_pdf_bundle._pdf_renderer_binary(allow_gui_renderer=False)

    assert renderer == str(browser_bin)
    assert kind == 'browser_app'


def test_latest_artifact_ignores_latest_aliases_and_local_fallback(tmp_path):
    (tmp_path / 'daily_runtime_summary_latest.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'daily_runtime_summary_20260312.json.local_fallback').write_text('{}', encoding='utf-8')
    target = tmp_path / 'daily_runtime_summary_20260312.json'
    target.write_text('{}', encoding='utf-8')

    result = report_pdf_bundle._latest_artifact(str(tmp_path / 'daily_runtime_summary_*.json*'))

    assert result == target


def test_build_specs_uses_latest_timestamped_sources(tmp_path):
    reports_dir = tmp_path / 'exports' / 'reports'
    system_explainers = reports_dir / 'system_explainers'
    showcase_dir = reports_dir / 'showcase'
    sql_reports = tmp_path / 'exports' / 'sql_reports'
    one_numbers = tmp_path / 'exports' / 'one_numbers'
    state_snapshot = tmp_path / 'exports' / 'state_snapshot_drills'
    system_summary = reports_dir / 'system_summary'
    governance_health = tmp_path / 'governance' / 'health'
    showcase_generated = tmp_path / 'docs' / 'showcase' / 'generated'

    (reports_dir / 'crash_reports').mkdir(parents=True)
    (reports_dir / 'project_timeline').mkdir(parents=True)
    (reports_dir / 'training_reports').mkdir(parents=True)
    showcase_dir.mkdir(parents=True)
    sql_reports.mkdir(parents=True)
    one_numbers.mkdir(parents=True)
    state_snapshot.mkdir(parents=True)
    system_summary.mkdir(parents=True)
    governance_health.mkdir(parents=True)
    showcase_generated.mkdir(parents=True)

    (reports_dir / 'crash_reports' / 'crash_report_digest_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'project_timeline' / 'project_timeline_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'training_reports' / 'training_report_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'incident_report_latest.html').write_text('<html></html>', encoding='utf-8')
    system_explainers.mkdir(parents=True)
    (reports_dir / 'daily_ops_report_latest.md').write_text('# Daily Ops', encoding='utf-8')
    (system_summary / 'system_summary_latest.html').write_text('<html><body>System Summary</body></html>', encoding='utf-8')
    (system_explainers / 'framework_map_v2_latest.html').write_text('<html><body>Framework</body></html>', encoding='utf-8')
    (showcase_generated / 'special_features_latest.html').write_text('<html><body>Special Features</body></html>', encoding='utf-8')
    (system_explainers / 'runtime_hierarchy_latest.md').write_text('# Runtime Hierarchy', encoding='utf-8')
    (system_explainers / 'data_intake_and_shards_latest.md').write_text('# Data Intake', encoding='utf-8')
    (system_explainers / 'health_gates_and_halt_logic_latest.md').write_text('# Health Gates', encoding='utf-8')
    (system_explainers / 'storage_routing_and_failover_latest.md').write_text('# Storage Routing', encoding='utf-8')
    (system_explainers / 'broker_truth_and_reconciliation_latest.md').write_text('# Broker Truth', encoding='utf-8')
    (system_explainers / 'training_and_promotion_latest.md').write_text('# Training', encoding='utf-8')
    retrain = sql_reports / 'retrain_scorecard_20260312_155123.md'
    retrain.write_text('# Retrain', encoding='utf-8')
    (sql_reports / 'unified_lane_scorecard_latest.md').write_text('# Lane', encoding='utf-8')
    runtime = sql_reports / 'daily_runtime_summary_20260312.json'
    runtime.write_text('{}', encoding='utf-8')
    replay = sql_reports / 'replay_feature_ablation_20260312_155120.json'
    replay.write_text('{}', encoding='utf-8')
    one_numbers_md = one_numbers / 'one_numbers_20260312_20260312_145014.md'
    one_numbers_md.write_text('# One Numbers', encoding='utf-8')
    (state_snapshot / 'latest.json').write_text('{}', encoding='utf-8')
    (governance_health / 'daily_auto_verify_latest.json').write_text('{}', encoding='utf-8')
    (governance_health / 'model_card_latest.json').write_text('{}', encoding='utf-8')
    (governance_health / 'bot_explainability_latest.json').write_text('{}', encoding='utf-8')
    (governance_health / 'paper_execution_calibration_latest.json').write_text('{}', encoding='utf-8')
    (reports_dir / 'sentiment_report_latest.html').write_text('<html></html>', encoding='utf-8')
    strategy_md = reports_dir / 'strategy_attribution_latest.md'
    strategy_md.write_text('# Strategy Attribution', encoding='utf-8')
    strategy_inventory_dir = reports_dir / 'strategy_inventory'
    strategy_inventory_dir.mkdir(parents=True)
    strategy_inventory_md = strategy_inventory_dir / 'strategy_inventory_latest.md'
    strategy_inventory_md.write_text('# Strategy Inventory', encoding='utf-8')
    post_trade_md = reports_dir / 'post_trade_analysis_latest.md'
    post_trade_md.write_text('# Post Trade Analysis', encoding='utf-8')

    specs = report_pdf_bundle._build_specs(tmp_path)
    by_slug = {row['slug']: row for row in specs}

    assert by_slug['retrain_scorecard']['source_path'] == retrain
    assert by_slug['daily_runtime_summary']['source_path'] == runtime
    assert by_slug['replay_feature_ablation']['source_path'] == replay
    assert by_slug['one_numbers']['source_path'] == one_numbers_md
    assert by_slug['model_card']['source_path'] == governance_health / 'model_card_latest.json'
    assert by_slug['bot_explainability']['source_path'] == governance_health / 'bot_explainability_latest.json'
    assert by_slug['paper_execution_calibration']['source_path'] == governance_health / 'paper_execution_calibration_latest.json'
    assert by_slug['system_summary']['source_path'] == system_summary / 'system_summary_latest.html'
    assert by_slug['system_summary']['pdf_path'] == system_summary / 'system_summary_latest.pdf'
    assert by_slug['incident_report']['source_path'] == reports_dir / 'incident_report_latest.html'
    assert by_slug['incident_report']['pdf_path'] == reports_dir / 'incident_report_latest.pdf'
    assert by_slug['incident_review_packet']['source_path'] == governance_health / 'incident_review_packet_latest.json'
    assert by_slug['incident_review_packet']['pdf_path'] == reports_dir / 'incident_review_packet_latest.pdf'
    assert by_slug['sentiment_report']['source_path'] == reports_dir / 'sentiment_report_latest.html'
    assert by_slug['strategy_attribution']['source_path'] == strategy_md
    assert by_slug['strategy_inventory']['source_path'] == strategy_inventory_md
    assert by_slug['strategy_inventory']['pdf_path'] == strategy_inventory_dir / 'strategy_inventory_latest.pdf'
    assert by_slug['post_trade_analysis']['source_path'] == post_trade_md
    assert by_slug['special_features']['source_path'] == showcase_generated / 'special_features_latest.html'
    assert by_slug['special_features']['pdf_path'] == showcase_dir / 'special_features_latest.pdf'
    assert by_slug['framework_map_v2']['source_path'] == system_explainers / 'framework_map_v2_latest.html'
    assert by_slug['runtime_hierarchy']['source_path'] == system_explainers / 'runtime_hierarchy_latest.md'
    assert by_slug['training_and_promotion']['source_path'] == system_explainers / 'training_and_promotion_latest.md'


def test_render_entry_html_formats_markdown_and_json(tmp_path):
    md_path = tmp_path / 'daily_ops_report_latest.md'
    md_path.write_text('# Daily Ops\n\n- promote_ok: false\n', encoding='utf-8')
    json_path = tmp_path / 'model_card_latest.json'
    json_path.write_text(json.dumps({'candidate_score': 1.23, 'promoted': False}, ensure_ascii=True), encoding='utf-8')

    md_html = report_pdf_bundle._render_entry_html(
        {'title': 'Daily Ops Report', 'kind': 'markdown', 'source_path': md_path},
        generated_utc='2026-03-12T16:30:00+00:00',
    )
    json_html = report_pdf_bundle._render_entry_html(
        {'title': 'Model Card', 'kind': 'json', 'source_path': json_path},
        generated_utc='2026-03-12T16:30:00+00:00',
    )

    assert '<h1>Daily Ops</h1>' in md_html
    assert '<li>promote_ok: false</li>' in md_html
    assert 'candidate_score' in json_html
    assert '<pre class="content">' in json_html


def test_extract_markdown_section_returns_requested_section_only():
    text = (
        "# Commands\n\n"
        "## Alpha\n\n"
        "alpha text\n\n"
        "## Reports And PDFs\n\n"
        "### Report catalog bundle\n\n"
        "```bash\n"
        "cd /repo\n"
        "./scripts/ops/opsctl.sh report-pdfs --json\n"
        "```\n\n"
        "## Omega\n\n"
        "omega text\n"
    )

    result = report_pdf_bundle._extract_markdown_section(text, "Reports And PDFs")

    assert result.startswith("## Reports And PDFs")
    assert "report-pdfs --json" in result
    assert "## Alpha" not in result
    assert "## Omega" not in result


def test_bundle_status_distinguishes_missing_sources_from_renderer_failure():
    assert report_pdf_bundle._bundle_status(index_ok=False, missing_count=0, error_count=0) == "blocked"
    assert report_pdf_bundle._bundle_status(index_ok=True, missing_count=1, error_count=0) == "degraded"
    assert report_pdf_bundle._bundle_status(index_ok=True, missing_count=0, error_count=1) == "degraded"
    assert report_pdf_bundle._bundle_status(index_ok=True, missing_count=0, error_count=0) == "ready"


def test_write_simple_pdf_creates_valid_pdf(tmp_path):
    out_path = tmp_path / "fallback.pdf"

    report_pdf_bundle._write_simple_pdf(
        out_path,
        title="Fallback",
        lines=["Renderer timed out", "Source: /tmp/source.html"],
    )

    payload = out_path.read_bytes()
    assert payload.startswith(b"%PDF-1.4")
    assert b"xref" in payload
    assert b"Fallback" in payload


def test_render_index_html_embeds_reports_and_pdfs_commands(tmp_path, monkeypatch):
    commands_path = tmp_path / "COMMANDS.md"
    commands_path.write_text(
        "# Commands\n\n"
        "## Reports And PDFs\n\n"
        "### Report catalog bundle\n\n"
        "```bash\n"
        "cd /Users/dankingsley/PycharmProjects/schwab_trading_bot\n"
        "./scripts/ops/opsctl.sh report-pdfs --json\n"
        "```\n\n"
        "### Open the report catalog PDF\n\n"
        "```bash\n"
        "cd /Users/dankingsley/PycharmProjects/schwab_trading_bot\n"
        "open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf\n"
        "```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report_pdf_bundle, "COMMANDS_PATH", commands_path)

    html = report_pdf_bundle._render_index_html(
        [
            {
                "title": "Report Bundle",
                "kind": "html",
                "status": "ok",
                "pdf_path": "/tmp/report.pdf",
                "source_path": "/tmp/report.html",
            }
        ],
        generated_utc="2026-04-20T16:30:00+00:00",
    )

    assert "Paste-Ready Terminal Commands" in html
    assert "./scripts/ops/opsctl.sh report-pdfs --json" in html
    assert "open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf" in html
