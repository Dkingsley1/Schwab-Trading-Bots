import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/ops/report_pdf_bundle.py')
spec = importlib.util.spec_from_file_location('report_pdf_bundle', MODULE_PATH)
report_pdf_bundle = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(report_pdf_bundle)


def test_latest_artifact_ignores_latest_aliases_and_local_fallback(tmp_path):
    (tmp_path / 'daily_runtime_summary_latest.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'daily_runtime_summary_20260312.json.local_fallback').write_text('{}', encoding='utf-8')
    target = tmp_path / 'daily_runtime_summary_20260312.json'
    target.write_text('{}', encoding='utf-8')

    result = report_pdf_bundle._latest_artifact(str(tmp_path / 'daily_runtime_summary_*.json*'))

    assert result == target


def test_build_specs_uses_latest_timestamped_sources(tmp_path):
    reports_dir = tmp_path / 'exports' / 'reports'
    sql_reports = tmp_path / 'exports' / 'sql_reports'
    one_numbers = tmp_path / 'exports' / 'one_numbers'
    state_snapshot = tmp_path / 'exports' / 'state_snapshot_drills'
    governance_health = tmp_path / 'governance' / 'health'

    (reports_dir / 'crash_reports').mkdir(parents=True)
    (reports_dir / 'project_timeline').mkdir(parents=True)
    (reports_dir / 'training_reports').mkdir(parents=True)
    sql_reports.mkdir(parents=True)
    one_numbers.mkdir(parents=True)
    state_snapshot.mkdir(parents=True)
    governance_health.mkdir(parents=True)

    (reports_dir / 'crash_reports' / 'crash_report_digest_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'project_timeline' / 'project_timeline_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'training_reports' / 'training_report_print_latest.html').write_text('<html></html>', encoding='utf-8')
    (reports_dir / 'daily_ops_report_latest.md').write_text('# Daily Ops', encoding='utf-8')
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
    strategy_md = reports_dir / 'strategy_attribution_latest.md'
    strategy_md.write_text('# Strategy Attribution', encoding='utf-8')
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
    assert by_slug['strategy_attribution']['source_path'] == strategy_md
    assert by_slug['post_trade_analysis']['source_path'] == post_trade_md


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
