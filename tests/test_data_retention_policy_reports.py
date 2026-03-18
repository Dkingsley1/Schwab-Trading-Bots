import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot')
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODULE_PATH = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/data_retention_policy.py')
spec = importlib.util.spec_from_file_location('data_retention_policy_reports', MODULE_PATH)
data_retention_policy = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(data_retention_policy)


def test_collect_old_stamped_files_for_crash_reports_preserves_latest_alias(tmp_path):
    old_md = tmp_path / 'crash_report_digest_20250101_010101.md'
    old_html = tmp_path / 'crash_report_digest_print_20250101_010101.html'
    latest_pdf = tmp_path / 'crash_report_digest_latest.pdf'
    old_md.write_text('x', encoding='utf-8')
    old_html.write_text('x', encoding='utf-8')
    latest_pdf.write_text('x', encoding='utf-8')

    rows, total_files, total_runs = data_retention_policy._collect_old_stamped_files(
        tmp_path,
        data_retention_policy.CRASH_REPORT_STAMP_RE,
        older_than_days=30,
        keep_latest_runs=0,
        parse_stamp_fn=data_retention_policy._parse_timeline_stamp,
    )

    assert set(rows) == {old_md, old_html}
    assert latest_pdf not in rows
    assert total_files == 3
    assert total_runs == 1


def test_collect_old_stamped_files_for_one_numbers_can_keep_latest_run(tmp_path):
    older_md = tmp_path / 'one_numbers_20250101_20250101_010101.md'
    older_csv = tmp_path / 'one_numbers_20250101_20250101_010101.csv'
    newer_md = tmp_path / 'one_numbers_20250102_20250102_010101.md'
    newer_csv = tmp_path / 'one_numbers_20250102_20250102_010101.csv'
    for path in (older_md, older_csv, newer_md, newer_csv):
        path.write_text('x', encoding='utf-8')

    rows, total_files, total_runs = data_retention_policy._collect_old_stamped_files(
        tmp_path,
        data_retention_policy.ONE_NUMBERS_STAMP_RE,
        older_than_days=30,
        keep_latest_runs=1,
        parse_stamp_fn=data_retention_policy._parse_timeline_stamp,
    )

    assert set(rows) == {older_md, older_csv}
    assert total_files == 4
    assert total_runs == 2


def test_main_reports_candidates_for_new_report_families(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    crash_dir = tmp_path / 'exports' / 'reports' / 'crash_reports'
    training_dir = tmp_path / 'exports' / 'reports' / 'training_reports'
    reports_dir = tmp_path / 'exports' / 'reports'
    one_numbers_dir = tmp_path / 'exports' / 'one_numbers'

    crash_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    one_numbers_dir.mkdir(parents=True)

    (crash_dir / 'crash_report_digest_20250101_010101.md').write_text('x', encoding='utf-8')
    (training_dir / 'training_report_20250101_010101.md').write_text('x', encoding='utf-8')
    (reports_dir / 'daily_ops_report_20250101.md').write_text('x', encoding='utf-8')
    (one_numbers_dir / 'one_numbers_20250101_20250101_010101.md').write_text('x', encoding='utf-8')

    monkeypatch.setattr(data_retention_policy.sys, 'argv', ['data_retention_policy.py', '--exports-days', '30'])

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert payload['targets']['exports_crash_reports']['candidates'] == 1
    assert payload['targets']['exports_training_reports']['candidates'] == 1
    assert payload['targets']['exports_daily_ops_reports']['candidates'] == 1
    assert payload['targets']['exports_one_numbers']['candidates'] == 1
