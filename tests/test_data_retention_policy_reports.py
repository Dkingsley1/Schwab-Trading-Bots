import fcntl
import importlib.util
import json
import os
import sqlite3
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


def test_main_reports_candidates_for_external_csv_exports(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    external_root = tmp_path / 'external_root'
    external_csv_dir = external_root / 'exports' / 'csv'
    external_csv_dir.mkdir(parents=True)
    old_csv = external_csv_dir / 'master_control_20250101.csv'
    old_csv.write_text('x', encoding='utf-8')
    old_local_fallback = external_csv_dir / 'latest_master_control.csv.local_fallback'
    old_local_fallback.write_text('x', encoding='utf-8')

    old_epoch = 1_735_689_600  # 2025-01-01T00:00:00Z
    os.utime(old_csv, (old_epoch, old_epoch))
    os.utime(old_local_fallback, (old_epoch, old_epoch))

    monkeypatch.setenv('BOT_LOGS_EXTERNAL_PROJECT_ROOT', str(external_root))
    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        ['data_retention_policy.py', '--csv-days', '10', '--data-local-fallback-days', '1'],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert payload['targets']['exports_csv_external']['candidates'] == 2
    assert payload['targets']['exports_csv_external_local_fallback']['candidates'] == 1


def test_main_reports_candidates_for_external_live_sqlite_when_on_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    external_root = tmp_path / 'external_root'
    external_data = external_root / 'data'
    external_data.mkdir(parents=True)
    local_data = tmp_path / 'local_fallback_storage' / 'data' / 'sql_link_shards'
    local_data.mkdir(parents=True)
    (tmp_path / 'local_fallback_storage' / 'data').mkdir(parents=True, exist_ok=True)
    watchdog_health = tmp_path / 'governance' / 'health'
    watchdog_health.mkdir(parents=True)

    (watchdog_health / 'process_watchdog_latest.json').write_text(
        json.dumps({'storage_mode': 'local_fallback'}),
        encoding='utf-8',
    )

    old_epoch = 1_735_689_600

    external_main = external_data / 'jsonl_link.sqlite3'
    external_main.write_text('x', encoding='utf-8')
    os.utime(external_main, (old_epoch, old_epoch))
    local_main = tmp_path / 'local_fallback_storage' / 'data' / 'jsonl_link.sqlite3'
    local_main.write_text('x', encoding='utf-8')

    external_queue = external_data / 'bot_channel_queue.sqlite3'
    external_queue.write_text('x', encoding='utf-8')
    os.utime(external_queue, (old_epoch, old_epoch))
    local_queue = tmp_path / 'local_fallback_storage' / 'data' / 'bot_channel_queue.sqlite3'
    local_queue.write_text('x', encoding='utf-8')

    external_shard = external_data / 'sql_link_shards'
    external_shard.mkdir()
    external_shard_file = external_shard / 'jsonl_link_trading.sqlite3'
    external_shard_file.write_text('x', encoding='utf-8')
    os.utime(external_shard_file, (old_epoch, old_epoch))
    local_shard_file = local_data / 'jsonl_link_trading.sqlite3'
    local_shard_file.write_text('x', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_EXTERNAL_PROJECT_ROOT', str(external_root))
    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        ['data_retention_policy.py', '--external-live-sqlite-days', '1'],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert payload['targets']['external_live_sqlite']['candidates'] == 3
    assert payload['targets']['external_live_sqlite']['storage_mode'] == 'local_fallback'


def test_collect_external_live_sqlite_pressure_rows_when_low_space(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    external_root = tmp_path / 'external_root'
    external_data = external_root / 'data'
    external_shards = external_data / 'sql_link_shards'
    external_shards.mkdir(parents=True)

    local_data = tmp_path / 'local_fallback_storage' / 'data'
    local_shards = local_data / 'sql_link_shards'
    local_shards.mkdir(parents=True)

    watchdog_health = tmp_path / 'governance' / 'health'
    watchdog_health.mkdir(parents=True)
    (watchdog_health / 'process_watchdog_latest.json').write_text(
        json.dumps({'storage_mode': 'local_fallback'}),
        encoding='utf-8',
    )

    (external_data / 'jsonl_link.sqlite3').write_text('x', encoding='utf-8')
    (local_data / 'jsonl_link.sqlite3').write_text('x', encoding='utf-8')
    (external_data / 'bot_channel_queue.sqlite3').write_text('x', encoding='utf-8')
    (local_data / 'bot_channel_queue.sqlite3').write_text('x', encoding='utf-8')
    (external_data / 'jsonl_link.sqlite3-wal.local_fallback').write_text('x', encoding='utf-8')
    (external_shards / 'jsonl_link_trading.sqlite3').write_text('x', encoding='utf-8')
    (external_shards / 'jsonl_link_trading.sqlite3.local_fallback').write_text('x', encoding='utf-8')
    (local_shards / 'jsonl_link_trading.sqlite3').write_text('x', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_EXTERNAL_PROJECT_ROOT', str(external_root))
    monkeypatch.setenv('BOT_LOGS_EXTERNAL_MIN_FREE_BYTES', '100')
    monkeypatch.setattr(data_retention_policy, '_disk_free_bytes', lambda _path: 40)

    rows, meta = data_retention_policy._collect_external_live_sqlite_pressure_rows(
        tmp_path,
        external_root,
        require_local_fallback=True,
    )

    assert len(rows) == 5
    assert meta['external_low_space'] is True
    assert meta['pressure_shard_file_candidates'] == 1
    assert meta['pressure_local_fallback_copy_candidates'] == 1
    assert meta['pressure_candidates'] == 5
    assert meta['pressure_shard_local_fallback_candidates'] == 1


def test_main_reports_candidates_for_nested_shard_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    shard_root = tmp_path / 'data' / 'sql_link_shards'
    shard_root.mkdir(parents=True)
    old_path = shard_root / 'jsonl_link_governance.sqlite3.local_fallback.1'
    old_path.write_text('x', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(old_path, (old_epoch, old_epoch))

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        ['data_retention_policy.py', '--data-local-fallback-days', '1'],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert payload['targets']['data_sql_link_shard_local_fallback']['candidates'] == 1


def test_main_runs_archive_pruning(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    archive_root = tmp_path / 'data' / 'jsonl_link_archives'
    archive_root.mkdir(parents=True)
    archive_path = archive_root / 'jsonl_link_archive_2025_01_01.sqlite3'
    sqlite3.connect(archive_path).close()

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--archive-retention-days',
            '1',
            '--archive-cold-export-root',
            '',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert payload['archive_pruning']['enabled'] is True
    assert payload['archive_pruning']['ran'] is True
    details = payload['archive_pruning']['details']
    assert str(archive_path) in details['deleted_archive_files']
    assert not archive_path.exists()


def test_main_skips_when_retention_lock_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    lock_path = tmp_path / 'governance' / 'locks' / 'data_retention.lock'
    lock_path.parent.mkdir(parents=True)
    lock_fh = open(lock_path, 'a+', encoding='utf-8')
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_fh.write('pid=999 started=2026-03-25T00:00:00+00:00 cmd=test')
    lock_fh.flush()

    monkeypatch.setenv('DATA_RETENTION_LOCK_PATH', str(lock_path))
    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        ['data_retention_policy.py', '--apply'],
    )

    try:
        rc = data_retention_policy.main()
        payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()

    assert rc == 0
    assert payload['busy'] is True
    assert payload['skipped_reason'] == 'lock_busy'
    assert payload['lock_path'] == str(lock_path)
    assert payload['archive_pruning']['ran'] is False


def test_main_can_stage_old_files_into_stale_section(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    log_dir = tmp_path / 'logs'
    health_dir = tmp_path / 'governance' / 'health'
    log_dir.mkdir(parents=True)
    health_dir.mkdir(parents=True)

    old_log = log_dir / 'old.log'
    old_health = health_dir / 'old_health.json'
    old_log.write_text('log', encoding='utf-8')
    old_health.write_text('{}', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(old_log, (old_epoch, old_epoch))
    os.utime(old_health, (old_epoch, old_epoch))

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--skip-sqlite-vacuum',
            '--logs-days',
            '1',
            '--governance-health-days',
            '1',
            '--stale-stage',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    stale_root = tmp_path / 'data' / 'stale_stage'
    manifest_path = stale_root / 'stale_manifest.jsonl'
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    assert rc == 0
    assert old_log.exists() is False
    assert old_health.exists() is False
    assert payload['deleted_files'] == 0
    assert payload['stale_stage']['staged_files'] == 2
    assert len(manifest_rows) == 2
    assert any('logs' in row['staged_path'] for row in manifest_rows)
    assert any('governance_health' in row['staged_path'] for row in manifest_rows)


def test_main_can_purge_old_stale_stage_files(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    stale_root = tmp_path / 'data' / 'stale_stage'
    stale_root.mkdir(parents=True)
    stale_file = stale_root / 'logs' / 'project' / 'logs' / 'old.log'
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text('old', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(stale_file, (old_epoch, old_epoch))

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--skip-sqlite-vacuum',
            '--stale-purge',
            '--stale-purge-days',
            '1',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert stale_file.exists() is False
    assert payload['stale_stage']['purge']['deleted_files'] == 1
