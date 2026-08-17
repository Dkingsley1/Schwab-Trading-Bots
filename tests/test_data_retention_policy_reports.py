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


def test_one_numbers_retention_groups_metrics_and_workbook_with_report_run(tmp_path):
    paths_by_stamp = {}
    for stamp in ('20250101_010101', '20250102_010101'):
        paths = {
            tmp_path / f'one_numbers_20250102_{stamp}.md',
            tmp_path / f'one_numbers_20250102_{stamp}.csv',
            tmp_path / f'one_numbers_20250102_{stamp}_metrics.csv',
            tmp_path / f'one_numbers_20250102_{stamp}.xlsx',
        }
        for path in paths:
            path.write_text('x', encoding='utf-8')
        paths_by_stamp[stamp] = paths

    rows, total_files, total_runs = data_retention_policy._collect_old_stamped_files(
        tmp_path,
        data_retention_policy.ONE_NUMBERS_STAMP_RE,
        older_than_days=30,
        keep_latest_runs=1,
        parse_stamp_fn=data_retention_policy._parse_timeline_stamp,
    )

    assert set(rows) == paths_by_stamp['20250101_010101']
    assert not set(rows).intersection(paths_by_stamp['20250102_010101'])
    assert total_files == 8
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
    (one_numbers_dir / 'one_numbers_20250102_20250102_010101.md').write_text('x', encoding='utf-8')

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
        json.dumps({'storage_mode': 'external'}),
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
    assert meta['storage_mode'] == 'external'
    assert meta['allow_external_mode_pressure_prune'] is True
    assert meta['pressure_shard_file_candidates'] == 1
    assert meta['pressure_local_fallback_copy_candidates'] == 1
    assert meta['pressure_candidates'] == 5
    assert meta['pressure_shard_local_fallback_candidates'] == 1
    assert meta['pressure_unmirrored_shard_candidates_skipped'] == 0


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
    assert payload['stale_stage']['candidate_by_temperature']
    assert payload['stale_stage']['staged_by_storage_tier']
    assert len(manifest_rows) == 2
    assert any('logs' in row['staged_path'] for row in manifest_rows)
    assert any('governance_health' in row['staged_path'] for row in manifest_rows)
    assert all(row['temperature_label'] in {'warm', 'cool', 'cold'} for row in manifest_rows)
    assert all(row['storage_tier'] in {'warm_stage', 'cool_stage', 'cold_stage'} for row in manifest_rows)
    assert all(row['age_bucket'] for row in manifest_rows)
    assert all(row['stale_reason'] for row in manifest_rows)
    assert all(row['sha256'] for row in manifest_rows)
    assert all(row['integrity_verified'] is True for row in manifest_rows)
    assert all(row['manifest_backed'] is True for row in manifest_rows)


def test_main_can_purge_old_stale_stage_files(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    stale_root = tmp_path / 'data' / 'stale_stage'
    stale_root.mkdir(parents=True)
    stale_file = stale_root / 'logs' / 'project' / 'logs' / 'old.log'
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text('old', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(stale_file, (old_epoch, old_epoch))
    manifest_path = stale_root / 'stale_manifest.jsonl'
    manifest_path.write_text(
        json.dumps(
            {
                'event': 'staged',
                'staged_path': str(stale_file),
                'sha256': data_retention_policy._path_sha256(stale_file),
                'integrity_verified': True,
                'economic_value': 'low',
                'protected_evidence': False,
            }
        ) + '\n',
        encoding='utf-8',
    )

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


def test_main_tiered_stale_purge_preserves_high_value_and_budget_limits_low_value(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    stale_root = tmp_path / 'data' / 'stale_stage'
    low_a = stale_root / 'logs' / 'project' / 'logs' / 'old-a.log'
    low_b = stale_root / 'exports_csv' / 'project' / 'exports' / 'csv' / 'old-b.csv'
    high = stale_root / 'decision_explanations' / 'project' / 'decision_explanations' / 'old-high.jsonl'
    for path in (low_a, low_b, high):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding='utf-8')

    old_epoch = 1_735_689_600
    for path in (low_a, low_b, high):
        os.utime(path, (old_epoch, old_epoch))
    manifest_path = stale_root / 'stale_manifest.jsonl'
    manifest_path.write_text(
        ''.join(
            json.dumps(
                {
                    'event': 'staged',
                    'staged_path': str(path),
                    'sha256': data_retention_policy._path_sha256(path),
                    'integrity_verified': True,
                    'economic_value': 'high' if path == high else 'low',
                    'protected_evidence': path == high,
                }
            ) + '\n'
            for path in (low_a, low_b, high)
        ),
        encoding='utf-8',
    )

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--skip-sqlite-vacuum',
            '--stale-purge',
            '--stale-purge-days',
            '30',
            '--stale-purge-low-value-days',
            '1',
            '--stale-purge-high-value-days',
            '99999',
            '--stale-purge-max-files',
            '1',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))
    purge = payload['stale_stage']['purge']

    assert rc == 0
    assert purge['deleted_files'] == 1
    assert purge['candidate_files_raw'] == 2
    assert purge['skipped_by_budget_files'] == 1
    assert purge['skipped_by_tier_files'] == 0
    assert purge['skipped_protected_evidence_files'] == 1
    assert purge['budget_limited'] is True
    assert high.exists() is True
    assert sum(1 for path in (low_a, low_b) if path.exists()) == 1


def test_main_stage_only_protects_decision_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    log_dir = tmp_path / 'logs'
    decisions_dir = tmp_path / 'decisions'
    log_dir.mkdir(parents=True)
    decisions_dir.mkdir(parents=True)

    old_log = log_dir / 'old.log'
    old_decision = decisions_dir / 'old.jsonl'
    old_log.write_text('log', encoding='utf-8')
    old_decision.write_text('decision', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(old_log, (old_epoch, old_epoch))
    os.utime(old_decision, (old_epoch, old_epoch))

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--skip-sqlite-vacuum',
            '--stale-stage',
            '--stale-stage-only',
            '--stale-stage-sections',
            'all',
            '--logs-days',
            '1',
            '--decisions-days',
            '1',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))
    stale_root = tmp_path / 'data' / 'stale_stage'

    assert rc == 0
    assert old_log.exists() is False
    assert old_decision.exists() is True
    assert payload['deleted_files'] == 0
    assert payload['stale_stage']['stage_only'] is True
    assert payload['stale_stage']['staged_files'] == 1
    assert payload['stale_stage']['protected_files'] == 1
    assert (stale_root / 'logs').exists() is True
    assert (stale_root / 'decisions').exists() is False


def test_retention_protects_stale_canonical_latest_pointer(tmp_path):
    latest = tmp_path / 'governance' / 'health' / 'dashboard_latest.json'
    latest.parent.mkdir(parents=True)
    latest.write_text('{}', encoding='utf-8')

    eligible, protected = data_retention_policy._partition_retention_candidates(
        'governance_health',
        [latest],
    )

    assert eligible == []
    assert protected == [
        {
            'path': str(latest),
            'reason': 'canonical_latest_pointer_requires_refresh_or_explicit_retirement',
        }
    ]

    state = latest.with_name('collector_state.json')
    state.write_text('{}', encoding='utf-8')
    eligible, protected = data_retention_policy._partition_retention_candidates(
        'governance_health',
        [state],
    )
    assert eligible == []
    assert protected[0]['reason'] == 'canonical_latest_pointer_requires_refresh_or_explicit_retirement'


def test_legacy_stale_reindex_is_bounded_and_protects_evidence(tmp_path):
    stale_root = tmp_path / 'stale_stage'
    fallback = stale_root / 'external_live_sqlite' / 'external' / 'data' / 'book.sqlite3.local_fallback.1'
    evidence = stale_root / 'decisions' / 'project' / 'decisions' / 'trade_decisions_20250101.jsonl'
    for path in (fallback, evidence):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding='utf-8')
    manifest = stale_root / 'stale_manifest.jsonl'

    result = data_retention_policy._reindex_legacy_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        max_files=2,
        max_bytes=1024 * 1024,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    by_path = {row['staged_path']: row for row in rows}

    assert result['reindexed_files'] == 2
    assert by_path[str(fallback)]['protected_evidence'] is False
    assert by_path[str(fallback)]['economic_value'] == 'low'
    assert by_path[str(evidence)]['protected_evidence'] is True
    assert by_path[str(evidence)]['economic_value'] == 'critical'
    assert all(row['sha256'] for row in rows)
    assert all(row['integrity_basis'] == 'legacy_quarantine_baseline' for row in rows)
    assert result['manifest_write_mode'] == 'single_fsync_batch'


def test_legacy_stale_reindex_commits_manifest_in_one_batch(monkeypatch, tmp_path):
    stale_root = tmp_path / 'stale_stage'
    paths = [stale_root / 'logs' / f'old-{index}.log' for index in range(3)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding='utf-8')
    manifest = stale_root / 'stale_manifest.jsonl'
    calls = []
    real_append = data_retention_policy._append_jsonl_rows

    def _capture(path, rows):
        calls.append(len(rows))
        real_append(path, rows)

    monkeypatch.setattr(data_retention_policy, '_append_jsonl_rows', _capture)

    result = data_retention_policy._reindex_legacy_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        max_files=3,
        max_bytes=1024 * 1024,
    )

    assert result['reindexed_files'] == 3
    assert calls == [3]


def test_legacy_stale_reindex_paces_one_old_low_value_oversized_file(tmp_path):
    stale_root = tmp_path / 'stale_stage'
    small_paths = [
        stale_root / 'external_live_sqlite' / f'small-{index}.sqlite3.local_fallback'
        for index in range(2)
    ]
    oversized = stale_root / 'external_live_sqlite' / 'large.sqlite3.local_fallback'
    for path in small_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x')
    oversized.write_bytes(b'x' * 12)
    old_epoch = 1_735_689_600
    for path in [*small_paths, oversized]:
        os.utime(path, (old_epoch, old_epoch))
    manifest = stale_root / 'stale_manifest.jsonl'

    result = data_retention_policy._reindex_legacy_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        max_files=3,
        max_bytes=8,
        oversized_max_files=1,
        oversized_max_bytes=16,
        oversized_min_age_days=3,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]

    assert result['reindexed_files'] == 3
    assert result['standard_selected_files'] == 2
    assert result['oversized_selected_files'] == 1
    assert result['oversized_selected_bytes'] == 12
    assert result['deferred_oversized_candidate_files'] == 0
    assert {row['legacy_reindex_lane'] for row in rows} == {'standard', 'oversized_low_value'}


def test_legacy_stale_reindex_paces_one_old_medium_value_oversized_file(tmp_path):
    stale_root = tmp_path / 'stale_stage'
    oversized = stale_root / 'external_live_sqlite' / 'large.sqlite3'
    oversized.parent.mkdir(parents=True, exist_ok=True)
    oversized.write_bytes(b'x' * 12)
    os.utime(oversized, (1_735_689_600, 1_735_689_600))
    manifest = stale_root / 'stale_manifest.jsonl'

    result = data_retention_policy._reindex_legacy_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        max_files=1,
        max_bytes=8,
        oversized_max_files=1,
        oversized_max_bytes=16,
        oversized_min_age_days=3,
    )
    row = json.loads(manifest.read_text(encoding='utf-8').strip())

    assert result['reindexed_files'] == 1
    assert result['oversized_selected_files'] == 1
    assert row['economic_value'] == 'medium'
    assert row['legacy_reindex_lane'] == 'oversized_medium_value'


def test_stale_purge_skips_hashing_until_retention_window_expires(monkeypatch, tmp_path):
    stale_root = tmp_path / 'stale_stage'
    staged = stale_root / 'external_live_sqlite' / 'recent.sqlite3'
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b'recent')
    manifest = stale_root / 'stale_manifest.jsonl'
    manifest.write_text(
        json.dumps(
            {
                'event': 'staged',
                'staged_path': str(staged),
                'sha256': 'expected-but-not-read-yet',
                'integrity_verified': True,
                'economic_value': 'medium',
                'protected_evidence': False,
            }
        ) + '\n',
        encoding='utf-8',
    )
    hash_calls = []
    monkeypatch.setattr(data_retention_policy, '_path_sha256', lambda path: hash_calls.append(path) or '')

    result = data_retention_policy._purge_old_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        older_than_days=30,
        medium_value_days=14,
        max_files=1,
        max_bytes=8,
        oversized_max_files=1,
        oversized_max_bytes=16,
    )

    assert hash_calls == []
    assert result['skipped_by_tier_files'] == 1
    assert staged.exists() is True


def test_stale_purge_allows_one_verified_expired_oversized_medium_file(tmp_path):
    stale_root = tmp_path / 'stale_stage'
    staged = stale_root / 'external_live_sqlite' / 'expired.sqlite3'
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b'x' * 12)
    os.utime(staged, (1_735_689_600, 1_735_689_600))
    manifest = stale_root / 'stale_manifest.jsonl'
    manifest.write_text(
        json.dumps(
            {
                'event': 'staged',
                'staged_path': str(staged),
                'sha256': data_retention_policy._path_sha256(staged),
                'integrity_verified': True,
                'economic_value': 'medium',
                'protected_evidence': False,
            }
        ) + '\n',
        encoding='utf-8',
    )

    result = data_retention_policy._purge_old_stale_stage(
        stale_root=stale_root,
        manifest_path=manifest,
        older_than_days=30,
        medium_value_days=14,
        max_files=1,
        max_bytes=8,
        oversized_max_files=1,
        oversized_max_bytes=16,
    )

    assert result['deleted_files'] == 1
    assert result['oversized_selected_files'] == 1
    assert result['oversized_selected_bytes'] == 12
    assert staged.exists() is False


def test_main_stage_only_leaves_unmatched_candidates_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr(data_retention_policy, 'PROJECT_ROOT', tmp_path)

    log_dir = tmp_path / 'logs'
    decisions_dir = tmp_path / 'decisions'
    log_dir.mkdir(parents=True)
    decisions_dir.mkdir(parents=True)

    old_log = log_dir / 'old.log'
    old_decision = decisions_dir / 'old.jsonl'
    old_log.write_text('log', encoding='utf-8')
    old_decision.write_text('decision', encoding='utf-8')

    old_epoch = 1_735_689_600
    os.utime(old_log, (old_epoch, old_epoch))
    os.utime(old_decision, (old_epoch, old_epoch))

    monkeypatch.setattr(
        data_retention_policy.sys,
        'argv',
        [
            'data_retention_policy.py',
            '--apply',
            '--skip-sqlite-vacuum',
            '--stale-stage',
            '--stale-stage-only',
            '--stale-stage-sections',
            'logs',
            '--logs-days',
            '1',
            '--decisions-days',
            '1',
        ],
    )

    rc = data_retention_policy.main()
    payload = json.loads((tmp_path / 'governance' / 'health' / 'data_retention_latest.json').read_text(encoding='utf-8'))

    assert rc == 0
    assert old_log.exists() is False
    assert old_decision.exists() is True
    assert payload['deleted_files'] == 0
    assert payload['stale_stage']['staged_files'] == 1
