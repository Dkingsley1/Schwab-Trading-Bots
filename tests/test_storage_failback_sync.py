import importlib.util
import fcntl
import json
import sys
from pathlib import Path


ROOT = Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot')
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


storage_failback_sync = _load_module(
    'storage_failback_sync_test',
    ROOT / 'scripts' / 'ops' / 'storage_failback_sync.py',
)
data_retention_policy = _load_module(
    'data_retention_policy_for_storage_sync_test',
    ROOT / 'scripts' / 'data_retention_policy.py',
)


def test_maybe_autoprune_external_low_space(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_failback_sync, 'PROJECT_ROOT', tmp_path)

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
    monkeypatch.setenv('BOT_LOGS_EXTERNAL_MOUNT', str(tmp_path))
    monkeypatch.setenv('BOT_LOGS_EXTERNAL_MIN_FREE_BYTES', '100')
    monkeypatch.setenv('BOT_LOGS_LOW_SPACE_AUTOPRUNE_ENABLED', '1')
    monkeypatch.setenv('RETENTION_EXTERNAL_LIVE_SQLITE_REQUIRE_LOCAL_FALLBACK', '1')
    monkeypatch.setattr(storage_failback_sync, '_disk_free_bytes', lambda _path: 40)
    monkeypatch.setattr(data_retention_policy, '_disk_free_bytes', lambda _path: 40)
    monkeypatch.setitem(sys.modules, 'data_retention_policy', data_retention_policy)

    payload = storage_failback_sync._maybe_autoprune_external_low_space(tmp_path, external_root)

    assert payload['attempted'] is True
    assert payload['candidate_count'] == 5
    assert payload['deleted_count'] == 5
    assert payload['error_count'] == 0
    assert payload['details']['allow_external_mode_pressure_prune'] is True
    assert not (external_data / 'jsonl_link.sqlite3').exists()
    assert not (external_data / 'bot_channel_queue.sqlite3').exists()
    assert not (external_data / 'jsonl_link.sqlite3-wal.local_fallback').exists()
    assert not (external_shards / 'jsonl_link_trading.sqlite3').exists()
    assert not (external_shards / 'jsonl_link_trading.sqlite3.local_fallback').exists()


def test_acquire_singleton_lock_reports_busy_owner(tmp_path):
    lock_path = tmp_path / 'governance' / 'locks' / 'storage_failback_sync.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open('a+', encoding='utf-8')
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fh.seek(0)
    fh.truncate(0)
    fh.write('pid=123 started=test')
    fh.flush()

    busy_handle, owner = storage_failback_sync._acquire_singleton_lock(lock_path)

    assert busy_handle is None
    assert owner == 'pid=123 started=test'

    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


def test_build_sqlite_skip_report_classifies_active_queue_and_warm_standby(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    (local_root / 'data').mkdir(parents=True, exist_ok=True)
    (external_root / 'data').mkdir(parents=True, exist_ok=True)

    (local_root / 'data' / 'jsonl_link.sqlite3').write_text('local-primary', encoding='utf-8')
    (external_root / 'data' / 'jsonl_link.sqlite3').write_text('external-primary', encoding='utf-8')
    (local_root / 'data' / 'bot_channel_queue.sqlite3').write_text('queue-local', encoding='utf-8')
    (external_root / 'data' / 'bot_channel_queue.sqlite3').write_text('queue-external', encoding='utf-8')
    (local_root / 'data' / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    (external_root / 'data' / 'snapshot_context.sqlite3').write_text('snapshot-external', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(local_root / 'data' / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    assert payload['queue_db_path'] == str(local_root / 'data' / 'bot_channel_queue.sqlite3')
    by_rel = {row['relative_path']: row for row in payload['entries']}
    assert by_rel['data/bot_channel_queue.sqlite3']['classification'] == 'active_local_queue'
    assert by_rel['data/jsonl_link.sqlite3']['classification'] == 'warm_standby_retained'
    assert by_rel['data/snapshot_context.sqlite3']['classification'] == 'warm_standby_retained'
    assert by_rel['data/jsonl_link.sqlite3']['route_verification']['state'] == 'verified'
    assert payload['summary']['active_local_count'] == 1
    assert payload['summary']['warm_standby_count'] == 2
    assert payload['summary']['verification_state'] == 'ready'
    assert payload['route_verification']['ready_count'] == 3


def test_build_sqlite_skip_report_certifies_curated_external_mode(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    (local_root / 'data').mkdir(parents=True, exist_ok=True)
    (external_root / 'data').mkdir(parents=True, exist_ok=True)

    (local_root / 'data' / 'jsonl_link.sqlite3').write_text('local-primary', encoding='utf-8')
    (local_root / 'data' / 'bot_channel_queue.sqlite3').write_text('queue-local', encoding='utf-8')
    (local_root / 'data' / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(local_root / 'data' / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    assert payload['certified_mode'] == 'external_curated'
    assert payload['summary']['verification_state'] == 'curated_ready'
    assert payload['summary']['curated_standby_count'] == 3
    assert payload['route_verification']['certified_mode'] == 'external_curated'
    assert payload['route_verification']['ready_count'] == 3


def test_build_sqlite_skip_report_treats_smaller_standby_copy_as_curated(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    (local_root / 'data').mkdir(parents=True, exist_ok=True)
    (external_root / 'data').mkdir(parents=True, exist_ok=True)

    (local_root / 'data' / 'jsonl_link.sqlite3').write_text('local-primary-is-bigger-than-external', encoding='utf-8')
    (external_root / 'data' / 'jsonl_link.sqlite3').write_text('small', encoding='utf-8')
    (local_root / 'data' / 'bot_channel_queue.sqlite3').write_text('queue-local', encoding='utf-8')
    (external_root / 'data' / 'bot_channel_queue.sqlite3').write_text('queue-local-verified', encoding='utf-8')
    (local_root / 'data' / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    (external_root / 'data' / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(local_root / 'data' / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    by_rel = {row['relative_path']: row for row in payload['entries']}
    assert by_rel['data/jsonl_link.sqlite3']['route_verification']['state'] == 'curated_standby'
    assert payload['summary']['verification_state'] == 'curated_ready'
    assert payload['certified_mode'] == 'external_curated'


def test_build_sqlite_skip_report_treats_routed_queue_db_as_external(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    local_data = local_root / 'data'
    external_data = external_root / 'data'
    local_data.mkdir(parents=True, exist_ok=True)
    external_data.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / 'data').symlink_to(external_data, target_is_directory=True)

    (local_data / 'jsonl_link.sqlite3').write_text('local-primary', encoding='utf-8')
    (external_data / 'jsonl_link.sqlite3').write_text('external-primary', encoding='utf-8')
    (local_data / 'bot_channel_queue.sqlite3').write_text('queue-local', encoding='utf-8')
    (external_data / 'bot_channel_queue.sqlite3').write_text('queue-external', encoding='utf-8')
    (local_data / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    (external_data / 'snapshot_context.sqlite3').write_text('snapshot-external', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_LOGS_PREFER_EXTERNAL', '1')
    monkeypatch.delenv('BOT_CHANNEL_QUEUE_DB', raising=False)
    monkeypatch.delenv('BOT_CHANNEL_QUEUE_PREFER_LOCAL', raising=False)

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    by_rel = {row['relative_path']: row for row in payload['entries']}
    assert payload['queue_db_path'] == str(project_root / 'data' / 'bot_channel_queue.sqlite3')
    assert by_rel['data/bot_channel_queue.sqlite3']['route_verification']['state'] == 'verified'
    assert by_rel['data/bot_channel_queue.sqlite3']['classification'] != 'active_local_queue'


def test_build_sqlite_skip_report_verifies_repo_passthrough_queue_db(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    repo_data = project_root / 'data'
    local_data = local_root / 'data'
    external_data = external_root / 'data'
    repo_data.mkdir(parents=True, exist_ok=True)
    local_data.mkdir(parents=True, exist_ok=True)
    external_data.mkdir(parents=True, exist_ok=True)

    (local_data / 'jsonl_link.sqlite3').write_text('local-primary', encoding='utf-8')
    (external_data / 'jsonl_link.sqlite3').write_text('external-primary', encoding='utf-8')
    (repo_data / 'bot_channel_queue.sqlite3').write_text('active-queue', encoding='utf-8')
    (local_data / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    (external_data / 'snapshot_context.sqlite3').write_text('snapshot-external', encoding='utf-8')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(repo_data / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    by_rel = {row['relative_path']: row for row in payload['entries']}
    queue = by_rel['data/bot_channel_queue.sqlite3']
    assert queue['classification'] == 'active_repo_queue_passthrough'
    assert queue['route_verification']['state'] == 'active_passthrough'
    assert queue['active_repo']['exists'] is True
    assert payload['summary']['active_passthrough_count'] == 1
    assert payload['summary']['verification_mismatch_count'] == 0
    assert payload['route_verification']['ready_count'] == 3
