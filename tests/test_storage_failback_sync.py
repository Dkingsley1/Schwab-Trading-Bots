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
