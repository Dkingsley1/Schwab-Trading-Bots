import importlib.util
import fcntl
import json
import os
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


def test_preserve_verified_local_route_intent_repairs_missing_override(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    local_root = project_root / "local_fallback_storage"
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "1")
    monkeypatch.delenv("BOT_STORAGE_ROUTE_EXPLICIT_SWITCH", raising=False)
    for relative_path in storage_failback_sync.TRACKED_SQLITE_ROUTES:
        target = local_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        route = project_root / relative_path
        route.parent.mkdir(parents=True, exist_ok=True)
        route.symlink_to(target)

    payload = storage_failback_sync._preserve_verified_local_route_intent(project_root)

    override = project_root / "config" / ".env.storage_override"
    assert payload["physical_local_sqlite_routes"] is True
    assert payload["override_repaired"] is True
    assert "BOT_LOGS_PREFER_EXTERNAL=0" in override.read_text(encoding="utf-8")
    assert os.environ["BOT_LOGS_PREFER_EXTERNAL"] == "0"


def test_preserve_verified_local_route_intent_yields_to_explicit_switch(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    local_root = project_root / "local_fallback_storage"
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(local_root))
    monkeypatch.setenv("BOT_STORAGE_ROUTE_EXPLICIT_SWITCH", "1")
    for relative_path in storage_failback_sync.TRACKED_SQLITE_ROUTES:
        target = local_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        route = project_root / relative_path
        route.parent.mkdir(parents=True, exist_ok=True)
        route.symlink_to(target)

    payload = storage_failback_sync._preserve_verified_local_route_intent(project_root)

    assert payload["explicit_route_switch"] is True
    assert payload["override_repaired"] is False
    assert not (project_root / "config" / ".env.storage_override").exists()


def test_lock_busy_payload_preserves_last_completed_route(tmp_path):
    out = tmp_path / 'governance' / 'health' / 'storage_failback_sync_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                'timestamp_utc': '2026-07-26T12:00:00+00:00',
                'mode': 'external',
                'certified_mode': 'external',
                'split_brain_conflicts': 0,
                'route_verification': {'verification_state': 'ready'},
            }
        ),
        encoding='utf-8',
    )
    lock_path = tmp_path / 'governance' / 'locks' / 'storage_failback_sync.lock'

    payload = storage_failback_sync._lock_busy_payload(lock_path, 'pid=123 cmd=storage_failback_sync', out)

    assert payload['mode'] == 'external'
    assert payload['certified_mode'] == 'external'
    assert payload['split_brain_conflicts'] == 0
    assert 'busy' not in payload
    assert payload['last_completed_timestamp_utc'] == '2026-07-26T12:00:00+00:00'
    assert payload['refresh_deferred']['busy'] is True
    assert payload['refresh_deferred']['skipped_reason'] == 'lock_busy_preserved_last_completed_route'


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


def test_build_sqlite_skip_report_certifies_active_local_nested_links(monkeypatch, tmp_path):
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
    (local_data / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    (external_data / 'bot_channel_queue.sqlite3').write_text('queue-external', encoding='utf-8')
    (repo_data / 'jsonl_link.sqlite3').symlink_to(local_data / 'jsonl_link.sqlite3')
    (repo_data / 'snapshot_context.sqlite3').symlink_to(local_data / 'snapshot_context.sqlite3')
    (repo_data / 'bot_channel_queue.sqlite3').symlink_to(external_data / 'bot_channel_queue.sqlite3')

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(repo_data / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='local_fallback',
        active_root=local_root,
    )

    by_rel = {row['relative_path']: row for row in payload['entries']}
    assert by_rel['data/jsonl_link.sqlite3']['classification'] == 'active_local_route'
    assert by_rel['data/jsonl_link.sqlite3']['route_verification']['state'] == 'active_local_ready'
    assert by_rel['data/snapshot_context.sqlite3']['route_verification']['state'] == 'active_local_ready'
    assert by_rel['data/bot_channel_queue.sqlite3']['route_verification']['state'] == 'verified'
    assert payload['summary']['verification_state'] == 'active_local_ready'
    assert payload['route_verification']['verification_state'] == 'active_local_ready'
    assert payload['route_verification']['ready_count'] == 3
    assert payload['route_verification']['mismatches'] == []


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


def test_build_sqlite_skip_report_verifies_newer_active_external_smaller_than_standby(monkeypatch, tmp_path):
    project_root = tmp_path / 'project'
    local_root = project_root / 'local_fallback_storage'
    external_root = tmp_path / 'external'
    local_data = local_root / 'data'
    external_data = external_root / 'data'
    local_data.mkdir(parents=True, exist_ok=True)
    external_data.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / 'data').symlink_to(external_data, target_is_directory=True)

    local_jsonl = local_data / 'jsonl_link.sqlite3'
    external_jsonl = external_data / 'jsonl_link.sqlite3'
    local_jsonl.write_text('local-standby-is-larger-than-current-external-route', encoding='utf-8')
    external_jsonl.write_text('external-live', encoding='utf-8')
    (external_data / 'bot_channel_queue.sqlite3').write_text('queue-external', encoding='utf-8')
    (external_data / 'snapshot_context.sqlite3').write_text('snapshot-external', encoding='utf-8')

    os.utime(local_jsonl, (100.0, 100.0))
    os.utime(external_jsonl, (200.0, 200.0))

    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(project_root / 'data' / 'bot_channel_queue.sqlite3'))

    payload = storage_failback_sync._build_sqlite_skip_report(
        project_root,
        external_root,
        mode='external',
        active_root=external_root,
    )

    by_rel = {row['relative_path']: row for row in payload['entries']}
    assert by_rel['data/jsonl_link.sqlite3']['classification'] == 'active_external_route'
    assert by_rel['data/jsonl_link.sqlite3']['route_verification']['state'] == 'active_external_newer_than_standby'
    assert payload['summary']['verification_state'] == 'ready'
    assert payload['route_verification']['ready_count'] == 3
    assert payload['route_verification']['mismatches'] == []


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


def test_refresh_frozen_sqlite_skip_report_updates_stale_queue_metadata(monkeypatch, tmp_path):
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
    (local_data / 'snapshot_context.sqlite3').write_text('snapshot-local', encoding='utf-8')
    queue_target = external_data / 'bot_channel_queue.sqlite3'
    queue_target.write_text('small-valid-queue', encoding='utf-8')
    (repo_data / 'jsonl_link.sqlite3').symlink_to(local_data / 'jsonl_link.sqlite3')
    (repo_data / 'snapshot_context.sqlite3').symlink_to(local_data / 'snapshot_context.sqlite3')
    (repo_data / 'bot_channel_queue.sqlite3').symlink_to(queue_target)

    monkeypatch.setattr(storage_failback_sync, 'PROJECT_ROOT', project_root)
    monkeypatch.setenv('BOT_LOGS_LOCAL_FALLBACK_ROOT', str(local_root))
    monkeypatch.setenv('BOT_CHANNEL_QUEUE_DB', str(repo_data / 'bot_channel_queue.sqlite3'))
    payload = {
        'mode': 'local_fallback_split_brain',
        'active_root': str(local_root),
        'sqlite_skip_report': {
            'entries': [
                {
                    'relative_path': 'data/bot_channel_queue.sqlite3',
                    'external': {'size_bytes': 268620754944},
                }
            ]
        },
    }

    refreshed = storage_failback_sync._refresh_frozen_sqlite_skip_report(payload, external_root)

    by_rel = {row['relative_path']: row for row in refreshed['sqlite_skip_report']['entries']}
    assert by_rel['data/bot_channel_queue.sqlite3']['external']['size_bytes'] == len('small-valid-queue')
    assert by_rel['data/bot_channel_queue.sqlite3']['route_verification']['state'] == 'verified'
    assert refreshed['frozen_lightweight_refresh']['sqlite_skip_report'] is True


def test_support_freeze_bypass_reason_requires_real_routing_when_previous_not_external(monkeypatch, tmp_path):
    previous_path = tmp_path / 'storage_failback_sync_latest.json'
    previous_path.write_text(
        json.dumps({'mode': 'local_fallback_split_brain', 'split_brain_conflicts': 3}),
        encoding='utf-8',
    )

    reason = storage_failback_sync._support_freeze_bypass_reason(previous_path, tmp_path / 'external')

    assert reason == 'previous_route_not_external:local_fallback_split_brain'


def test_support_freeze_bypass_reason_honors_explicit_local_route(monkeypatch, tmp_path):
    previous_path = tmp_path / 'storage_failback_sync_latest.json'
    previous_path.write_text(
        json.dumps({'mode': 'external', 'certified_mode': 'external', 'split_brain_conflicts': 0}),
        encoding='utf-8',
    )
    monkeypatch.setenv('BOT_LOGS_PREFER_EXTERNAL', '0')

    reason = storage_failback_sync._support_freeze_bypass_reason(previous_path, tmp_path / 'external')

    assert reason == 'explicit_local_route_requested'


def test_support_freeze_bypass_reason_allows_freeze_when_external_is_healthy(monkeypatch, tmp_path):
    previous_path = tmp_path / 'storage_failback_sync_latest.json'
    previous_path.write_text(
        json.dumps({'mode': 'external', 'certified_mode': 'external', 'split_brain_conflicts': 0}),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        storage_failback_sync,
        '_probe_external_storage',
        lambda _external_root: {
            'mount_present': True,
            'external_root_exists': True,
            'external_root_writable': True,
        },
    )

    reason = storage_failback_sync._support_freeze_bypass_reason(previous_path, tmp_path / 'external')

    assert reason == ''
