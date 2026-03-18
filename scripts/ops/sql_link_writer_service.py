import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
LINK_SCRIPT = PROJECT_ROOT / 'scripts' / 'link_jsonl_to_sql.py'
HOT_RETENTION_SCRIPT = PROJECT_ROOT / 'scripts' / 'sql_hot_retention.py'
QUEUE_RETENTION_SCRIPT = PROJECT_ROOT / 'scripts' / 'sql_queue_retention.py'
SQLITE_MAINTENANCE_SCRIPT = PROJECT_ROOT / 'scripts' / 'sqlite_performance_maintenance.py'
SQLITE_DB_PATH = PROJECT_ROOT / 'data' / 'jsonl_link.sqlite3'
QUEUE_DB_PATH = PROJECT_ROOT / 'data' / 'bot_channel_queue.sqlite3'


def _db_size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _wal_size_gb(path: Path) -> float:
    return _db_size_gb(Path(f'{path}-wal'))


def _parse_json_output(text: str) -> dict:
    raw = str(text or '').strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _run_link(timeout_s: int, lock_retries: int, retry_delay_s: float, *, skip_json_files: bool = False) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(LINK_SCRIPT),
        '--mode', 'sqlite',
        '--sqlite-timeout-seconds', str(max(timeout_s, 30)),
        '--sqlite-lock-retries', str(max(lock_retries, 0)),
        '--sqlite-lock-retry-delay-seconds', str(max(retry_delay_s, 0.1)),
    ]
    if skip_json_files:
        cmd.append('--skip-json-files')
    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def _run_hot_retention(
    *,
    hot_days: int,
    batch_size: int,
    max_rows: int,
    archive_db: str,
    archive_root: str,
    archive_period: str,
    archive_retention_days: int,
    cold_export_root: str,
    cold_export_format: str,
    cold_export_batch_size: int,
    cold_export_compression: str,
    vacuum: bool,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(HOT_RETENTION_SCRIPT),
        '--db', str(SQLITE_DB_PATH),
        '--archive-db', str(archive_db),
        '--hot-days', str(max(hot_days, 1)),
        '--batch-size', str(max(batch_size, 1000)),
        '--max-rows', str(max(max_rows, 0)),
        '--archive-period', str(archive_period or 'single'),
        '--archive-retention-days', str(max(archive_retention_days, 0)),
        '--json',
    ]
    if str(archive_root or '').strip():
        cmd.extend(['--archive-root', str(archive_root)])
    if str(cold_export_root or '').strip():
        cmd.extend([
            '--cold-export-root', str(cold_export_root),
            '--cold-export-format', str(cold_export_format or 'parquet'),
            '--cold-export-batch-size', str(max(cold_export_batch_size, 1000)),
            '--cold-export-compression', str(cold_export_compression or 'zstd'),
        ])
    if vacuum:
        cmd.append('--vacuum')

    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def _run_queue_retention(
    *,
    db_path: str,
    acked_days: int,
    batch_size: int,
    max_rows: int,
    cleanup_consumer_state_days: int,
    prune_orphans: bool,
    orphan_days: int,
    vacuum: bool,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(QUEUE_RETENTION_SCRIPT),
        '--db', str(db_path),
        '--acked-days', str(max(acked_days, 1)),
        '--batch-size', str(max(batch_size, 1000)),
        '--max-rows', str(max(max_rows, 0)),
        '--cleanup-consumer-state-days', str(max(cleanup_consumer_state_days, 1)),
        '--json',
    ]
    if prune_orphans:
        cmd.extend(['--prune-orphans', '--orphan-days', str(max(orphan_days, 1))])
    if vacuum:
        cmd.append('--vacuum')

    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def _run_wal_checkpoint(
    *,
    db_path: Path,
    checkpoint_threshold_gb: float,
    truncate_max_gb: float,
    checkpoint_mode: str,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(SQLITE_MAINTENANCE_SCRIPT),
        '--db', str(db_path),
        '--checkpoint-only',
        '--wal-checkpoint-threshold-gb', str(max(checkpoint_threshold_gb, 0.0)),
        '--wal-truncate-max-gb', str(max(truncate_max_gb, 0.0)),
        '--wal-checkpoint-mode', str(checkpoint_mode or 'auto'),
        '--json',
    ]
    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def main() -> int:
    parser = argparse.ArgumentParser(description='Single-writer SQLite linker service with lock arbitration.')
    parser.add_argument('--interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_INTERVAL_SECONDS', '120')))
    parser.add_argument('--sqlite-timeout-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_SQLITE_TIMEOUT', '300')))
    parser.add_argument('--sqlite-lock-retries', type=int, default=int(os.getenv('SQL_LINK_SERVICE_LOCK_RETRIES', '200')))
    parser.add_argument('--sqlite-lock-retry-delay-seconds', type=float, default=float(os.getenv('SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS', '0.5')))
    parser.add_argument('--lock-path', default=str(PROJECT_ROOT / 'governance' / 'locks' / 'jsonl_sql_writer.lock'))
    parser.add_argument('--json-file-sync-min-interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS', '1800')))
    parser.add_argument('--auto-wal-checkpoint', action='store_true', default=os.getenv('SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT', '1') == '1')
    parser.add_argument('--wal-checkpoint-threshold-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB', '2')))
    parser.add_argument('--wal-checkpoint-min-interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS', '900')))
    parser.add_argument('--wal-truncate-max-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB', '8')))
    parser.add_argument('--wal-checkpoint-mode', choices=('auto', 'passive', 'truncate', 'restart'), default=os.getenv('SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE', 'auto'))
    parser.add_argument('--auto-hot-retention', action='store_true', default=os.getenv('SQL_LINK_SERVICE_AUTO_HOT_RETENTION', '1') == '1')
    parser.add_argument('--hot-retention-max-db-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_HOT_MAX_DB_GB', '25')))
    parser.add_argument('--hot-retention-hot-days', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_DAYS', '5')))
    parser.add_argument('--hot-retention-batch-size', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_BATCH_SIZE', '120000')))
    parser.add_argument('--hot-retention-max-rows', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_MAX_ROWS', '1000000')))
    parser.add_argument('--hot-retention-min-interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS', '300')))
    parser.add_argument('--hot-retention-vacuum-threshold-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_HOT_VACUUM_THRESHOLD_GB', '400')))
    parser.add_argument('--hot-retention-archive-db', default=os.getenv('SQL_LINK_SERVICE_HOT_ARCHIVE_DB', str(PROJECT_ROOT / 'data' / 'jsonl_link_archive.sqlite3')))
    parser.add_argument('--hot-retention-archive-root', default=os.getenv('SQL_LINK_SERVICE_HOT_ARCHIVE_ROOT', str(PROJECT_ROOT / 'data' / 'jsonl_link_archives')))
    parser.add_argument('--hot-retention-archive-period', choices=('single', 'month'), default=os.getenv('SQL_LINK_SERVICE_HOT_ARCHIVE_PERIOD', 'month'))
    parser.add_argument('--hot-retention-archive-retention-days', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_ARCHIVE_RETENTION_DAYS', '365')))
    parser.add_argument('--hot-retention-cold-export-root', default=os.getenv('SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_ROOT', ''))
    parser.add_argument('--hot-retention-cold-export-format', choices=('parquet',), default=os.getenv('SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_FORMAT', 'parquet'))
    parser.add_argument('--hot-retention-cold-export-batch-size', type=int, default=int(os.getenv('SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_BATCH_SIZE', '50000')))
    parser.add_argument('--hot-retention-cold-export-compression', default=os.getenv('SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_COMPRESSION', 'zstd'))
    parser.add_argument('--auto-queue-retention', action='store_true', default=os.getenv('SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION', '1') == '1')
    parser.add_argument('--queue-retention-db', default=os.getenv('SQL_LINK_SERVICE_QUEUE_DB', str(QUEUE_DB_PATH)))
    parser.add_argument('--queue-retention-max-db-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_QUEUE_MAX_DB_GB', '10')))
    parser.add_argument('--queue-retention-acked-days', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_ACKED_DAYS', '7')))
    parser.add_argument('--queue-retention-batch-size', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_BATCH_SIZE', '80000')))
    parser.add_argument('--queue-retention-max-rows', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_MAX_ROWS', '240000')))
    parser.add_argument('--queue-retention-min-interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS', '900')))
    parser.add_argument('--queue-retention-vacuum-threshold-gb', type=float, default=float(os.getenv('SQL_LINK_SERVICE_QUEUE_VACUUM_THRESHOLD_GB', '20')))
    parser.add_argument('--queue-retention-cleanup-consumer-state-days', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_CLEANUP_CONSUMER_STATE_DAYS', '30')))
    parser.add_argument('--queue-retention-prune-orphans', action='store_true', default=os.getenv('SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS', '0') == '1')
    parser.add_argument('--queue-retention-orphan-days', type=int, default=int(os.getenv('SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS', '45')))
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fh = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        owner = fh.read().strip()
        msg = {'ok': False, 'reason': 'writer_lock_busy', 'lock_path': str(lock_path), 'owner': owner}
        print(json.dumps(msg, ensure_ascii=True) if args.json else f"sql_link_writer_service busy owner={owner or 'unknown'}")
        return 0

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()} cmd=sql_link_writer_service")
    fh.flush()

    out_path = PROJECT_ROOT / 'governance' / 'health' / 'sql_link_service_latest.json'
    last_wal_checkpoint_ts = 0.0
    last_hot_retention_ts = 0.0
    last_queue_retention_ts = 0.0
    last_json_file_sync_ts = 0.0

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        cycle_ts = time.time()
        json_file_sync_interval = max(int(args.json_file_sync_min_interval_seconds), 60)
        include_json_files = (cycle_ts - float(last_json_file_sync_ts)) >= json_file_sync_interval
        rc, out, err = _run_link(
            timeout_s=int(args.sqlite_timeout_seconds),
            lock_retries=int(args.sqlite_lock_retries),
            retry_delay_s=float(args.sqlite_lock_retry_delay_seconds),
            skip_json_files=not include_json_files,
        )
        if rc == 0 and include_json_files:
            last_json_file_sync_ts = cycle_ts

        wal_size = _wal_size_gb(SQLITE_DB_PATH)
        wal_checkpoint = {
            'enabled': bool(args.auto_wal_checkpoint),
            'wal_size_gb_before': round(wal_size, 3),
            'threshold_gb': float(args.wal_checkpoint_threshold_gb),
            'truncate_max_gb': float(args.wal_truncate_max_gb),
            'mode': str(args.wal_checkpoint_mode),
            'ran': False,
            'rc': 0,
            'stdout_tail': '',
            'stderr_tail': '',
            'details': {},
            'skipped_reason': '',
        }

        now_ts = cycle_ts
        if args.auto_wal_checkpoint and rc == 0 and wal_size <= 0.0:
            wal_checkpoint['skipped_reason'] = 'no_wal'
        elif args.auto_wal_checkpoint and rc == 0 and wal_size >= float(args.wal_checkpoint_threshold_gb):
            since_last = now_ts - float(last_wal_checkpoint_ts)
            if since_last >= max(int(args.wal_checkpoint_min_interval_seconds), 60):
                c_rc, c_out, c_err = _run_wal_checkpoint(
                    db_path=SQLITE_DB_PATH,
                    checkpoint_threshold_gb=float(args.wal_checkpoint_threshold_gb),
                    truncate_max_gb=float(args.wal_truncate_max_gb),
                    checkpoint_mode=str(args.wal_checkpoint_mode),
                )
                wal_checkpoint['ran'] = True
                wal_checkpoint['rc'] = int(c_rc)
                wal_checkpoint['stdout_tail'] = '\n'.join(c_out.splitlines()[-12:])
                wal_checkpoint['stderr_tail'] = '\n'.join(c_err.splitlines()[-12:])
                wal_checkpoint['details'] = _parse_json_output(c_out)
                last_wal_checkpoint_ts = now_ts
            else:
                wal_checkpoint['skipped_reason'] = f'min_interval_not_met:{int(since_last)}s'
        elif args.auto_wal_checkpoint and rc != 0:
            wal_checkpoint['skipped_reason'] = 'link_failed'
        elif args.auto_wal_checkpoint:
            wal_checkpoint['skipped_reason'] = 'wal_below_threshold'

        wal_checkpoint['wal_size_gb_after'] = round(_wal_size_gb(SQLITE_DB_PATH), 3)

        db_size = _db_size_gb(SQLITE_DB_PATH)
        hot_retention = {
            'enabled': bool(args.auto_hot_retention),
            'db_size_gb_before': round(db_size, 3),
            'max_db_gb': float(args.hot_retention_max_db_gb),
            'archive_db': str(args.hot_retention_archive_db),
            'archive_root': str(args.hot_retention_archive_root or ''),
            'archive_period': str(args.hot_retention_archive_period),
            'archive_retention_days': int(args.hot_retention_archive_retention_days),
            'cold_export_root': str(args.hot_retention_cold_export_root or ''),
            'cold_export_format': str(args.hot_retention_cold_export_format),
            'batch_size': int(args.hot_retention_batch_size),
            'max_rows': int(args.hot_retention_max_rows),
            'ran': False,
            'rc': 0,
            'stdout_tail': '',
            'stderr_tail': '',
            'details': {},
            'skipped_reason': '',
        }

        if args.auto_hot_retention and rc == 0 and db_size >= float(args.hot_retention_max_db_gb):
            since_last = now_ts - float(last_hot_retention_ts)
            if since_last >= max(int(args.hot_retention_min_interval_seconds), 60):
                do_vacuum = db_size >= float(args.hot_retention_vacuum_threshold_gb)
                h_rc, h_out, h_err = _run_hot_retention(
                    hot_days=int(args.hot_retention_hot_days),
                    batch_size=int(args.hot_retention_batch_size),
                    max_rows=int(args.hot_retention_max_rows),
                    archive_db=str(args.hot_retention_archive_db),
                    archive_root=str(args.hot_retention_archive_root or ''),
                    archive_period=str(args.hot_retention_archive_period),
                    archive_retention_days=int(args.hot_retention_archive_retention_days),
                    cold_export_root=str(args.hot_retention_cold_export_root or ''),
                    cold_export_format=str(args.hot_retention_cold_export_format),
                    cold_export_batch_size=int(args.hot_retention_cold_export_batch_size),
                    cold_export_compression=str(args.hot_retention_cold_export_compression),
                    vacuum=do_vacuum,
                )
                hot_retention['ran'] = True
                hot_retention['rc'] = int(h_rc)
                hot_retention['stdout_tail'] = '\n'.join(h_out.splitlines()[-12:])
                hot_retention['stderr_tail'] = '\n'.join(h_err.splitlines()[-12:])
                hot_retention['details'] = _parse_json_output(h_out)
                hot_retention['vacuum'] = bool(do_vacuum)
                last_hot_retention_ts = now_ts
            else:
                hot_retention['skipped_reason'] = f'min_interval_not_met:{int(since_last)}s'
        elif args.auto_hot_retention and rc != 0:
            hot_retention['skipped_reason'] = 'link_failed'
        elif args.auto_hot_retention:
            hot_retention['skipped_reason'] = 'db_below_threshold'

        hot_retention['db_size_gb_after'] = round(_db_size_gb(SQLITE_DB_PATH), 3)

        queue_db_path = Path(str(args.queue_retention_db))
        queue_db_size = _db_size_gb(queue_db_path)
        queue_retention = {
            'enabled': bool(args.auto_queue_retention),
            'db_path': str(queue_db_path),
            'db_size_gb_before': round(queue_db_size, 3),
            'max_db_gb': float(args.queue_retention_max_db_gb),
            'acked_days': int(args.queue_retention_acked_days),
            'batch_size': int(args.queue_retention_batch_size),
            'max_rows': int(args.queue_retention_max_rows),
            'prune_orphans': bool(args.queue_retention_prune_orphans),
            'orphan_days': int(args.queue_retention_orphan_days),
            'cleanup_consumer_state_days': int(args.queue_retention_cleanup_consumer_state_days),
            'ran': False,
            'rc': 0,
            'stdout_tail': '',
            'stderr_tail': '',
            'details': {},
            'skipped_reason': '',
        }

        if args.auto_queue_retention and rc == 0 and not queue_db_path.exists():
            queue_retention['skipped_reason'] = 'db_missing'
        elif args.auto_queue_retention and rc == 0 and queue_db_size >= float(args.queue_retention_max_db_gb):
            since_last = now_ts - float(last_queue_retention_ts)
            if since_last >= max(int(args.queue_retention_min_interval_seconds), 60):
                do_vacuum = queue_db_size >= float(args.queue_retention_vacuum_threshold_gb)
                q_rc, q_out, q_err = _run_queue_retention(
                    db_path=str(queue_db_path),
                    acked_days=int(args.queue_retention_acked_days),
                    batch_size=int(args.queue_retention_batch_size),
                    max_rows=int(args.queue_retention_max_rows),
                    cleanup_consumer_state_days=int(args.queue_retention_cleanup_consumer_state_days),
                    prune_orphans=bool(args.queue_retention_prune_orphans),
                    orphan_days=int(args.queue_retention_orphan_days),
                    vacuum=do_vacuum,
                )
                queue_retention['ran'] = True
                queue_retention['rc'] = int(q_rc)
                queue_retention['stdout_tail'] = '\n'.join(q_out.splitlines()[-12:])
                queue_retention['stderr_tail'] = '\n'.join(q_err.splitlines()[-12:])
                queue_retention['details'] = _parse_json_output(q_out)
                queue_retention['vacuum'] = bool(do_vacuum)
                last_queue_retention_ts = now_ts
            else:
                queue_retention['skipped_reason'] = f'min_interval_not_met:{int(since_last)}s'
        elif args.auto_queue_retention and rc != 0:
            queue_retention['skipped_reason'] = 'link_failed'
        elif args.auto_queue_retention:
            queue_retention['skipped_reason'] = 'db_below_threshold'

        queue_retention['db_size_gb_after'] = round(_db_size_gb(queue_db_path), 3)

        json_file_sync = {
            'enabled': True,
            'min_interval_seconds': int(json_file_sync_interval),
            'included_this_cycle': bool(include_json_files),
            'last_sync_age_seconds': 0 if last_json_file_sync_ts <= 0 else int(max(now_ts - float(last_json_file_sync_ts), 0.0)),
            'skipped_reason': '' if include_json_files else 'min_interval_not_met',
        }

        payload = {
            'timestamp_utc': ts,
            'ok': rc == 0,
            'rc': int(rc),
            'lock_path': str(lock_path),
            'stdout_tail': '\n'.join(out.splitlines()[-20:]),
            'stderr_tail': '\n'.join(err.splitlines()[-20:]),
            'sqlite_db_size_gb': round(_db_size_gb(SQLITE_DB_PATH), 3),
            'sqlite_wal_size_gb': round(_wal_size_gb(SQLITE_DB_PATH), 3),
            'queue_db_size_gb': round(_db_size_gb(queue_db_path), 3),
            'json_file_sync': json_file_sync,
            'wal_checkpoint': wal_checkpoint,
            'hot_retention': hot_retention,
            'queue_retention': queue_retention,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"sql_link_writer_service rc={rc} ok={rc == 0} ts={ts}")

        if args.once:
            break
        time.sleep(max(int(args.interval_seconds), 10))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
