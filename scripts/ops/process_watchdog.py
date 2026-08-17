import argparse
import fcntl
import glob
import json
import os
import signal
import shlex
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from core.runtime_maintenance import maintenance_hold_snapshot
from core.storage_mounts import find_target_external_volume, resolve_external_storage

PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_STATE_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_state.json'
DEFAULT_OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_latest.json'
FALLBACK_STATE_PATH = Path('/tmp/process_watchdog_state.json')
FALLBACK_OUT_PATH = Path('/tmp/process_watchdog_latest.json')
DEFAULT_SINGLETON_LOCK_PATH = Path('/tmp/schwab_trading_bot/process_watchdog.lock')
HEALTH_DIR = PROJECT_ROOT / 'governance' / 'health'
GLOBAL_HALT_FLAG = HEALTH_DIR / 'GLOBAL_TRADING_HALT.flag'
OPERATOR_STOP_FLAG = HEALTH_DIR / 'OPERATOR_STOP.flag'
DEFAULT_STORAGE_MOUNT_GUARD_PATH = PROJECT_ROOT / 'governance' / 'health' / 'storage_mount_guard_latest.json'
FALLBACK_STORAGE_MOUNT_GUARD_PATH = Path('/tmp/storage_mount_guard_latest.json')
DEFAULT_CREATIVE_PAUSE_PATH = PROJECT_ROOT / 'governance' / 'health' / 'creative_heavy_research_pause_latest.json'
DEFAULT_RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH = PROJECT_ROOT / 'config' / '.env.runtime_resource_guard_override'
SNAPSHOT_SCRIPT = PROJECT_ROOT / 'scripts' / 'collect_debug_snapshot.sh'
ALERT_ROUTER = PROJECT_ROOT / 'scripts' / 'pager_alert_router.py'
DEFAULT_MAINTENANCE_TIMEOUT_SECONDS = max(
    float(os.getenv('OPS_WATCHDOG_MAINTENANCE_TIMEOUT_SECONDS', '45') or 45.0),
    1.0,
)
DEFAULT_CREATIVE_PAUSE_MAX_AGE_SECONDS = max(
    float(os.getenv('OPS_WATCHDOG_CREATIVE_PAUSE_MAX_AGE_SECONDS', '120') or 120.0),
    20.0,
)
PROCESS_WRAPPER_MATCH_EXCLUDES = (
    'scripts/shadow_watchdog.py',
    'scripts/failover_hot_standby.py',
    'scripts/ops/master_infrastructure_supervisor.py',
    'scripts/ops/process_watchdog.py',
)
READ_ONLY_COLLECTION_RESTART_STORM_TARGETS = {
    'all_sleeves',
    'coinbase_loop',
    'coinbase_futures_loop',
}
RESTART_STORM_IMPACTS = {
    'execution_lane',
    'read_only_collection',
    'storage_writer',
    'support_or_unknown',
}
SECRET_PLACEHOLDER_VALUES = {
    '',
    'YOUR_KEY_HERE',
    'YOUR_SECRET_HERE',
    'YOUR_REAL_KEY',
    'YOUR_REAL_SECRET',
    'YOUR_REAL_CLIENT_ID',
    '<real_key>',
    '<real_secret>',
    '<real_client_id>',
}


def _env_flag(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


def _hot_storage_prefers_external() -> bool:
    return str(os.getenv('BOT_LOGS_PREFER_EXTERNAL', '1') or '1').strip().lower() not in {
        '0',
        'false',
        'no',
        'off',
    }


def _acquire_singleton_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+', encoding='utf-8')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        return None, owner
    handle.seek(0)
    handle.truncate(0)
    handle.write(f'pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}')
    handle.flush()
    return handle, ''


def _placeholder_or_empty(raw: Any) -> bool:
    return str(raw or '').strip() in SECRET_PLACEHOLDER_VALUES


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    return text in {'1', 'true', 'yes', 'on'}


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or '').split(',') if x.strip()]


def _default_require_paper_executor() -> bool:
    explicit = os.getenv('OPS_WATCHDOG_REQUIRE_PAPER_EXECUTOR', '').strip()
    if explicit:
        return explicit == '1'

    require_all_sleeves = _env_flag('OPS_WATCHDOG_REQUIRE_ALL_SLEEVES', '1')
    run_all_sleeves_with_paper = _env_flag('RUN_ALL_SLEEVES_WITH_PAPER_EXECUTOR', '1')
    if require_all_sleeves and run_all_sleeves_with_paper:
        return False
    return True


def _safety_pause_state() -> Dict[str, Any]:
    operator_stop_active = OPERATOR_STOP_FLAG.exists()
    global_halt_active = GLOBAL_HALT_FLAG.exists()
    maintenance_hold = maintenance_hold_snapshot(PROJECT_ROOT)
    maintenance_hold_active = bool(maintenance_hold.get('active', False))
    pause_reason = ''
    if maintenance_hold_active:
        pause_reason = 'runtime_maintenance_hold_active'
    elif operator_stop_active:
        pause_reason = 'operator_stop_active'
    elif global_halt_active:
        pause_reason = 'global_halt_active'
    child_fanout_grace_seconds = max(
        float(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_CHILD_GRACE_SECONDS', '180') or 180.0),
        60.0,
    )

    return {
        'operator_stop_active': bool(operator_stop_active),
        'global_halt_active': bool(global_halt_active),
        'runtime_maintenance_hold_active': maintenance_hold_active,
        'runtime_maintenance_hold': maintenance_hold,
        'active': bool(maintenance_hold_active or operator_stop_active or global_halt_active),
        'reason': pause_reason,
    }


def _creative_cotenant_pause_state() -> Dict[str, Any]:
    payload = _load_json_payload(DEFAULT_CREATIVE_PAUSE_PATH)
    if not payload:
        return {'active': False, 'reason': 'creative_pause_artifact_missing'}
    active = bool(payload.get('active', False))
    level = str(payload.get('creative_session_level') or '').strip().lower()
    kind = str(payload.get('creative_session_kind') or '').strip().lower()
    hard_pause = payload.get('hard_pause') if isinstance(payload.get('hard_pause'), dict) else {}
    hard_pause_terminate_processes = (
        _safe_bool(hard_pause.get('terminate_processes'), default=True)
        if 'terminate_processes' in hard_pause
        else None
    )
    hard_pause_action = str(hard_pause.get('action') or '').strip().lower()
    timestamp_utc = str(payload.get('timestamp_utc') or '')
    ts_epoch = _iso_epoch(timestamp_utc)
    age_seconds = max(time.time() - ts_epoch, 0.0) if ts_epoch is not None else None
    stale = bool(active and (age_seconds is None or age_seconds > DEFAULT_CREATIVE_PAUSE_MAX_AGE_SECONDS))
    if stale:
        return {
            'active': False,
            'reason': 'creative_pause_artifact_stale',
            'creative_session_level': level,
            'creative_session_kind': kind,
            'timestamp_utc': timestamp_utc,
            'age_seconds': round(float(age_seconds), 3) if age_seconds is not None else None,
            'max_age_seconds': DEFAULT_CREATIVE_PAUSE_MAX_AGE_SECONDS,
            'stale_active_artifact': True,
            'raw_active': active,
            'hard_pause_terminate_processes': hard_pause_terminate_processes,
            'hard_pause_action': hard_pause_action,
        }
    return {
        'active': active,
        'reason': str(payload.get('reason') or kind or level or 'creative_cotenant_pause_active'),
        'creative_session_level': level,
        'creative_session_kind': kind,
        'timestamp_utc': timestamp_utc,
        'age_seconds': round(float(age_seconds), 3) if age_seconds is not None else None,
        'max_age_seconds': DEFAULT_CREATIVE_PAUSE_MAX_AGE_SECONDS,
        'stale_active_artifact': False,
        'hard_pause_terminate_processes': hard_pause_terminate_processes,
        'hard_pause_action': hard_pause_action,
    }


def _creative_pause_suppresses_target(target_name: str, pause: Dict[str, Any]) -> bool:
    if not bool(pause.get('active', False)):
        return False
    name = str(target_name or '').strip().lower()
    suppressible = {
        'all_sleeves',
        'coinbase_loop',
        'coinbase_futures_loop',
    }
    if name not in suppressible:
        return False
    kind = str(pause.get('creative_session_kind') or '').strip().lower()
    level = str(pause.get('creative_session_level') or '').strip().lower()
    if 'music' in kind or 'audio' in kind:
        hard_pause_terminate_processes = pause.get('hard_pause_terminate_processes')
        hard_pause_action = str(pause.get('hard_pause_action') or '').strip().lower()
        soft_audio_pause = bool(
            level == 'active'
            and (
                hard_pause_terminate_processes is False
                or hard_pause_action in {'soft_pause_optional_heavy_research', 'lightweight_pause_contract_refresh'}
            )
        )
        if soft_audio_pause:
            return False
        return True
    return level in {'active', 'hot', 'realtime', 'dual_pro'}


def _effective_process_excludes(exclude_patterns: List[str] | None = None) -> List[str]:
    excludes: List[str] = []
    for marker in [*PROCESS_WRAPPER_MATCH_EXCLUDES, *(exclude_patterns or [])]:
        if marker and marker not in excludes:
            excludes.append(marker)
    return excludes


def _live_data_excludes(simulate: bool, extra: List[str] | None = None) -> List[str]:
    excludes = [x for x in (extra or []) if x]
    if not simulate:
        excludes.append('--simulate')
    return excludes


def _proc_running(pattern: str, exclude_patterns: List[str] | None = None) -> int:
    p = subprocess.run(['ps', '-axo', 'stat=,command='], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    excludes = _effective_process_excludes(exclude_patterns)
    running = 0
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        stat, command = parts
        if stat.startswith('T'):
            continue
        if pattern in command and not any(marker in command for marker in excludes):
            running += 1
    return running


def _matching_pids(pattern: str, exclude_patterns: List[str] | None = None) -> List[int]:
    p = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    excludes = _effective_process_excludes(exclude_patterns)
    pids: List[int] = []
    for line in out.splitlines()[1:]:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_raw, command = parts
        if pattern not in command or any(marker in command for marker in excludes):
            continue
        try:
            pid = int(pid_raw)
        except Exception:
            continue
        if pid > 0 and pid != os.getpid():
            pids.append(pid)
    return pids


def _terminate_matching_processes(
    patterns: List[str],
    *,
    exclude_patterns: List[str] | None = None,
    grace_seconds: float = 3.0,
) -> Dict[str, Any]:
    pids: List[int] = []
    for pattern in patterns:
        if not pattern:
            continue
        for pid in _matching_pids(pattern, exclude_patterns=exclude_patterns):
            if pid not in pids:
                pids.append(pid)
    terminated: List[int] = []
    errors: List[Dict[str, Any]] = []
    for pid in pids:
        try:
            try:
                os.kill(pid, signal.SIGCONT)
            except Exception:
                pass
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append({'pid': int(pid), 'error': str(exc)})

    deadline = time.time() + max(float(grace_seconds), 0.0)
    while time.time() < deadline:
        remaining: List[int] = []
        for pid in terminated:
            try:
                os.kill(pid, 0)
                remaining.append(pid)
            except ProcessLookupError:
                continue
            except Exception:
                remaining.append(pid)
        if not remaining:
            break
        time.sleep(0.2)

    killed: List[int] = []
    for pid in terminated:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append({'pid': int(pid), 'error': f'sigkill_failed:{exc}'})

    if killed:
        time.sleep(0.2)

    still_running: List[int] = []
    for pid in terminated:
        try:
            os.kill(pid, 0)
            still_running.append(pid)
        except ProcessLookupError:
            continue
        except Exception:
            still_running.append(pid)

    return {
        'attempted': bool(patterns),
        'matched_pids': pids,
        'terminated_pids': terminated,
        'killed_pids': killed,
        'still_running_pids': still_running,
        'errors': errors,
    }


def _trim_duplicate_processes(
    pattern: str,
    *,
    max_running: int,
    exclude_patterns: List[str] | None = None,
    grace_seconds: float = 1.0,
) -> Dict[str, Any]:
    keep_count = max(int(max_running), 1)
    pids = sorted(_matching_pids(pattern, exclude_patterns=exclude_patterns))
    if len(pids) <= keep_count:
        return {
            'attempted': False,
            'matched_pids': pids,
            'kept_pids': pids,
            'terminated_pids': [],
            'still_running_pids': [],
            'errors': [],
            'policy': 'single_instance_duplicate_trim_keep_newest',
        }

    kept = pids[-keep_count:]
    trim = [pid for pid in pids if pid not in kept]
    terminated: List[int] = []
    errors: List[Dict[str, Any]] = []
    for pid in trim:
        try:
            try:
                os.kill(pid, signal.SIGCONT)
            except Exception:
                pass
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append({'pid': int(pid), 'error': str(exc)})

    deadline = time.time() + max(float(grace_seconds), 0.0)
    while time.time() < deadline:
        still_running: List[int] = []
        for pid in terminated:
            try:
                os.kill(pid, 0)
                still_running.append(pid)
            except ProcessLookupError:
                continue
            except Exception:
                still_running.append(pid)
        if not still_running:
            break
        time.sleep(0.2)

    killed: List[int] = []
    for pid in terminated:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except Exception as exc:
            errors.append({'pid': int(pid), 'error': f'sigkill_failed:{exc}'})

    if killed:
        time.sleep(0.2)

    still_running = []
    for pid in terminated:
        try:
            os.kill(pid, 0)
            still_running.append(pid)
        except ProcessLookupError:
            continue
        except Exception:
            still_running.append(pid)

    return {
        'attempted': True,
        'matched_pids': pids,
        'kept_pids': kept,
        'terminated_pids': terminated,
        'killed_pids': killed,
        'still_running_pids': still_running,
        'errors': errors,
        'policy': 'single_instance_duplicate_trim_keep_newest',
    }


def _parse_ps_etime_seconds(raw: str) -> float | None:
    text = str(raw or '').strip()
    if not text:
        return None
    days = 0
    if '-' in text:
        day_raw, text = text.split('-', 1)
        try:
            days = int(day_raw)
        except Exception:
            return None
    parts = text.split(':')
    try:
        values = [int(part) for part in parts]
    except Exception:
        return None
    if len(values) == 2:
        hours = 0
        minutes, seconds = values
    elif len(values) == 3:
        hours, minutes, seconds = values
    else:
        return None
    return float((((days * 24) + hours) * 60 + minutes) * 60 + seconds)


def _proc_elapsed_seconds(pattern: str, exclude_patterns: List[str] | None = None) -> float | None:
    p = subprocess.run(['ps', '-axo', 'etime=,command='], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    excludes = _effective_process_excludes(exclude_patterns)
    matched: list[float] = []
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        etime_raw, cmd = parts
        if pattern not in cmd or any(marker in cmd for marker in excludes):
            continue
        elapsed = _parse_ps_etime_seconds(etime_raw)
        if elapsed is not None:
            matched.append(float(elapsed))
    if not matched:
        return None
    return max(matched)


def _spawn(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'a', encoding='utf-8')
    env = dict(os.environ)
    env.pop('__PYVENV_LAUNCHER__', None)
    env.setdefault('BOT_RUNTIME_LANE', 'canary314')
    env.setdefault('BOT_PYTHON_VERSION', '3.14.5')
    env.setdefault('BOT_TRAINING_RUNTIME_LANE', 'canary314')
    env.setdefault('BOT_TRAINING_PYTHON_VERSION', '3.14.5')
    env.setdefault('PY314_RUNTIME_FLIP_APPROVED', '1')
    env.setdefault('PY314_RETIRE_312_ANCHOR', '1')
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS', '0')
    p = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return int(p.pid)


def _run(cmd: List[str], *, timeout_seconds: float | None = None) -> Tuple[int, str, str]:
    timeout = None if timeout_seconds is None else max(float(timeout_seconds), 0.1)
    p: subprocess.Popen[str] | None = None
    try:
        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = p.communicate(timeout=timeout)
        return int(p.returncode or 0), (stdout or '').strip(), (stderr or '').strip()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b'').decode('utf-8', errors='ignore')
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b'').decode('utf-8', errors='ignore')
        if p is not None:
            for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                except Exception:
                    try:
                        p.send_signal(sig)
                    except Exception:
                        pass
                try:
                    tail_out, tail_err = p.communicate(timeout=grace)
                    stdout = tail_out or stdout
                    stderr = tail_err or stderr
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                if p.stdout is not None:
                    p.stdout.close()
                if p.stderr is not None:
                    p.stderr.close()
        detail = f'timeout_after_seconds={max(float(timeout_seconds or 0.0), 0.1):.1f}'
        if stderr.strip():
            detail = f'{detail} {stderr.strip()}'
        return 124, stdout.strip(), detail


def _latest_heartbeat_age_seconds(glob_pattern: str) -> float:
    if not glob_pattern:
        return 1e12
    newest: float = 0.0
    matched = False
    for fp in glob.glob(glob_pattern):
        try:
            ts = Path(fp).stat().st_mtime
            if (not matched) or ts > newest:
                newest = ts
                matched = True
        except Exception:
            continue
    if not matched:
        return 1e12
    return max(time.time() - newest, 0.0)


def _child_fanout_health(
    target: Dict[str, Any],
    *,
    running: int,
    alt_running: int,
    parent_elapsed_seconds: float | None,
) -> Dict[str, Any]:
    min_child_processes = max(_safe_int(target.get('min_child_processes'), 0), 0)
    grace_seconds = max(float(target.get('child_fanout_grace_seconds', 0.0) or 0.0), 0.0)
    if min_child_processes <= 0:
        return {
            'required': False,
            'ok': True,
            'reason': 'not_required',
            'min_child_processes': 0,
            'child_process_count': int(alt_running),
            'child_fanout_grace_seconds': round(float(grace_seconds), 3),
            'parent_elapsed_seconds': parent_elapsed_seconds,
        }
    if int(running) <= 0:
        ok = int(alt_running) >= int(min_child_processes)
        return {
            'required': True,
            'ok': bool(ok),
            'reason': 'parent_missing_child_fanout_present' if ok else 'parent_missing_child_fanout_below_floor',
            'min_child_processes': int(min_child_processes),
            'child_process_count': int(alt_running),
            'child_fanout_grace_seconds': round(float(grace_seconds), 3),
            'parent_elapsed_seconds': parent_elapsed_seconds,
        }
    if parent_elapsed_seconds is not None and float(parent_elapsed_seconds) < grace_seconds:
        return {
            'required': True,
            'ok': True,
            'reason': 'startup_grace',
            'min_child_processes': int(min_child_processes),
            'child_process_count': int(alt_running),
            'child_fanout_grace_seconds': round(float(grace_seconds), 3),
            'parent_elapsed_seconds': round(float(parent_elapsed_seconds), 3),
        }
    ok = int(alt_running) >= int(min_child_processes)
    return {
        'required': True,
        'ok': bool(ok),
        'reason': 'ready' if ok else 'child_fanout_below_floor',
        'min_child_processes': int(min_child_processes),
        'child_process_count': int(alt_running),
        'child_fanout_grace_seconds': round(float(grace_seconds), 3),
        'parent_elapsed_seconds': (
            round(float(parent_elapsed_seconds), 3) if parent_elapsed_seconds is not None else None
        ),
    }


def _load_state(path: Path, fallback: Path) -> Dict[str, Any]:
    for candidate in (path, fallback):
        try:
            return json.loads(candidate.read_text(encoding='utf-8'))
        except Exception:
            continue
    return {'events': []}


def _load_json_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iso_epoch(raw: Any) -> float | None:
    text = str(raw or '').strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def _all_sleeves_launcher_artifact_health(
    target: Dict[str, Any],
    *,
    now_epoch: float | None = None,
) -> Dict[str, Any]:
    raw_path = str(target.get('launcher_health_path') or '').strip()
    if not raw_path:
        return {'present': False, 'ok': False, 'reason': 'launcher_health_path_missing'}

    path = Path(raw_path)
    payload = _load_json_payload(path)
    if not payload:
        return {
            'present': False,
            'ok': False,
            'path': str(path),
            'reason': 'launcher_artifact_missing_or_invalid',
        }

    current = float(now_epoch if now_epoch is not None else time.time())
    ts_epoch = _iso_epoch(payload.get('timestamp_utc') or payload.get('updated_at_utc'))
    age_seconds = 1e12 if ts_epoch is None else max(current - ts_epoch, 0.0)
    max_age_seconds = max(
        _safe_float(target.get('heartbeat_max_age_seconds'), 0.0),
        _safe_float(target.get('child_fanout_grace_seconds'), 0.0),
        60.0,
    )
    overall_status = str(payload.get('overall_status') or payload.get('status') or '').strip().lower()
    phase = str(payload.get('phase') or payload.get('current_step') or '').strip().lower()
    expected = _safe_int(payload.get('expected_job_count'), 0)
    running = _safe_int(payload.get('running_job_count'), 0)
    missing = _safe_int(payload.get('missing_job_count'), 0)
    exited = _safe_int(payload.get('exited_job_count'), 0)
    policy_parked = _safe_int(payload.get('policy_parked_job_count'), 0)
    clean_exited = _safe_int(payload.get('clean_exited_job_count'), 0)
    repair_packet = payload.get('repair_packet') if isinstance(payload.get('repair_packet'), dict) else {}
    readiness_contract = (
        payload.get('launcher_readiness_contract')
        if isinstance(payload.get('launcher_readiness_contract'), dict)
        else {}
    )
    problem_default = (
        _safe_int(repair_packet.get('problem_job_count'), 0)
        if repair_packet
        else missing + exited
    )
    problem = _safe_int(payload.get('problem_job_count'), problem_default)
    exact_needs = payload.get('exact_needs') if isinstance(payload.get('exact_needs'), list) else []
    if not exact_needs:
        exact_needs = (
            readiness_contract.get('exact_needs')
            if isinstance(readiness_contract.get('exact_needs'), list)
            else []
        )
    fresh = bool(age_seconds <= max_age_seconds)
    complete = bool(expected > 0 and running >= expected and missing == 0 and exited == 0 and problem == 0 and not exact_needs)
    stable_non_running = bool(
        expected > 0
        and running + policy_parked + clean_exited >= expected
        and missing == 0
        and problem == 0
        and not exact_needs
    )
    ok = bool(fresh and (complete or stable_non_running) and phase == 'running' and overall_status in {'ready', 'guarded_ready'})
    if ok and complete:
        reason = 'fresh_launcher_artifact_certifies_full_fanout'
    elif ok:
        reason = 'fresh_launcher_artifact_certifies_stable_fanout'
    else:
        reason = 'launcher_artifact_not_certifying_fanout'
    if not fresh:
        reason = 'launcher_artifact_stale'
    elif not (complete or stable_non_running):
        reason = 'launcher_artifact_jobs_not_all_running'
    elif phase != 'running':
        reason = 'launcher_artifact_phase_not_running'
    elif overall_status not in {'ready', 'guarded_ready'}:
        reason = 'launcher_artifact_status_not_ready'

    return {
        'present': True,
        'ok': ok,
        'path': str(path),
        'reason': reason,
        'timestamp_utc': str(payload.get('timestamp_utc') or payload.get('updated_at_utc') or ''),
        'age_seconds': round(float(age_seconds), 3) if age_seconds < 1e11 else None,
        'max_age_seconds': round(float(max_age_seconds), 3),
        'overall_status': overall_status,
        'phase': phase,
        'expected_job_count': int(expected),
        'running_job_count': int(running),
        'missing_job_count': int(missing),
        'exited_job_count': int(exited),
        'policy_parked_job_count': int(policy_parked),
        'clean_exited_job_count': int(clean_exited),
        'problem_job_count': int(problem),
        'exact_need_count': len(exact_needs),
        'policy': 'fresh_all_sleeves_launcher_artifact_can_certify_child_fanout_when_wrapper_is_absent',
    }


def _save_state(path: Path, fallback: Path, state: Dict[str, Any]) -> Path:
    encoded = json.dumps(state, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return path
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return fallback


def _write_payload(path: Path, fallback: Path, payload: Dict[str, Any]) -> Path:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return path
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return fallback


def _bootstrap_runtime_env(profile: str) -> None:
    loader = PROJECT_ROOT / 'scripts' / 'ops' / 'load_runtime_env.sh'
    if not loader.exists():
        return

    normalized = (profile or 'live').strip() or 'live'
    if normalized not in {'sim', 'live'}:
        normalized = 'live'

    cmd = [
        '/bin/zsh',
        '-lc',
        f"source {shlex.quote(str(loader))} {shlex.quote(normalized)} --quiet >/dev/null && env -0",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return

    for chunk in proc.stdout.decode('utf-8', errors='ignore').split('\0'):
        if not chunk or '=' not in chunk:
            continue
        key, value = chunk.split('=', 1)
        if not key:
            continue
        if key not in os.environ or _placeholder_or_empty(os.environ.get(key, '')):
            os.environ[key] = value


def _load_env_override_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return values
    for raw_line in lines:
        line = str(raw_line or '').strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if not key or not key.replace('_', '').isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif ' #' in value:
            value = value.split(' #', 1)[0].strip()
        values[key] = value
    return values


def _paper_execution_runtime_pause_state(
    override_path: Path | None = None,
) -> Dict[str, Any]:
    path = Path(override_path) if override_path is not None else DEFAULT_RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH
    override_values = _load_env_override_values(path)

    def _value(name: str, default: str) -> str:
        if name in override_values:
            return str(override_values.get(name, default))
        return os.getenv(name, default)

    consumer_raw = _value('PAPER_EXECUTION_QUEUE_CONSUMER_ENABLED', '1').strip().lower()
    runtime_paused_raw = _value('PAPER_EXECUTION_RUNTIME_PAUSED_FOR_PRESSURE', '0').strip()
    ramp_blocked_raw = _value('PAPER_400_RAMP_BLOCKED_RUNTIME_PAUSE', '0').strip()
    consumer_disabled = consumer_raw in {'0', 'false', 'no', 'off'}
    runtime_paused = _safe_bool(runtime_paused_raw, default=False)
    ramp_blocked = _safe_bool(ramp_blocked_raw, default=False)
    paused = bool(consumer_disabled or runtime_paused or ramp_blocked)
    reasons: List[str] = []
    if consumer_disabled:
        reasons.append('paper_queue_consumer_disabled')
    if runtime_paused:
        reasons.append('runtime_pressure_pause')
    if ramp_blocked:
        reasons.append('paper_400_ramp_blocked')

    return {
        'paused': paused,
        'reason': '+'.join(reasons) if reasons else '',
        'consumer_enabled': not consumer_disabled,
        'runtime_paused_for_pressure': runtime_paused,
        'paper_400_ramp_blocked_runtime_pause': ramp_blocked,
        'override_file_present': path.exists(),
        'override_path': str(path),
        'policy': 'respect_runtime_paper_execution_pause_without_watchdog_restart',
    }


def _within_budget(events: List[Dict[str, Any]], name: str, max_per_hour: int) -> bool:
    cutoff = time.time() - 3600
    recent = [e for e in events if e.get('name') == name and float(e.get('ts_epoch', 0)) >= cutoff]
    return len(recent) < max(max_per_hour, 1)


def _last_restart_age_seconds(
    events: List[Dict[str, Any]],
    name: str,
    *,
    now_epoch: float | None = None,
) -> float | None:
    now = float(now_epoch or time.time())
    latest = 0.0
    for event in events:
        if event.get('event') != 'restart' or str(event.get('name') or '') != str(name or ''):
            continue
        latest = max(latest, float(event.get('ts_epoch', 0.0) or 0.0))
    if latest <= 0.0:
        return None
    return max(now - latest, 0.0)


def _restart_storm_impact(name: str, row: Dict[str, Any] | None = None) -> str:
    row = row or {}
    raw = str(row.get('restart_storm_impact') or '').strip().lower()
    if raw in RESTART_STORM_IMPACTS:
        return raw
    if str(name or '').startswith('execution_lane_'):
        return 'execution_lane'
    if str(name or '') in READ_ONLY_COLLECTION_RESTART_STORM_TARGETS:
        return 'read_only_collection'
    if str(name or '') == 'sql_link_writer':
        return 'storage_writer'
    return 'support_or_unknown'


def _restart_storm_quarantine_allowed(name: str, row: Dict[str, Any] | None = None) -> bool:
    row = row or {}
    impact = _restart_storm_impact(name, row)
    explicit_allowed = row.get('restart_storm_quarantine_allowed')
    allowed = _safe_bool(explicit_allowed, default=(impact == 'read_only_collection'))
    live_execution_critical = _safe_bool(row.get('live_execution_critical'), default=(impact == 'execution_lane'))
    return bool(impact == 'read_only_collection' and allowed and not live_execution_critical)


def _restart_storm_isolation_contract(restart_storms: List[Dict[str, Any]]) -> Dict[str, Any]:
    isolated: List[str] = []
    execution_blocking: List[str] = []
    for storm in restart_storms:
        if not isinstance(storm, dict) or bool(storm.get('resolved', False)):
            continue
        name = str(storm.get('name') or '').strip()
        if not name:
            continue
        quarantinable = _safe_bool(storm.get('quarantinable'), default=False)
        blocks_execution_clear = _safe_bool(storm.get('blocks_execution_clear'), default=not quarantinable)
        if not blocks_execution_clear:
            isolated.append(name)
        else:
            execution_blocking.append(name)

    return {
        'policy': 'isolate_non_execution_restart_storms_from_execution_clearance_while_execution_is_off',
        'isolated_count': len(isolated),
        'execution_blocking_count': len(execution_blocking),
        'isolated_targets': sorted(isolated),
        'execution_blocking_targets': sorted(execution_blocking),
        'all_active_storms_isolated': bool(isolated and not execution_blocking),
    }


def _restart_budget_alert_metadata(name: str, row: Dict[str, Any]) -> Tuple[str, str]:
    if _restart_storm_quarantine_allowed(name, row):
        return 'warn', 'watchdog_restart_budget_exhausted_isolated'
    return 'critical', 'watchdog_restart_budget_exhausted'


def _restart_budget_repair_probe(
    *,
    events: List[Dict[str, Any]],
    name: str,
    row: Dict[str, Any],
    cooldown_seconds: int,
    now_epoch: float | None = None,
) -> Dict[str, Any]:
    if not _env_flag('OPS_WATCHDOG_READONLY_BUDGET_REPAIR_PROBE', '1'):
        return {'allowed': False, 'reason': 'repair_probe_disabled'}
    if not _restart_storm_quarantine_allowed(name, row):
        return {'allowed': False, 'reason': 'not_read_only_quarantinable'}

    last_age = _last_restart_age_seconds(events, name, now_epoch=now_epoch)
    cooldown = max(int(cooldown_seconds), 60)
    if last_age is not None and last_age < cooldown:
        return {
            'allowed': False,
            'reason': 'repair_probe_cooldown',
            'last_restart_age_seconds': round(float(last_age), 3),
            'cooldown_seconds': int(cooldown),
        }

    return {
        'allowed': True,
        'reason': 'read_only_collector_repair_probe_after_restart_budget_exhausted',
        'last_restart_age_seconds': round(float(last_age), 3) if last_age is not None else None,
        'cooldown_seconds': int(cooldown),
        'policy': 'allow_bounded_read_only_collector_restart_probe_without_affecting_live_execution',
    }


def _resolved_restart_storms(
    *,
    events: List[Dict[str, Any]],
    status_rows: List[Dict[str, Any]],
    restart_window_seconds: int,
    restart_storm_threshold: int,
    settle_seconds: int,
    now_epoch: float | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    now = float(now_epoch or time.time())
    cutoff = now - max(int(restart_window_seconds), 60)
    by_name: Dict[str, int] = {}
    last_restart_epoch: Dict[str, float] = {}
    for event in events:
        if event.get('event') != 'restart':
            continue
        ts_epoch = float(event.get('ts_epoch', 0) or 0.0)
        if ts_epoch < cutoff:
            continue
        name = str(event.get('name', 'unknown'))
        by_name[name] = by_name.get(name, 0) + 1
        last_restart_epoch[name] = max(last_restart_epoch.get(name, 0.0), ts_epoch)

    status_by_name = {
        str((row or {}).get('name') or ''): row
        for row in status_rows
        if isinstance(row, dict) and str((row or {}).get('name') or '').strip()
    }
    active: List[Dict[str, Any]] = []
    recent: List[Dict[str, Any]] = []
    for name, count in sorted(by_name.items()):
        if count < max(int(restart_storm_threshold), 1):
            continue
        row = status_by_name.get(name, {})
        row_settle_seconds = max(int(row.get('restart_storm_settle_seconds', settle_seconds) or settle_seconds), 60)
        row_min_healthy_seconds = max(
            int(row.get('restart_storm_min_healthy_seconds', min(row_settle_seconds, 120)) or min(row_settle_seconds, 120)),
            60,
        )
        if bool(row.get('parent_process_required', False)):
            if bool(row.get('effective_process_live', False)) or bool(row.get('launcher_artifact_certified_fanout', False)):
                running_count = 1
            else:
                running_count = int(row.get('running', 0) or 0)
        else:
            running_count = int(row.get('running', 0) or 0) + int(row.get('alt_running', 0) or 0)
        heartbeat_ok = bool(row.get('heartbeat_ok', False))
        heartbeat_age_seconds = float(row.get('heartbeat_age_seconds', 0.0) or 0.0)
        heartbeat_max_age_seconds = float(row.get('heartbeat_max_age_seconds', 0.0) or 0.0)
        paused_by_safety_flags = bool(row.get('paused_by_safety_flags', False))
        paused_by_creative_cotenant = bool(row.get('paused_by_creative_cotenant_guard', False))
        paused_by_runtime_gate = bool(row.get('paused_by_runtime_gate', False))
        last_age = max(now - last_restart_epoch.get(name, now), 0.0)
        unresolved = running_count <= 0 or not heartbeat_ok or last_age < row_settle_seconds
        healthy_and_fresh = (
            running_count > 0
            and heartbeat_ok
            and (heartbeat_max_age_seconds <= 0.0 or heartbeat_age_seconds <= heartbeat_max_age_seconds)
        )
        if healthy_and_fresh and last_age >= row_min_healthy_seconds:
            unresolved = False
        if paused_by_safety_flags or paused_by_creative_cotenant or paused_by_runtime_gate:
            unresolved = False
        sql_writer_idle_complete = bool(
            name == 'sql_link_writer'
            and row.get('writer_idle_ok', False)
            and not _safe_bool(row.get('live_execution_critical'), default=False)
        )
        if sql_writer_idle_complete:
            unresolved = False
        impact = _restart_storm_impact(name, row)
        quarantinable = _restart_storm_quarantine_allowed(name, row)
        live_execution_critical = _safe_bool(row.get('live_execution_critical'), default=(impact == 'execution_lane'))
        sql_writer_recovered = bool(
            name == 'sql_link_writer'
            and row.get('writer_recovered_ok', False)
            and not live_execution_critical
        )
        if sql_writer_recovered:
            unresolved = False
        blocks_execution_clear = bool(live_execution_critical or impact == 'execution_lane')
        storm = {
            'name': name,
            'count': int(count),
            'window_seconds': int(restart_window_seconds),
            'last_restart_age_seconds': round(float(last_age), 3),
            'settle_seconds': int(row_settle_seconds),
            'resolved': not unresolved,
            'impact': impact,
            'quarantinable': bool(quarantinable),
            'quarantine_state': 'isolated_read_only_collection' if unresolved and quarantinable else '',
            'live_execution_critical': bool(live_execution_critical),
            'blocks_execution_clear': bool(blocks_execution_clear),
        }
        if paused_by_safety_flags:
            storm['resolution_reason'] = str(row.get('safety_pause_reason') or 'paused_by_safety_flags')
        elif paused_by_creative_cotenant:
            storm['resolution_reason'] = str(row.get('creative_pause_reason') or 'creative_cotenant_pause_active')
        elif paused_by_runtime_gate:
            storm['resolution_reason'] = str(row.get('runtime_pause_reason') or 'runtime_paper_execution_paused')
        elif sql_writer_idle_complete:
            storm['resolution_reason'] = 'sql_writer_on_demand_idle_complete'
        elif sql_writer_recovered:
            storm['resolution_reason'] = 'sql_writer_active_progress_recovered'
        recent.append(storm)
        if unresolved:
            active.append(storm)
    return active, recent


def _forgive_resolved_restart_debt(
    events: List[Dict[str, Any]],
    recent_restart_storms: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    forgiven_names = {
        str(storm.get('name') or '')
        for storm in recent_restart_storms
        if bool(storm.get('resolved', False)) and str(storm.get('name') or '').strip()
    }
    if not forgiven_names:
        return events, {
            'active': False,
            'forgiven_names': [],
            'removed_event_count': 0,
            'policy': 'resolved_restart_storms_keep_history_until_storm_recovery_is_observed',
        }

    kept: List[Dict[str, Any]] = []
    removed = 0
    for event in events:
        if event.get('event') == 'restart' and str(event.get('name') or '') in forgiven_names:
            removed += 1
            continue
        kept.append(event)
    return kept, {
        'active': bool(removed),
        'forgiven_names': sorted(forgiven_names),
        'removed_event_count': int(removed),
        'policy': 'clear_restart_budget_debt_after_target_is_running_with_fresh_heartbeat_and_storm_is_resolved',
    }


def _sql_link_writer_idle_health() -> Dict[str, Any]:
    cycle_path = HEALTH_DIR / 'writer_cycle_coordinator_latest.json'
    process_path = HEALTH_DIR / 'writer_process_intelligence_latest.json'
    progress_path = HEALTH_DIR / 'sql_link_service_progress_latest.json'
    queue_path = HEALTH_DIR / 'ingestion_backpressure_latest.json'
    storage_path = HEALTH_DIR / 'ingestion_storage_control_latest.json'
    cycle_payload = _load_json_payload(cycle_path)
    process_payload = _load_json_payload(process_path)
    progress_payload = _load_json_payload(progress_path)

    def _fresh(path: Path, max_age_seconds: float = 600.0) -> bool:
        try:
            return (time.time() - float(path.stat().st_mtime)) <= max_age_seconds
        except Exception:
            return False

    def _state_idle(state: Dict[str, Any]) -> bool:
        if not state:
            return False
        if state.get('ok') is False:
            return False
        status = str(state.get('overall_status') or state.get('status') or '').strip().lower()
        if status in {'critical', 'degraded', 'error', 'failed', 'failure'}:
            return False
        current_step = str(state.get('effective_current_step') or state.get('current_step') or '').strip()
        completed_shards = _safe_int(state.get('completed_shard_count'), 0)
        planned_shards = _safe_int(state.get('planned_shard_count'), 0)
        pending_shards = _safe_int(state.get('pending_shard_count'), 0)
        timed_out_shards = _safe_int(state.get('timed_out_shard_count'), 0)
        return (
            current_step == 'complete'
            and not bool(state.get('active', False))
            and not bool(state.get('running', False))
            and not bool(state.get('child_writer_active', False))
            and not bool(state.get('writer_lock_held', False))
            and pending_shards == 0
            and timed_out_shards == 0
            and (planned_shards <= 0 or completed_shards >= planned_shards)
        )

    cycle_state = {}
    for key in ('writer_state_after_wait', 'writer_state_after_remediation', 'writer_state_before'):
        candidate = cycle_payload.get(key)
        if isinstance(candidate, dict) and candidate:
            cycle_state = candidate
            break

    process_health = process_payload.get('writer_health')
    if not isinstance(process_health, dict):
        process_health = {}

    queue_payload: Dict[str, Any] = {}
    queue_source = ''
    if _fresh(queue_path, 300.0):
        queue_payload = _load_json_payload(queue_path)
        queue_source = 'ingestion_backpressure'
    elif _fresh(storage_path, 300.0):
        storage_payload = _load_json_payload(storage_path)
        candidate = storage_payload.get('backpressure') if isinstance(storage_payload.get('backpressure'), dict) else {}
        if candidate:
            queue_payload = candidate
            queue_source = 'ingestion_storage_control.backpressure'
    core_pending = _safe_int(
        queue_payload.get('core_pending_lines', queue_payload.get('pending_lines')),
        0,
    )
    total_pending = _safe_int(
        queue_payload.get('total_pending_lines', queue_payload.get('pending_lines_total')),
        core_pending,
    )
    oldest_pending_age = _safe_float(queue_payload.get('oldest_pending_age_seconds'), 0.0)
    idle_max_core = max(_safe_int(os.getenv('OPS_WATCHDOG_SQL_WRITER_IDLE_MAX_CORE_PENDING_LINES'), 500), 0)
    idle_max_total = max(_safe_int(os.getenv('OPS_WATCHDOG_SQL_WRITER_IDLE_MAX_TOTAL_PENDING_LINES'), 1500), idle_max_core)
    idle_max_age = max(_safe_float(os.getenv('OPS_WATCHDOG_SQL_WRITER_IDLE_MAX_AGE_SECONDS'), 300.0), 0.0)
    queue_idle_clear = bool(
        queue_source
        and core_pending <= idle_max_core
        and total_pending <= idle_max_total
        and oldest_pending_age <= idle_max_age
    )

    cycle_idle = _fresh(cycle_path) and _state_idle(cycle_state)
    process_idle = _fresh(process_path) and _state_idle(process_health)
    progress_idle = _fresh(progress_path) and _state_idle(progress_payload)
    artifact_idle = bool(cycle_idle or process_idle or progress_idle)
    ok = bool(artifact_idle and queue_idle_clear)
    if ok:
        reason = 'sql_writer_on_demand_idle_complete'
    elif artifact_idle and not queue_source:
        reason = 'sql_writer_idle_queue_evidence_missing_or_stale'
    elif artifact_idle:
        reason = 'sql_writer_idle_backlog_pending'
    else:
        reason = 'sql_writer_idle_health_not_clear'
    return {
        'ok': ok,
        'reason': reason,
        'cycle_artifact_fresh': _fresh(cycle_path),
        'process_artifact_fresh': _fresh(process_path),
        'progress_artifact_fresh': _fresh(progress_path),
        'cycle_idle_complete': bool(cycle_idle),
        'process_idle_complete': bool(process_idle),
        'progress_idle_complete': bool(progress_idle),
        'queue_evidence_source': queue_source,
        'queue_idle_clear': queue_idle_clear,
        'core_pending_lines': core_pending,
        'total_pending_lines': total_pending,
        'oldest_pending_age_seconds': round(oldest_pending_age, 3),
        'idle_limits': {
            'core_pending_lines': idle_max_core,
            'total_pending_lines': idle_max_total,
            'oldest_pending_age_seconds': idle_max_age,
        },
        'cycle_current_step': str(cycle_state.get('effective_current_step') or cycle_state.get('current_step') or ''),
        'process_current_step': str(process_health.get('current_step') or ''),
        'progress_current_step': str(progress_payload.get('effective_current_step') or progress_payload.get('current_step') or ''),
        'completed_shard_count': _safe_int(
            cycle_state.get('completed_shard_count', process_health.get('completed_shard_count')),
            0,
        ),
        'planned_shard_count': _safe_int(
            cycle_state.get('planned_shard_count', process_health.get('planned_shard_count')),
            0,
        ),
        'writer_lock_held': bool(cycle_state.get('writer_lock_held', process_health.get('writer_lock_held', False))),
        'policy': 'treat fresh complete writer progress as healthy idle only while fresh queue evidence remains below bounded idle ceilings',
    }


def _sql_link_writer_recovery_health() -> Dict[str, Any]:
    cycle_path = HEALTH_DIR / 'writer_cycle_coordinator_latest.json'
    process_path = HEALTH_DIR / 'writer_process_intelligence_latest.json'
    cycle_payload = _load_json_payload(cycle_path)
    process_payload = _load_json_payload(process_path)

    def _fresh(path: Path, max_age_seconds: float = 600.0) -> bool:
        try:
            return (time.time() - float(path.stat().st_mtime)) <= max_age_seconds
        except Exception:
            return False

    cycle_state: Dict[str, Any] = {}
    for key in ('writer_state_after_wait', 'writer_state_after_remediation', 'writer_state_before'):
        candidate = cycle_payload.get(key)
        if isinstance(candidate, dict) and candidate:
            cycle_state = candidate
            break

    process_health = process_payload.get('writer_health')
    if not isinstance(process_health, dict):
        process_health = {}

    state = cycle_state if cycle_state else process_health
    artifact_fresh = _fresh(cycle_path) or _fresh(process_path)
    current_step = str(state.get('effective_current_step') or state.get('current_step') or process_health.get('state') or '').strip()
    progress_age_minutes = _safe_float(
        state.get('progress_age_minutes', process_health.get('progress_age_minutes')),
        999.0,
    )
    active = bool(
        state.get('active', False)
        or state.get('running', False)
        or process_health.get('active', False)
        or str(process_health.get('state') or '').strip().lower() == 'active_progressing'
    )
    orphaned = bool(state.get('progress_orphaned', process_health.get('progress_orphaned', False)))
    handoff_needed = bool(
        state.get('complete_lock_handoff_needed', state.get('completed_lock_handoff_needed', False))
        or process_health.get('complete_lock_handoff_needed', False)
    )
    ok = bool(
        artifact_fresh
        and active
        and not orphaned
        and not handoff_needed
        and progress_age_minutes <= 10.0
        and current_step not in {'', 'stalled', 'stale_progress'}
    )
    return {
        'ok': ok,
        'reason': 'sql_writer_active_progress_recovered' if ok else 'sql_writer_active_progress_not_clear',
        'cycle_artifact_fresh': _fresh(cycle_path),
        'process_artifact_fresh': _fresh(process_path),
        'current_step': current_step,
        'active': bool(active),
        'progress_age_minutes': round(float(progress_age_minutes), 3),
        'progress_orphaned': bool(orphaned),
        'complete_lock_handoff_needed': bool(handoff_needed),
        'writer_lock_held': bool(state.get('writer_lock_held', process_health.get('writer_lock_held', False))),
        'child_writer_active': bool(state.get('child_writer_active', process_health.get('child_writer_active', False))),
        'completed_shard_count': _safe_int(
            state.get('completed_shard_count', process_health.get('completed_shard_count')),
            0,
        ),
        'planned_shard_count': _safe_int(
            state.get('planned_shard_count', process_health.get('planned_shard_count')),
            0,
        ),
        'policy': 'fresh_active_sql_writer_progress_forgives_restart_storm_debt_without_enabling_live_execution',
    }


INTENTIONAL_RESTART_SKIPS = {
    'paused_by_safety_flags',
    'creative_cotenant_pause_active',
    'network_outage_active',
    'runtime_paper_execution_paused',
}

INTENTIONAL_STARTUP_REASONS = {
    'process_fanout_guard_active',
    'process_fanout_guard_core_sleeve_pressure_mode',
    'all_sleeves_explicitly_paused_by_operator_mode',
    'all_sleeves_explicitly_paused_for_computer_task',
    'operator_mode_paused_training_or_research',
    'computer_task_paused_training_or_research',
    'operator_mode_daily_driver',
    'computer_normal_use_governor_active',
    'operator_mode_backlog_intake_governor',
}


def _row_intentionally_held(row: Dict[str, Any]) -> bool:
    skip = str(row.get('restart_skipped') or '').strip()
    reason = str(row.get('reason') or row.get('safety_pause_reason') or row.get('creative_pause_reason') or '').strip()
    if (
        bool(row.get('paused_by_safety_flags', False))
        or bool(row.get('paused_by_creative_cotenant_guard', False))
        or bool(row.get('paused_by_runtime_gate', False))
    ):
        return True
    if skip in INTENTIONAL_RESTART_SKIPS:
        return True
    if skip == 'startup_not_ready' and reason in INTENTIONAL_STARTUP_REASONS:
        return True
    return False


def _target_repair_command(name: str, row: Dict[str, Any]) -> List[str]:
    if row.get('repair_commands'):
        command = row.get('repair_commands')[0]
        if isinstance(command, list) and command:
            return [str(part) for part in command]
    if name == 'all_sleeves':
        return ['./scripts/ops/opsctl.sh', 'watchdog-intelligence', '--apply', '--json']
    if name.startswith('coinbase'):
        return ['./scripts/ops/opsctl.sh', 'coinbase-api-health', '--snapshot', '--json']
    if name == 'sql_link_writer':
        return ['./scripts/ops/opsctl.sh', 'writer-process-intelligence', '--json']
    return ['./scripts/ops/opsctl.sh', 'health-fast', '--json']


def _row_effective_process_live(row: Dict[str, Any]) -> bool:
    return bool(
        row.get('process_live', False)
        or row.get('effective_process_live', False)
        or row.get('launcher_artifact_certified_fanout', False)
    )


def _row_effective_heartbeat_ok(row: Dict[str, Any]) -> bool:
    if bool(row.get('heartbeat_ok', False)):
        return True
    return bool(
        str(row.get('name') or '') == 'all_sleeves'
        and row.get('launcher_artifact_certified_fanout', False)
        and _row_effective_process_live(row)
        and bool(row.get('child_fanout_ok', True))
    )


def _watchdog_need_for_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    name = str(row.get('name') or 'unknown')
    heartbeat_ok = _row_effective_heartbeat_ok(row)
    process_live = _row_effective_process_live(row)
    intentionally_held = _row_intentionally_held(row)
    if name == 'sql_link_writer' and bool(row.get('writer_idle_ok', False)):
        return None
    if heartbeat_ok and process_live:
        return None

    restart_skipped = str(row.get('restart_skipped') or '')
    reason = str(row.get('reason') or row.get('restart_reason') or restart_skipped or 'unknown')
    if intentionally_held:
        return {
            'target': name,
            'severity': 'info',
            'status': 'intentional_hold',
            'blocker': reason or restart_skipped or 'guarded_downshift',
            'exact_file': str(DEFAULT_OUT_PATH),
            'exact_command': ['./scripts/ops/opsctl.sh', 'watchdog-intelligence', '--json'],
            'expected_impact': 'keeps the launcher quiet while another guard intentionally protects the computer',
            'risk_level': 'low',
            'when_to_stop': 'stop when the guard hold clears or the target reports heartbeat_ok=true',
        }

    if not process_live:
        blocker = 'process_missing'
        expected = 'restart the missing read-only watchdog target and refresh its heartbeat'
    elif not heartbeat_ok:
        blocker = 'heartbeat_stale'
        expected = 'repair or restart the stale target after preserving restart budget'
    else:
        blocker = reason or 'unknown_watchdog_issue'
        expected = 'inspect the target state and repair the specific failing lane'

    if restart_skipped == 'budget_exhausted':
        blocker = 'restart_budget_exhausted'
        expected = 'prevent a restart storm by pausing restarts until the target is stable'

    impact = _restart_storm_impact(name, row)
    quarantinable = _restart_storm_quarantine_allowed(name, row)
    severity = 'critical' if blocker in {'restart_budget_exhausted', 'process_missing'} else 'warn'
    if quarantinable:
        severity = 'warn'
        expected = 'quarantine the read-only collector while repair runs; do not widen or enable live execution from this signal'

    return {
        'target': name,
        'severity': severity,
        'status': 'needs_repair',
        'blocker': blocker,
        'reason': reason,
        'exact_file': str(DEFAULT_OUT_PATH),
        'exact_command': _target_repair_command(name, row),
        'expected_impact': expected,
        'risk_level': 'medium' if impact == 'execution_lane' else 'low',
        'restart_storm_impact': impact,
        'restart_storm_quarantinable': bool(quarantinable),
        'when_to_stop': 'stop after heartbeat_ok=true and restart_storms=[] for the target',
    }


def _watchdog_intelligence_contract(
    *,
    status_rows: List[Dict[str, Any]],
    restarts: List[Dict[str, Any]],
    restart_storms: List[Dict[str, Any]],
    recent_restart_storms: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    safety_pause: Dict[str, Any],
    creative_pause: Dict[str, Any],
    network_payload: Dict[str, Any],
) -> Dict[str, Any]:
    needs = [need for row in status_rows if (need := _watchdog_need_for_row(row))]
    active_needs = [need for need in needs if need.get('status') != 'intentional_hold']
    intentional_holds = [need for need in needs if need.get('status') == 'intentional_hold']
    stale_targets = [
        str(row.get('name') or '')
        for row in status_rows
        if row.get('heartbeat_fresh') is False
        and not _row_effective_heartbeat_ok(row)
        and not _row_intentionally_held(row)
    ]
    missing_targets = [
        str(row.get('name') or '')
        for row in status_rows
        if not _row_effective_process_live(row) and not _row_intentionally_held(row)
    ]
    restart_budget_blocks = [
        str(row.get('name') or '')
        for row in status_rows
        if str(row.get('restart_skipped') or '') == 'budget_exhausted'
    ]
    restart_budget_execution_blocks = [
        str(row.get('name') or '')
        for row in status_rows
        if str(row.get('restart_skipped') or '') == 'budget_exhausted'
        and not _restart_storm_quarantine_allowed(str(row.get('name') or ''), row)
    ]
    restart_budget_isolated_blocks = [
        str(row.get('name') or '')
        for row in status_rows
        if str(row.get('restart_skipped') or '') == 'budget_exhausted'
        and _restart_storm_quarantine_allowed(str(row.get('name') or ''), row)
    ]
    restart_storm_isolation = _restart_storm_isolation_contract(restart_storms)
    score = 100.0
    score -= min(float(restart_storm_isolation['execution_blocking_count']) * 28.0, 56.0)
    score -= min(float(restart_storm_isolation['isolated_count']) * 12.0, 24.0)
    score -= min(float(len(alerts)) * 12.0, 36.0)
    score -= min(float(len(active_needs)) * 16.0, 48.0)
    score -= min(float(len(restarts)) * 5.0, 20.0)
    score -= min(float(len(intentional_holds)) * 2.0, 8.0)
    score = max(round(score, 1), 0.0)

    if int(restart_storm_isolation['execution_blocking_count']) > 0 or restart_budget_execution_blocks:
        overall_status = 'critical'
    elif restart_storms or restart_budget_blocks or active_needs or alerts:
        overall_status = 'degraded'
    elif intentional_holds or recent_restart_storms:
        overall_status = 'ready'
    else:
        overall_status = 'ready'

    if score >= 94:
        grade = 'A'
    elif score >= 86:
        grade = 'B'
    elif score >= 76:
        grade = 'C'
    elif score >= 66:
        grade = 'D'
    else:
        grade = 'F'

    recommended_commands: List[List[str]] = []
    for need in active_needs[:5]:
        command = need.get('exact_command')
        if isinstance(command, list) and command and command not in recommended_commands:
            recommended_commands.append([str(part) for part in command])
    if not recommended_commands:
        recommended_commands.append(['./scripts/ops/opsctl.sh', 'watchdog-intelligence', '--json'])

    return {
        'overall_status': overall_status,
        'grade': grade,
        'score': score,
        'target_count': len(status_rows),
        'healthy_target_count': sum(1 for row in status_rows if _row_effective_heartbeat_ok(row)),
        'active_issue_count': len(active_needs),
        'intentional_hold_count': len(intentional_holds),
        'restart_count': len(restarts),
        'restart_storm_count': len(restart_storms),
        'recent_restart_storm_count': len(recent_restart_storms),
        'restart_storm_isolation': restart_storm_isolation,
        'alert_count': len(alerts),
        'stale_targets': [name for name in stale_targets if name],
        'missing_targets': [name for name in missing_targets if name],
        'restart_budget_blocks': [name for name in restart_budget_blocks if name],
        'restart_budget_execution_blocks': [name for name in restart_budget_execution_blocks if name],
        'restart_budget_isolated_blocks': [name for name in restart_budget_isolated_blocks if name],
        'exact_needs': needs,
        'recommended_commands': recommended_commands,
        'notification_policy': {
            'suppress_intentional_holds': True,
            'suppress_resolved_restart_storms': True,
            'escalate_active_restart_storms': True,
        },
        'guard_context': {
            'safety_pause_active': bool(safety_pause.get('active', False)),
            'creative_cotenant_pause_active': bool(creative_pause.get('active', False)),
            'network_outage_active': bool(network_payload.get('outage_active', False)),
        },
    }


def _file_age_seconds(path: Path) -> float:
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except Exception:
        return 1e12


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _resource_guard_allows_job(job_name: str, profile: str = 'optional') -> tuple[bool, str]:
    guard_script = PROJECT_ROOT / 'scripts' / 'resource_guard.py'
    if not guard_script.exists():
        return True, 'resource_guard_missing'
    rc, stdout, err = _run([str(PY), str(guard_script), '--profile', profile])
    detail = (stdout or err or '').strip()
    if not detail:
        detail = f'rc={rc}'
    return rc == 0, f'{job_name}:{detail}'


def _refresh_health_fast(max_age_seconds: int) -> Dict[str, Any]:
    health_fast = PROJECT_ROOT / 'governance' / 'health' / 'health_fast_latest.json'
    freshness_budget = max(int(max_age_seconds), 60)
    row: Dict[str, Any] = {
        'path': str(health_fast),
        'refreshed': False,
        'age_seconds_before': round(_file_age_seconds(health_fast), 2),
        'rc': 0,
        'error': '',
        'freshness_budget_seconds': freshness_budget,
        'lightweight_always_on': True,
    }
    if _file_age_seconds(health_fast) > float(freshness_budget):
        rc, _stdout, err = _run([
            str(PY),
            str(PROJECT_ROOT / 'scripts' / 'ops' / 'health_fast.py'),
            '--json',
        ])
        row['refreshed'] = rc in {0, 2}
        row['rc'] = int(rc)
        row['error'] = err[-500:] if rc not in {0, 2} and err else ''
    row['age_seconds_after'] = round(_file_age_seconds(health_fast), 2)
    return row


def _refresh_runtime_reports(
    max_age_seconds: int,
    *,
    backpressure_max_age_seconds: int | None = None,
    divergence_max_age_seconds: int | None = None,
    health_fast_max_age_seconds: int | None = None,
) -> Dict[str, Any]:
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    one_numbers = PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json'
    one_numbers_health = PROJECT_ROOT / 'governance' / 'health' / 'one_numbers_latest.json'
    daily_summary = PROJECT_ROOT / 'exports' / 'sql_reports' / f'daily_runtime_summary_{day}.json'
    daily_summary_health = PROJECT_ROOT / 'governance' / 'health' / 'daily_runtime_summary_latest.json'
    paper_performance = PROJECT_ROOT / 'governance' / 'health' / 'paper_performance_latest.json'
    backpressure = PROJECT_ROOT / 'governance' / 'health' / 'ingestion_backpressure_latest.json'
    divergence = PROJECT_ROOT / 'governance' / 'health' / 'data_source_divergence_latest.json'
    backpressure_age = max(int(backpressure_max_age_seconds or max_age_seconds), 60)
    divergence_age = max(int(divergence_max_age_seconds or max_age_seconds), 60)
    stuck_refresh_seconds = max(int(os.getenv('OPS_WATCHDOG_REFRESH_STUCK_SECONDS', '900') or 900), 120)

    out: Dict[str, Any] = {
        'one_numbers': {
            'path': str(one_numbers),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(one_numbers), 2),
            'rc': 0,
            'error': '',
            'synced_health': False,
        },
        'paper_performance': {
            'path': str(paper_performance),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(paper_performance), 2),
            'rc': 0,
            'error': '',
            'synced_health': True,
        },
        'daily_runtime_summary': {
            'path': str(daily_summary),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(daily_summary), 2),
            'rc': 0,
            'error': '',
            'synced_health': False,
        },
        'ingestion_backpressure': {
            'path': str(backpressure),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(backpressure), 2),
            'rc': 0,
            'error': '',
        },
        'data_source_divergence': {
            'path': str(divergence),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(divergence), 2),
            'rc': 0,
            'error': '',
        },
        'health_fast': _refresh_health_fast(int(health_fast_max_age_seconds or 300)),
    }

    if _file_age_seconds(one_numbers) > float(max_age_seconds):
        guard_ok, guard_detail = _resource_guard_allows_job('one_numbers_refresh', profile='refresh')
        out['one_numbers']['resource_guard_ok'] = bool(guard_ok)
        out['one_numbers']['resource_guard_detail'] = guard_detail
        if not guard_ok:
            out['one_numbers']['error'] = 'resource_guard_blocked'
        elif _proc_running('scripts/build_one_numbers_report.py') > 0:
            running_seconds = _proc_elapsed_seconds('scripts/build_one_numbers_report.py')
            out['one_numbers']['running_seconds'] = round(float(running_seconds), 2) if running_seconds is not None else None
            out['one_numbers']['error'] = (
                'refresh_stuck_suspected'
                if running_seconds is not None and float(running_seconds) >= float(stuck_refresh_seconds)
                else 'refresh_already_running'
            )
        else:
            rc, _stdout, err = _run([
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'build_one_numbers_report.py'),
            ])
            out['one_numbers']['refreshed'] = rc == 0
            out['one_numbers']['rc'] = int(rc)
            out['one_numbers']['error'] = err[-500:] if err else ''

    out['one_numbers']['synced_health'] = _copy_if_exists(one_numbers, one_numbers_health)
    out['one_numbers']['age_seconds_after'] = round(_file_age_seconds(one_numbers), 2)

    if _file_age_seconds(paper_performance) > float(max_age_seconds):
        guard_ok, guard_detail = _resource_guard_allows_job('paper_performance_refresh', profile='refresh')
        out['paper_performance']['resource_guard_ok'] = bool(guard_ok)
        out['paper_performance']['resource_guard_detail'] = guard_detail
        if not guard_ok:
            out['paper_performance']['error'] = 'resource_guard_blocked'
        elif _proc_running('scripts/paper_performance_report.py') > 0:
            running_seconds = _proc_elapsed_seconds('scripts/paper_performance_report.py')
            out['paper_performance']['running_seconds'] = round(float(running_seconds), 2) if running_seconds is not None else None
            out['paper_performance']['error'] = (
                'refresh_stuck_suspected'
                if running_seconds is not None and float(running_seconds) >= float(stuck_refresh_seconds)
                else 'refresh_already_running'
            )
        else:
            rc, _stdout, err = _run([
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'paper_performance_report.py'),
                '--day',
                day,
                '--week-days',
                os.getenv('PAPER_PERFORMANCE_REFRESH_WEEK_DAYS', '7'),
                '--json-only',
            ])
            out['paper_performance']['refreshed'] = rc == 0
            out['paper_performance']['rc'] = int(rc)
            out['paper_performance']['error'] = err[-500:] if err else ''

    out['paper_performance']['age_seconds_after'] = round(_file_age_seconds(paper_performance), 2)

    if _file_age_seconds(daily_summary) > float(max_age_seconds):
        guard_profile = str(os.getenv('OPS_WATCHDOG_DAILY_SUMMARY_RESOURCE_GUARD_PROFILE', 'refresh') or 'refresh')
        guard_ok, guard_detail = _resource_guard_allows_job('daily_runtime_summary_refresh', profile=guard_profile)
        out['daily_runtime_summary']['resource_guard_ok'] = bool(guard_ok)
        out['daily_runtime_summary']['resource_guard_detail'] = guard_detail
        if not guard_ok:
            out['daily_runtime_summary']['error'] = 'resource_guard_blocked'
        else:
            rc, stdout, err = _run([str(PY), str(PROJECT_ROOT / 'scripts' / 'daily_runtime_summary.py'), '--day', day, '--json'])
            out['daily_runtime_summary']['refreshed'] = rc == 0
            out['daily_runtime_summary']['rc'] = int(rc)
            out['daily_runtime_summary']['error'] = err[-500:] if err else ''
            if rc == 0 and stdout:
                try:
                    payload = json.loads(stdout)
                    daily_summary.parent.mkdir(parents=True, exist_ok=True)
                    daily_summary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
                except Exception as exc:
                    out['daily_runtime_summary']['error'] = f'parse_or_write_failed:{exc}'

    out['daily_runtime_summary']['synced_health'] = _copy_if_exists(daily_summary, daily_summary_health)
    out['daily_runtime_summary']['age_seconds_after'] = round(_file_age_seconds(daily_summary), 2)

    if _file_age_seconds(backpressure) > float(backpressure_age):
        guard_ok, guard_detail = _resource_guard_allows_job('ingestion_backpressure_refresh', profile='refresh')
        out['ingestion_backpressure']['resource_guard_ok'] = bool(guard_ok)
        out['ingestion_backpressure']['resource_guard_detail'] = guard_detail
        if not guard_ok:
            out['ingestion_backpressure']['error'] = 'resource_guard_blocked'
        elif _proc_running('scripts/ingestion_backpressure_guard.py') > 0:
            running_seconds = _proc_elapsed_seconds('scripts/ingestion_backpressure_guard.py')
            out['ingestion_backpressure']['running_seconds'] = round(float(running_seconds), 2) if running_seconds is not None else None
            out['ingestion_backpressure']['error'] = (
                'refresh_stuck_suspected'
                if running_seconds is not None and float(running_seconds) >= float(stuck_refresh_seconds)
                else 'refresh_already_running'
            )
        else:
            rc, _stdout, err = _run([
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'ingestion_backpressure_guard.py'),
                '--json',
            ])
            out['ingestion_backpressure']['refreshed'] = rc == 0
            out['ingestion_backpressure']['rc'] = int(rc)
            out['ingestion_backpressure']['error'] = err[-500:] if err else ''
    out['ingestion_backpressure']['age_seconds_after'] = round(_file_age_seconds(backpressure), 2)

    if _file_age_seconds(divergence) > float(divergence_age):
        guard_ok, guard_detail = _resource_guard_allows_job('data_source_divergence_refresh', profile='refresh')
        out['data_source_divergence']['resource_guard_ok'] = bool(guard_ok)
        out['data_source_divergence']['resource_guard_detail'] = guard_detail
        if not guard_ok:
            out['data_source_divergence']['error'] = 'resource_guard_blocked'
        elif _proc_running('scripts/data_source_divergence_bot.py') > 0:
            running_seconds = _proc_elapsed_seconds('scripts/data_source_divergence_bot.py')
            out['data_source_divergence']['running_seconds'] = round(float(running_seconds), 2) if running_seconds is not None else None
            out['data_source_divergence']['error'] = (
                'refresh_stuck_suspected'
                if running_seconds is not None and float(running_seconds) >= float(stuck_refresh_seconds)
                else 'refresh_already_running'
            )
        else:
            rc, _stdout, err = _run([
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'data_source_divergence_bot.py'),
                '--json',
            ])
            out['data_source_divergence']['refreshed'] = rc in {0, 2}
            out['data_source_divergence']['rc'] = int(rc)
            out['data_source_divergence']['error'] = err[-500:] if err else ''
    out['data_source_divergence']['age_seconds_after'] = round(_file_age_seconds(divergence), 2)
    return out


def _fanout_guard_allows_core_sleeve_restart() -> Tuple[bool, str]:
    if not (_env_flag('PROCESS_FANOUT_GUARD_ACTIVE', '0') or _env_flag('TRAINING_RUNTIME_PAUSED_FOR_FANOUT', '0')):
        return True, 'fanout_guard_inactive'
    if _env_flag('PROCESS_FANOUT_GUARD_CORE_SLEEVE_RESTART_ALLOWED', '0'):
        return True, 'process_fanout_guard_core_sleeve_pressure_mode'

    payload = _load_json_payload(HEALTH_DIR / 'process_fanout_guard_latest.json')
    if not payload:
        return False, 'process_fanout_guard_active'
    startup_policy = payload.get('startup_policy') if isinstance(payload.get('startup_policy'), dict) else {}
    if bool(startup_policy.get('core_sleeve_restart_allowed', False)):
        return True, 'process_fanout_guard_core_sleeve_pressure_mode'
    fanout = payload.get('fanout') if isinstance(payload.get('fanout'), dict) else {}
    kill_plan = payload.get('kill_plan') if isinstance(payload.get('kill_plan'), list) else []
    if _safe_int(fanout.get('targetable_count'), 0) <= 0 and not kill_plan:
        return True, 'process_fanout_guard_core_sleeve_pressure_mode'
    return False, 'process_fanout_guard_active'


def _operator_mode_allows_core_sleeve_restart() -> Tuple[bool, str]:
    mode = os.getenv('SYSTEM_OPERATOR_MODE', '').strip().lower()
    if _env_flag('ALL_SLEEVES_PAUSED_BY_OPERATOR_MODE', '0') or _env_flag('LIVE_SLEEVES_PAUSED_BY_OPERATOR_MODE', '0'):
        return False, 'all_sleeves_explicitly_paused_by_operator_mode'
    if _env_flag('ALL_SLEEVES_PAUSED_FOR_COMPUTER_TASK', '0') or _env_flag('LIVE_SLEEVES_PAUSED_FOR_COMPUTER_TASK', '0'):
        return False, 'all_sleeves_explicitly_paused_for_computer_task'
    allow_readonly_restart = _env_flag('OPS_WATCHDOG_ALLOW_READONLY_SLEEVE_RESTART_DURING_DOWNSHIFT', '1')
    if _env_flag('TRAINING_RUNTIME_PAUSED_BY_OPERATOR_MODE', '0') or _env_flag('SHADOW_RESEARCH_PAUSED_BY_OPERATOR_MODE', '0'):
        if allow_readonly_restart:
            return True, 'training_or_research_paused_but_readonly_sleeve_restart_allowed'
        return False, 'operator_mode_paused_training_or_research'
    if _env_flag('TRAINING_RUNTIME_PAUSED_FOR_COMPUTER_TASK', '0') or _env_flag('SHADOW_RESEARCH_PAUSED_FOR_COMPUTER_TASK', '0'):
        if allow_readonly_restart:
            return True, 'computer_task_paused_training_or_research_but_readonly_sleeve_restart_allowed'
        return False, 'computer_task_paused_training_or_research'
    if mode == 'daily_driver' and not _env_flag('ROSTER_EXPANSION_ALLOWED', '0'):
        if allow_readonly_restart:
            return True, 'daily_driver_but_readonly_sleeve_restart_allowed'
        return False, 'operator_mode_daily_driver'
    if _env_flag('COMPUTER_NORMAL_USE_GOVERNOR_ACTIVE', '0') and not _env_flag('ROSTER_EXPANSION_ALLOWED', '0'):
        if allow_readonly_restart:
            return True, 'computer_normal_use_governor_but_readonly_sleeve_restart_allowed'
        return False, 'computer_normal_use_governor_active'
    if _env_flag('BACKLOG_INTAKE_GOVERNOR_ACTIVE', '0') and not _env_flag('ROSTER_EXPANSION_ALLOWED', '0'):
        if allow_readonly_restart:
            return True, 'backlog_intake_governor_but_readonly_sleeve_restart_allowed'
        return False, 'operator_mode_backlog_intake_governor'
    return True, 'operator_mode_allows_core_sleeve_restart'


def _all_sleeves_start_ready(broker: str, simulate: bool) -> Tuple[bool, str]:
    operator_ready, operator_reason = _operator_mode_allows_core_sleeve_restart()
    if not operator_ready:
        return False, operator_reason

    fanout_ready, fanout_reason = _fanout_guard_allows_core_sleeve_restart()
    if not fanout_ready:
        return False, fanout_reason

    missing = []
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_CORE', '')):
        missing.append('SHADOW_SYMBOLS_CORE')
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_VOLATILE', '')):
        missing.append('SHADOW_SYMBOLS_VOLATILE')
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_DEFENSIVE', '')) and not _split_csv(os.getenv('SHADOW_SYMBOLS_COMMOD_FX_INTL', '')):
        missing.append('SHADOW_SYMBOLS_DEFENSIVE_or_SHADOW_SYMBOLS_COMMOD_FX_INTL')
    if missing:
        return False, 'missing_symbol_env:' + ','.join(missing)

    if broker == 'schwab' and (not simulate):
        key = os.getenv('SCHWAB_API_KEY', '').strip()
        secret = os.getenv('SCHWAB_SECRET', '').strip()
        if _placeholder_or_empty(key) or _placeholder_or_empty(secret):
            return False, 'missing_schwab_credentials'

    if operator_reason != 'operator_mode_allows_core_sleeve_restart':
        return True, operator_reason
    return True, fanout_reason if fanout_reason != 'fanout_guard_inactive' else 'ready'


def _build_all_sleeves_target(heartbeat_max_age_seconds: int) -> Dict[str, Any]:
    broker = os.getenv('DATA_BROKER', 'schwab').strip().lower()
    if broker not in {'schwab', 'coinbase'}:
        broker = 'schwab'

    simulate = _env_flag('OPS_WATCHDOG_ALL_SLEEVES_SIMULATE', '0')
    with_aggressive = _env_flag('OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE', '1')

    cmd: List[str] = [str(PY), str(PROJECT_ROOT / 'scripts' / 'run_all_sleeves.py')]
    if with_aggressive:
        cmd.append('--with-aggressive-modes')
    cmd.extend(['--broker', broker])
    if simulate:
        cmd.append('--simulate')
    if _env_flag('OPS_WATCHDOG_ALL_SLEEVES_DISABLE_BREAKERS', '0'):
        cmd.append('--disable-circuit-breakers')

    arg_env = [
        ('--symbols-core', 'SHADOW_SYMBOLS_CORE'),
        ('--symbols-volatile', 'SHADOW_SYMBOLS_VOLATILE'),
        ('--symbols-defensive', 'SHADOW_SYMBOLS_DEFENSIVE'),
        ('--dividend-symbols', 'DIVIDEND_SYMBOLS'),
        ('--bond-symbols', 'BOND_SYMBOLS'),
    ]
    for arg, env_name in arg_env:
        val = os.getenv(env_name, '').strip()
        if val:
            cmd.extend([arg, val])

    child_fanout_grace_seconds = max(
        float(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_CHILD_GRACE_SECONDS', '180') or 180.0),
        60.0,
    )

    return {
        'name': 'all_sleeves',
        'pattern': 'scripts/run_all_sleeves.py',
        'alt_patterns': [
            'scripts/run_parallel_shadows.py',
            'scripts/run_dividend_shadow.py',
            'scripts/run_bond_shadow.py',
            'scripts/run_parallel_aggressive_modes.py',
            'scripts/run_shadow_training_loop.py --broker schwab',
        ],
        'parent_process_required': _env_flag('OPS_WATCHDOG_ALL_SLEEVES_PARENT_REQUIRED', '1'),
        'orphan_cleanup_patterns': [
            'scripts/run_parallel_shadows.py',
            'scripts/run_dividend_shadow.py',
            'scripts/run_bond_shadow.py',
            'scripts/run_fx_shadow.py',
            'scripts/run_parallel_aggressive_modes.py',
            'scripts/run_specialized_sleeve_shadow.py',
            'scripts/run_shadow_training_loop.py --broker schwab',
        ],
        'orphan_cleanup_grace_seconds': float(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_ORPHAN_CLEANUP_GRACE_SECONDS', '3') or 3),
        'min_child_processes': max(_safe_int(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_MIN_CHILDREN', '4'), 4), 0),
        'child_fanout_grace_seconds': child_fanout_grace_seconds,
        'heartbeat_startup_grace_seconds': max(
            float(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_HEARTBEAT_STARTUP_GRACE_SECONDS', str(int(child_fanout_grace_seconds))) or child_fanout_grace_seconds),
            child_fanout_grace_seconds,
        ),
        'launcher_health_path': str(PROJECT_ROOT / 'governance' / 'health' / 'all_sleeves_launcher_latest.json'),
        'repair_infrabots': [
            'sleeve_launcher_parent_watchdog',
            'sleeve_child_recycler',
            'sleeve_preflight_repair_guard',
            'sleeve_backpressure_guard',
            'sleeve_fanout_integrity_infrabot',
            'sleeve_restart_storm_circuit_infrabot',
            'sleeve_expansion_admission_infrabot',
            'sleeve_computer_coexistence_infrabot',
        ],
        'repair_commands': [
            [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'process_watchdog.py'), '--json'],
            ['./scripts/ops/opsctl.sh', 'post-restart-settle', '--apply', '--json'],
            ['./scripts/ops/opsctl.sh', 'storage-backpressure-autopilot', '--apply', '--json'],
        ],
        'repair_policy': 'restart_read_only_sleeve_collection_and_clean_orphans_without_enabling_live_execution',
        'restart_storm_impact': 'read_only_collection',
        'restart_storm_quarantine_allowed': True,
        'live_execution_critical': False,
        'exclude_patterns': _live_data_excludes(simulate),
        'cmd': cmd,
        'log': PROJECT_ROOT / 'logs' / 'watchdog_all_sleeves.log',
        'broker': broker,
        'simulate': simulate,
        'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_*_equities_schwab_*.json'),
        'heartbeat_max_age_seconds': max(int(heartbeat_max_age_seconds), 60),
        'restart_storm_settle_seconds': max(
            int(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_RESTART_STORM_SETTLE_SECONDS', '180') or '180'),
            60,
        ),
        'restart_storm_min_healthy_seconds': max(
            int(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_MIN_HEALTHY_SECONDS', '90') or '90'),
            60,
        ),
    }


def _build_execution_lane_target(mode: str, *, heartbeat_max_age_seconds: int) -> Dict[str, Any]:
    safe_mode = str(mode or 'paper').strip().lower() or 'paper'
    settle_seconds_env = (
        os.getenv('OPS_WATCHDOG_PAPER_EXECUTOR_RESTART_STORM_SETTLE_SECONDS', '120')
        if safe_mode == 'paper'
        else os.getenv('OPS_WATCHDOG_EXECUTION_RESTART_STORM_SETTLE_SECONDS', '300')
    )
    min_healthy_seconds_env = (
        os.getenv('OPS_WATCHDOG_PAPER_EXECUTOR_MIN_HEALTHY_SECONDS', '120')
        if safe_mode == 'paper'
        else os.getenv('OPS_WATCHDOG_EXECUTION_MIN_HEALTHY_SECONDS', '180')
    )
    return {
        'name': f'execution_lane_{safe_mode}',
        'pattern': f'scripts/run_execution_lane.py --mode {safe_mode}',
        'alt_patterns': [],
        'cmd': [
            str(PY),
            str(PROJECT_ROOT / 'scripts' / 'run_execution_lane.py'),
            '--mode',
            safe_mode,
        ],
        'log': PROJECT_ROOT / 'logs' / f'watchdog_execution_lane_{safe_mode}.log',
        'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / f'execution_lane_{safe_mode}_latest.json'),
        'heartbeat_max_age_seconds': max(int(heartbeat_max_age_seconds), 60),
        'heartbeat_startup_grace_seconds': max(int(heartbeat_max_age_seconds), 60),
        'max_running': 1,
        'duplicate_cleanup_grace_seconds': max(
            float(os.getenv('OPS_WATCHDOG_EXECUTION_DUPLICATE_CLEANUP_GRACE_SECONDS', '1.0') or 1.0),
            0.2,
        ),
        'restart_storm_settle_seconds': max(int(settle_seconds_env or '120'), 60),
        'restart_storm_min_healthy_seconds': max(int(min_healthy_seconds_env or '120'), 60),
        'restart_storm_impact': 'execution_lane',
        'restart_storm_quarantine_allowed': False,
        'live_execution_critical': True,
    }


def _probe_host(hostport: str, timeout_seconds: float) -> Dict[str, Any]:
    raw = (hostport or '').strip()
    if not raw:
        return {'hostport': '', 'ok': False, 'error': 'empty'}

    host = raw
    port = 443
    if ':' in raw:
        h, p = raw.rsplit(':', 1)
        host = h.strip()
        try:
            port = int(p.strip())
        except Exception:
            port = 443
    if not host:
        return {'hostport': raw, 'ok': False, 'error': 'empty_host'}

    try:
        with socket.create_connection((host, port), timeout=max(float(timeout_seconds), 0.2)):
            return {'hostport': f'{host}:{port}', 'ok': True}
    except Exception as exc:
        return {'hostport': f'{host}:{port}', 'ok': False, 'error': f'{type(exc).__name__}:{exc}'}


def _storage_failback_sync_cmd() -> List[str]:
    return [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'storage_failback_sync.py'), '--json']


def _resolve_external_storage_paths() -> Tuple[Path, Path]:
    resolution = resolve_external_storage()
    return resolution.mount_root, resolution.external_root


def _external_min_free_bytes() -> int:
    raw_bytes = os.getenv('BOT_LOGS_EXTERNAL_MIN_FREE_BYTES', '').strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0

    raw_gb = os.getenv('BOT_LOGS_EXTERNAL_MIN_FREE_GB', '').strip()
    if raw_gb:
        try:
            return max(int(float(raw_gb) * (1024 ** 3)), 0)
        except Exception:
            return 0

    return 0


def _disk_free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(path).free)
    except Exception:
        return None


def _probe_storage_mount() -> Dict[str, Any]:
    if not _hot_storage_prefers_external():
        mount_root = Path(os.getenv('BOT_LOGS_EXTERNAL_MOUNT', '/Volumes/BOT_LOGS')).expanduser()
        configured_root = str(os.getenv('BOT_LOGS_EXTERNAL_PROJECT_ROOT', '') or '').strip()
        project_dir = str(os.getenv('BOT_LOGS_EXTERNAL_PROJECT_DIR', 'schwab_trading_bot') or 'schwab_trading_bot').strip()
        external_root = Path(configured_root).expanduser() if configured_root else mount_root / project_dir
        return {
            'mount_root': str(mount_root),
            'external_root': str(external_root),
            'configured_mount_root': str(mount_root),
            'configured_project_root': str(external_root),
            'candidate_mount_roots': [str(mount_root)],
            'matched_mount_root': '',
            'match_reason': 'external_io_probe_skipped_local_hot_storage_policy',
            'target_volume_device_identifier': str(os.getenv('BOT_LOGS_EXTERNAL_DISK_IDENTIFIER', '') or ''),
            'target_volume_name': str(os.getenv('BOT_LOGS_EXTERNAL_VOLUME_NAME', 'BOT_LOGS') or 'BOT_LOGS'),
            'target_volume_uuid': str(os.getenv('BOT_LOGS_EXTERNAL_VOLUME_UUID', '') or ''),
            'target_volume_mount_point': '',
            'target_volume_present': False,
            'target_volume_mounted': False,
            'mount_present': False,
            'external_root_exists': False,
            'external_root_writable': False,
            'external_free_bytes': None,
            'external_min_free_bytes': int(_external_min_free_bytes()),
            'external_low_space': False,
            'external_unavailable_reason': 'cold_archive_only_local_hot_storage_policy',
            'external_available': False,
            'external_required_for_hot_path': False,
            'hot_storage_available': True,
            'probe_skipped_external_io': True,
            'storage_mode': 'local_fallback',
        }
    resolution = resolve_external_storage()
    mount_root, external_root = resolution.mount_root, resolution.external_root
    mount_present = bool(mount_root.exists() and mount_root.is_dir())
    target_volume = find_target_external_volume() if not mount_present else None
    external_root_exists = bool(external_root.exists() and external_root.is_dir())
    external_root_writable = bool(external_root_exists and os.access(external_root, os.W_OK))
    probe_root = external_root if external_root_exists else mount_root
    external_free_bytes = _disk_free_bytes(probe_root) if mount_present else None
    external_min_free_bytes = _external_min_free_bytes()
    external_low_space = bool(
        external_root_exists
        and external_root_writable
        and external_min_free_bytes > 0
        and external_free_bytes is not None
        and external_free_bytes < external_min_free_bytes
    )

    if not mount_present:
        if target_volume is not None and not target_volume.is_mounted:
            unavailable_reason = 'volume_unmounted'
        else:
            unavailable_reason = 'mount_missing'
    elif target_volume is not None and not target_volume.is_mounted:
        unavailable_reason = 'volume_unmounted'
    elif not external_root_exists:
        unavailable_reason = 'root_missing'
    elif not external_root_writable:
        unavailable_reason = 'not_writable'
    elif external_low_space:
        unavailable_reason = 'low_space'
    else:
        unavailable_reason = 'ok'

    external_available = bool(mount_present and external_root_exists and external_root_writable and not external_low_space)
    return {
        'mount_root': str(mount_root),
        'external_root': str(external_root),
        'configured_mount_root': str(resolution.configured_mount_root),
        'configured_project_root': str(resolution.configured_project_root) if resolution.configured_project_root else '',
        'candidate_mount_roots': [str(path) for path in resolution.candidate_mount_roots],
        'matched_mount_root': str(resolution.matched_mount_root) if resolution.matched_mount_root else '',
        'match_reason': str(resolution.match_reason),
        'target_volume_device_identifier': str(target_volume.device_identifier) if target_volume else '',
        'target_volume_name': str(target_volume.volume_name) if target_volume else '',
        'target_volume_uuid': str(target_volume.volume_uuid) if target_volume else '',
        'target_volume_mount_point': str(target_volume.mount_point) if target_volume else '',
        'target_volume_present': bool(target_volume is not None),
        'target_volume_mounted': bool(target_volume.is_mounted) if target_volume else False,
        'mount_present': mount_present,
        'external_root_exists': external_root_exists,
        'external_root_writable': external_root_writable,
        'external_free_bytes': external_free_bytes,
        'external_min_free_bytes': int(external_min_free_bytes),
        'external_low_space': external_low_space,
        'external_unavailable_reason': unavailable_reason,
        'external_available': external_available,
    }


def _evaluate_storage_mount_transition(previous_mount_present: Any, mount_present_now: bool) -> Dict[str, Any]:
    if previous_mount_present is None:
        if mount_present_now:
            return {}
        return {'from': 'unknown', 'to': False}

    prev = bool(previous_mount_present)
    if prev == mount_present_now:
        return {}
    return {'from': prev, 'to': mount_present_now}


def _kickstart_labels(labels: List[str]) -> List[Dict[str, Any]]:
    uid = os.getuid()
    actions: List[Dict[str, Any]] = []
    for label in labels:
        full = f'gui/{uid}/{label}'
        rc, out, err = _run(['launchctl', 'kickstart', '-k', full])
        actions.append({'label': label, 'rc': int(rc), 'stdout': out[-200:], 'stderr': err[-200:]})
    return actions


def _alert(severity: str, event: str, message: str, suppress_seconds: int = 600, force: bool = False) -> Dict[str, Any]:
    if not ALERT_ROUTER.exists() or not PY.exists():
        return {'attempted': False, 'reason': 'alert_router_missing'}

    cmd = [
        str(PY),
        str(ALERT_ROUTER),
        '--severity',
        severity,
        '--event',
        event,
        '--message',
        message,
        '--suppress-seconds',
        str(max(int(suppress_seconds), 0)),
    ]
    if force:
        cmd.append('--force')
    rc, out, err = _run(cmd)
    return {
        'attempted': True,
        'rc': int(rc),
        'stdout': out[-500:],
        'stderr': err[-500:],
    }


def _storage_mode_transition_alert(previous_mode: str, current_mode: str, *, suppress_seconds: int) -> Dict[str, Any] | None:
    if current_mode in {'local_fallback', 'local_fallback_split_brain'}:
        message = 'External BOT_LOGS unavailable or not writable. Switched to local fallback storage.'
        if current_mode == 'local_fallback_split_brain':
            message = 'External BOT_LOGS available, but failback is blocked by divergent local fallback data. Remaining on local fallback storage.'
        return _alert(
            'critical',
            'storage_fallback_activated',
            message,
            suppress_seconds=max(suppress_seconds, 60),
        )

    if previous_mode in {'local_fallback', 'local_fallback_split_brain'} and current_mode in {'external', 'external_curated'}:
        return _alert(
            'info',
            'storage_external_restored',
            'External BOT_LOGS restored. Storage routing back on external root.',
            suppress_seconds=max(suppress_seconds, 60),
        )

    return None


def main() -> int:
    _bootstrap_runtime_env(os.getenv('BOT_RUNTIME_PROFILE', 'live'))
    parser = argparse.ArgumentParser(description='Watchdog: restart key loops with bounded backoff.')
    parser.add_argument('--max-restarts-per-hour', type=int, default=int(os.getenv('OPS_WATCHDOG_MAX_RESTARTS_PER_HOUR', '6')))
    parser.add_argument('--require-all-sleeves', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_ALL_SLEEVES', '1') == '1')
    parser.add_argument('--require-coinbase', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_COINBASE', '1') == '1')
    parser.add_argument('--require-coinbase-futures', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_COINBASE_FUTURES', '1') == '1')
    parser.add_argument('--require-paper-executor', action='store_true', default=_default_require_paper_executor())
    parser.add_argument('--refresh-reports', action='store_true', default=os.getenv('OPS_WATCHDOG_REFRESH_REPORTS', '1') == '1')
    parser.add_argument('--refresh-max-age-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS', '7200')))
    parser.add_argument('--all-sleeves-heartbeat-stale-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_HEARTBEAT_STALE_SECONDS', '360')))
    parser.add_argument('--coinbase-heartbeat-stale-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_COINBASE_HEARTBEAT_STALE_SECONDS', '420')))
    parser.add_argument('--paper-executor-heartbeat-stale-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_PAPER_EXECUTOR_HEARTBEAT_STALE_SECONDS', '240')))
    parser.add_argument('--network-guard', action='store_true', default=_env_flag('OPS_WATCHDOG_NETWORK_GUARD', '1'))
    parser.add_argument('--network-hosts', default=os.getenv('OPS_WATCHDOG_NETWORK_HOSTS', 'api.schwabapi.com:443,api.exchange.coinbase.com:443'))
    parser.add_argument('--network-timeout-seconds', type=float, default=float(os.getenv('OPS_WATCHDOG_NETWORK_TIMEOUT_SECONDS', '2.5')))
    parser.add_argument('--network-fail-threshold', type=int, default=int(os.getenv('OPS_WATCHDOG_NETWORK_FAIL_THRESHOLD', '3')))
    parser.add_argument('--restart-storm-threshold', type=int, default=int(os.getenv('OPS_WATCHDOG_RESTART_STORM_THRESHOLD', '4')))
    parser.add_argument('--restart-storm-window-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_RESTART_STORM_WINDOW_SECONDS', '3600')))
    parser.add_argument('--restart-storm-settle-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_RESTART_STORM_SETTLE_SECONDS', '900')))
    parser.add_argument('--alert-suppress-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_ALERT_SUPPRESS_SECONDS', '600')))
    parser.add_argument('--maintenance-timeout-seconds', type=float, default=DEFAULT_MAINTENANCE_TIMEOUT_SECONDS)
    parser.add_argument('--singleton-lock-path', default=str(DEFAULT_SINGLETON_LOCK_PATH))
    parser.add_argument('--readonly-repair-probe-cooldown-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_READONLY_REPAIR_PROBE_COOLDOWN_SECONDS', '900')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    singleton_handle, singleton_owner = _acquire_singleton_lock(Path(args.singleton_lock_path).expanduser())
    if singleton_handle is None:
        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'ready',
            'skipped': True,
            'reason': 'process_watchdog_already_running',
            'singleton_lock_path': str(Path(args.singleton_lock_path).expanduser()),
            'singleton_owner': singleton_owner,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"process_watchdog skipped=already_running owner={singleton_owner or 'unknown'}")
        return 0

    state = _load_state(DEFAULT_STATE_PATH, FALLBACK_STATE_PATH)
    events = state.get('events') if isinstance(state.get('events'), list) else []

    maintenance: List[Dict[str, Any]] = []
    hot_storage_prefers_external = _hot_storage_prefers_external()
    storage_mode = str(state.get('storage_mode', '') or '')
    if not hot_storage_prefers_external:
        storage_mode = 'local_fallback'
    storage_mode_transition: Dict[str, Any] = {}
    storage_mount_prev_raw = state.get('storage_mount_present', None)
    storage_mount_prev = None if storage_mount_prev_raw is None else bool(storage_mount_prev_raw)
    storage_mount_transition: Dict[str, Any] = {}
    storage_mount_guard: Dict[str, Any] = {}

    maintenance_jobs = [
        ('lock_watchdog', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'lock_watchdog.py'), '--apply', '--json']),
        ('canary_auto_tuner', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'canary_auto_tuner.py'), '--json']),
    ]
    if hot_storage_prefers_external:
        maintenance_jobs.insert(1, ('storage_failback_sync', _storage_failback_sync_cmd()))
    else:
        maintenance.append(
            {
                'name': 'storage_failback_sync',
                'ok': True,
                'rc': 0,
                'skipped': True,
                'reason': 'cold_archive_only_local_hot_storage_policy',
                'stdout_tail': '',
                'stderr_tail': '',
            }
        )

    for name, cmd in maintenance_jobs:
        rc, out, err = _run(cmd, timeout_seconds=float(args.maintenance_timeout_seconds))
        row: Dict[str, Any] = {
            'name': name,
            'ok': rc == 0,
            'rc': int(rc),
            'stdout_tail': '\n'.join((out or '').splitlines()[-6:]),
            'stderr_tail': '\n'.join((err or '').splitlines()[-6:]),
        }

        if name == 'storage_failback_sync':
            try:
                payload = json.loads(out) if out else {}
            except Exception:
                payload = {}
            row['payload'] = payload
            mode_now = str(payload.get('mode', '') or '')
            if mode_now and mode_now != storage_mode:
                storage_mode_transition = {
                    'from': storage_mode or 'unknown',
                    'to': mode_now,
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                }
                alert = _storage_mode_transition_alert(
                    storage_mode,
                    mode_now,
                    suppress_seconds=args.alert_suppress_seconds,
                )
                if alert is not None:
                    storage_mode_transition['alert'] = alert
                storage_mode = mode_now
            elif mode_now:
                storage_mode = mode_now

        maintenance.append(row)

    storage_mount_guard = _probe_storage_mount()
    storage_mount_present = bool(storage_mount_guard.get('mount_present', False))
    storage_mount_guard['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    storage_mount_guard['storage_mode'] = storage_mode or 'unknown'
    storage_mount_guard['previous_mount_present'] = storage_mount_prev

    mount_transition_base = (
        {}
        if bool(storage_mount_guard.get('probe_skipped_external_io', False))
        else _evaluate_storage_mount_transition(storage_mount_prev, storage_mount_present)
    )
    if mount_transition_base:
        storage_mount_transition = {
            **mount_transition_base,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'mount_root': storage_mount_guard.get('mount_root'),
            'external_root': storage_mount_guard.get('external_root'),
        }

        if not storage_mount_present:
            unavailable_reason = str(storage_mount_guard.get('external_unavailable_reason') or '')
            if unavailable_reason == 'volume_unmounted':
                alert_message = (
                    f"External BOT_LOGS volume is present but not mounted at {storage_mount_guard.get('mount_root')}."
                )
            else:
                alert_message = f"External BOT_LOGS mount missing at {storage_mount_guard.get('mount_root')}."
            storage_mount_transition['alert'] = _alert(
                'critical',
                'storage_external_mount_missing',
                alert_message,
                suppress_seconds=max(args.alert_suppress_seconds, 60),
            )
        else:
            recovery: Dict[str, Any] = {'attempted': False}
            if storage_mode != 'external':
                rc, out, err = _run(
                    _storage_failback_sync_cmd(),
                    timeout_seconds=float(args.maintenance_timeout_seconds),
                )
                try:
                    recovery_payload = json.loads(out) if out else {}
                except Exception:
                    recovery_payload = {}
                recovery = {
                    'attempted': True,
                    'ok': rc == 0,
                    'rc': int(rc),
                    'payload': recovery_payload,
                    'stdout_tail': '\n'.join((out or '').splitlines()[-6:]),
                    'stderr_tail': '\n'.join((err or '').splitlines()[-6:]),
                }
                maintenance.append(
                    {
                        'name': 'storage_failback_sync_recovery',
                        'ok': rc == 0,
                        'rc': int(rc),
                        'payload': recovery_payload,
                        'stdout_tail': recovery['stdout_tail'],
                        'stderr_tail': recovery['stderr_tail'],
                    }
                )
                mode_now = str(recovery_payload.get('mode', '') or '')
                if mode_now:
                    storage_mode = mode_now
            storage_mount_transition['recovery'] = recovery
            if not storage_mode_transition:
                storage_mount_transition['alert'] = _alert(
                    'info',
                    'storage_external_mount_restored',
                    f"External BOT_LOGS mount restored at {storage_mount_guard.get('mount_root')}.",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )

    refresh_payload: Dict[str, Any] = {}
    health_fast_refresh_age = max(int(os.getenv('OPS_WATCHDOG_HEALTH_FAST_MAX_AGE_SECONDS', '300')), 60)
    if args.refresh_reports:
        refresh_payload = _refresh_runtime_reports(
            max_age_seconds=max(int(args.refresh_max_age_seconds), 60),
            backpressure_max_age_seconds=max(int(os.getenv('OPS_WATCHDOG_BACKPRESSURE_MAX_AGE_SECONDS', '300')), 60),
            divergence_max_age_seconds=max(int(os.getenv('OPS_WATCHDOG_DIVERGENCE_MAX_AGE_SECONDS', '600')), 60),
            health_fast_max_age_seconds=health_fast_refresh_age,
        )
    else:
        refresh_payload = {
            'heavy_reports_enabled': False,
            'health_fast': _refresh_health_fast(health_fast_refresh_age),
        }

    network_payload: Dict[str, Any] = {'enabled': bool(args.network_guard), 'results': []}
    network_outage_active = False
    network_fail_streak_prev = int(state.get('network_fail_streak', 0) or 0)
    network_fail_streak_now = network_fail_streak_prev

    if args.network_guard:
        hosts = _split_csv(args.network_hosts)
        results = [_probe_host(h, timeout_seconds=float(args.network_timeout_seconds)) for h in hosts]
        any_ok = any(r.get('ok') for r in results) if results else True

        if any_ok:
            network_fail_streak_now = 0
            if network_fail_streak_prev >= max(int(args.network_fail_threshold), 1):
                kickstarts = _kickstart_labels([
                    'com.dankingsley.shadow_watchdog',
                    'com.dankingsley.all_sleeves',
                ])
                recovered_alert = _alert(
                    'warn',
                    'network_recovered_restart',
                    f'Network recovered after fail_streak={network_fail_streak_prev}; kickstarted core services.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )
                network_payload['recovery_actions'] = {
                    'kickstarts': kickstarts,
                    'alert': recovered_alert,
                }
        else:
            network_fail_streak_now = network_fail_streak_prev + 1
            if network_fail_streak_now == max(int(args.network_fail_threshold), 1):
                network_payload['degraded_alert'] = _alert(
                    'critical',
                    'network_outage_detected',
                    f'Network probe failed for {network_fail_streak_now} consecutive checks.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )

        network_outage_active = network_fail_streak_now >= max(int(args.network_fail_threshold), 1) and (not any_ok)
        network_payload.update(
            {
                'hosts': hosts,
                'results': results,
                'any_ok': bool(any_ok),
                'fail_streak_prev': int(network_fail_streak_prev),
                'fail_streak_now': int(network_fail_streak_now),
                'outage_active': bool(network_outage_active),
            }
        )

    targets: List[Dict[str, Any]] = [
        {
            'name': 'sql_link_writer',
            'pattern': 'scripts/ops/sql_link_shard_manager.py',
            'cmd': [str(PROJECT_ROOT / 'scripts' / 'ops' / 'run_sql_link_writer_launchd.sh')],
            'log': PROJECT_ROOT / 'logs' / 'watchdog_sql_link_writer.log',
            'alt_patterns': ['scripts/ops/sql_link_writer_service.py'],
            'heartbeat_glob': '',
            'heartbeat_max_age_seconds': 0,
            'restart_storm_impact': 'storage_writer',
            'restart_storm_quarantine_allowed': False,
            'live_execution_critical': False,
        },
    ]

    if args.require_all_sleeves:
        targets.append(_build_all_sleeves_target(heartbeat_max_age_seconds=args.all_sleeves_heartbeat_stale_seconds))

    if args.require_paper_executor:
        targets.append(
            _build_execution_lane_target(
                'paper',
                heartbeat_max_age_seconds=args.paper_executor_heartbeat_stale_seconds,
            )
        )

    if args.require_coinbase:
        coinbase_cmd: List[str] = [
            str(PY),
            str(PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'),
            '--broker',
            'coinbase',
            '--symbols',
            os.getenv('COINBASE_WATCH_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD'),
            '--interval-seconds',
            os.getenv('COINBASE_WATCH_INTERVAL_SECONDS', '20'),
            '--max-iterations',
            '0',
        ]
        coinbase_simulate = _env_flag('OPS_WATCHDOG_COINBASE_SIMULATE', '0')
        if coinbase_simulate:
            coinbase_cmd.append('--simulate')
        targets.append(
            {
                'name': 'coinbase_loop',
                'pattern': 'scripts/run_shadow_training_loop.py --broker coinbase',
                'exclude_patterns': _live_data_excludes(coinbase_simulate, ['--profile crypto_futures']),
                'cmd': coinbase_cmd,
                'log': PROJECT_ROOT / 'logs' / 'watchdog_coinbase_loop.log',
                'alt_patterns': [],
                'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_default_crypto_coinbase_*.json'),
                'heartbeat_max_age_seconds': max(int(args.coinbase_heartbeat_stale_seconds), 60),
                'max_running': 1,
                'duplicate_cleanup_grace_seconds': max(
                    float(os.getenv('OPS_WATCHDOG_COINBASE_DUPLICATE_CLEANUP_GRACE_SECONDS', '1.0') or 1.0),
                    0.2,
                ),
                'restart_storm_impact': 'read_only_collection',
                'restart_storm_quarantine_allowed': True,
                'live_execution_critical': False,
            }
        )

    if args.require_coinbase_futures:
        futures_profile = os.getenv('COINBASE_FUTURES_PROFILE', 'crypto_futures')
        coinbase_futures_cmd: List[str] = [
            str(PY),
            str(PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'),
            '--broker',
            'coinbase',
            '--profile',
            futures_profile,
            '--domain',
            'crypto',
            '--symbols',
            os.getenv('COINBASE_FUTURES_WATCH_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD'),
            '--context-symbols',
            os.getenv('COINBASE_FUTURES_CONTEXT_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD'),
            '--interval-seconds',
            os.getenv('COINBASE_FUTURES_WATCH_INTERVAL_SECONDS', '20'),
            '--max-iterations',
            '0',
        ]
        coinbase_futures_simulate = _env_flag('OPS_WATCHDOG_COINBASE_FUTURES_SIMULATE', '0')
        if coinbase_futures_simulate:
            coinbase_futures_cmd.append('--simulate')
        targets.append(
            {
                'name': 'coinbase_futures_loop',
                'pattern': f'scripts/run_shadow_training_loop.py --broker coinbase --profile {futures_profile}',
                'exclude_patterns': _live_data_excludes(coinbase_futures_simulate),
                'cmd': coinbase_futures_cmd,
                'log': PROJECT_ROOT / 'logs' / 'watchdog_coinbase_futures_loop.log',
                'alt_patterns': [],
                'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_*crypto_futures*_crypto_coinbase_*.json'),
                'heartbeat_max_age_seconds': max(int(args.coinbase_heartbeat_stale_seconds), 60),
                'max_running': 1,
                'duplicate_cleanup_grace_seconds': max(
                    float(os.getenv('OPS_WATCHDOG_COINBASE_DUPLICATE_CLEANUP_GRACE_SECONDS', '1.0') or 1.0),
                    0.2,
                ),
                'restart_storm_impact': 'read_only_collection',
                'restart_storm_quarantine_allowed': True,
                'live_execution_critical': False,
            }
        )

    restarts: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []
    safety_pause = _safety_pause_state()
    creative_pause = _creative_cotenant_pause_state()
    safety_pause_target_names = {str(t.get('name') or '') for t in targets if str(t.get('name') or '') != 'sql_link_writer'}

    for t in targets:
        running = _proc_running(t['pattern'], exclude_patterns=t.get('exclude_patterns', []))
        alt_running = sum(
            _proc_running(p, exclude_patterns=t.get('exclude_patterns', []))
            for p in t.get('alt_patterns', [])
            if p
        )
        duplicate_cleanup: Dict[str, Any] = {}
        max_running = _safe_int(t.get('max_running'), 0)
        matched_pid_count = 0
        if max_running > 0:
            matched_pid_count = len(_matching_pids(str(t['pattern']), exclude_patterns=t.get('exclude_patterns', [])))
        if max_running > 0 and matched_pid_count > max_running:
            duplicate_cleanup = _trim_duplicate_processes(
                str(t['pattern']),
                max_running=max_running,
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('duplicate_cleanup_grace_seconds', 1.0) or 1.0),
            )
            running = _proc_running(t['pattern'], exclude_patterns=t.get('exclude_patterns', []))
        heartbeat_glob = str(t.get('heartbeat_glob', '') or '')
        heartbeat_required = bool(heartbeat_glob)
        heartbeat_age = _latest_heartbeat_age_seconds(heartbeat_glob) if heartbeat_required else 0.0
        heartbeat_max_age = float(t.get('heartbeat_max_age_seconds', 0) or 0.0)
        heartbeat_fresh = (not heartbeat_required) or (heartbeat_age <= heartbeat_max_age)
        parent_process_required = bool(t.get('parent_process_required', False))
        child_process_live = alt_running > 0
        process_elapsed_seconds = (
            _proc_elapsed_seconds(t['pattern'], exclude_patterns=t.get('exclude_patterns', []))
            if running > 0
            else None
        )
        parent_elapsed_seconds = process_elapsed_seconds if parent_process_required else None
        child_fanout = _child_fanout_health(
            t,
            running=int(running),
            alt_running=int(alt_running),
            parent_elapsed_seconds=parent_elapsed_seconds,
        )
        child_fanout_ok = bool(child_fanout.get('ok', True))
        launcher_artifact_health: Dict[str, Any] = {}
        launcher_artifact_certified_fanout = False
        if t.get('name') == 'all_sleeves':
            launcher_artifact_health = _all_sleeves_launcher_artifact_health(t)
            launcher_artifact_certified_fanout = bool(
                parent_process_required
                and child_process_live
                and child_fanout_ok
                and launcher_artifact_health.get('ok', False)
            )
        process_live = (running > 0) if parent_process_required else ((running > 0) or child_process_live)
        if launcher_artifact_certified_fanout:
            process_live = True
            heartbeat_fresh = True
        heartbeat_startup_grace_seconds = max(float(t.get('heartbeat_startup_grace_seconds', 0.0) or 0.0), 0.0)
        heartbeat_startup_grace_active = bool(
            heartbeat_required
            and not heartbeat_fresh
            and process_live
            and heartbeat_startup_grace_seconds > 0.0
            and process_elapsed_seconds is not None
            and float(process_elapsed_seconds) < heartbeat_startup_grace_seconds
        )
        if heartbeat_startup_grace_active:
            heartbeat_fresh = True
        heartbeat_ok = heartbeat_fresh and (process_live or not heartbeat_required) and child_fanout_ok

        row: Dict[str, Any] = {
            'name': t['name'],
            'running': int(running),
            'heartbeat_ok': bool(heartbeat_ok),
            'process_live': bool(process_live),
        }
        if launcher_artifact_health:
            row['launcher_artifact_health'] = launcher_artifact_health
        if launcher_artifact_certified_fanout:
            row['effective_process_live'] = True
            row['launcher_artifact_certified_fanout'] = True
            row['process_live_reason'] = str(launcher_artifact_health.get('reason') or 'launcher_artifact_certified_fanout')
        if duplicate_cleanup:
            row['duplicate_cleanup'] = duplicate_cleanup
        if process_elapsed_seconds is not None:
            row['process_elapsed_seconds'] = round(float(process_elapsed_seconds), 2)
        for key in ('restart_storm_impact', 'restart_storm_quarantine_allowed', 'live_execution_critical'):
            if key in t:
                row[key] = t[key]
        if t.get('repair_infrabots'):
            row['repair_infrabots'] = list(t.get('repair_infrabots') or [])
            row['repair_policy'] = str(t.get('repair_policy') or '')
            if not heartbeat_ok:
                row['repair_commands'] = list(t.get('repair_commands') or [])
        if parent_process_required:
            row['parent_process_required'] = True
            row['launcher_live'] = bool(running > 0)
            row['child_process_live'] = bool(child_process_live)
            row['child_fanout_ok'] = bool(child_fanout_ok)
            row['child_fanout'] = child_fanout
            if parent_elapsed_seconds is not None:
                row['parent_elapsed_seconds'] = round(float(parent_elapsed_seconds), 2)
        if 'restart_storm_settle_seconds' in t:
            row['restart_storm_settle_seconds'] = int(t['restart_storm_settle_seconds'])
        if 'restart_storm_min_healthy_seconds' in t:
            row['restart_storm_min_healthy_seconds'] = int(t['restart_storm_min_healthy_seconds'])
        if alt_running > 0:
            row['alt_running'] = int(alt_running)
        if heartbeat_required:
            row['heartbeat_age_seconds'] = round(float(heartbeat_age), 2)
            row['heartbeat_max_age_seconds'] = float(heartbeat_max_age)
            row['heartbeat_fresh'] = bool(heartbeat_fresh)
            if heartbeat_startup_grace_active:
                row['heartbeat_startup_grace_active'] = True
                row['heartbeat_startup_grace_seconds'] = round(float(heartbeat_startup_grace_seconds), 2)
                row['heartbeat_fresh_reason'] = 'startup_grace'

        if duplicate_cleanup and duplicate_cleanup.get('still_running_pids'):
            row['restart_skipped'] = 'duplicate_cleanup_pending'
            row['reason'] = 'single_instance_duplicate_cleanup_pending'
            status.append(row)
            continue

        if t['name'] == 'sql_link_writer' and not process_live:
            writer_idle_health = _sql_link_writer_idle_health()
            row['writer_idle_health'] = writer_idle_health
            if bool(writer_idle_health.get('ok', False)):
                process_live = True
                heartbeat_ok = True
                row['process_live'] = True
                row['heartbeat_ok'] = True
                row['writer_idle_ok'] = True
                row['virtual_process_live'] = True
                row['process_live_reason'] = str(writer_idle_health.get('reason') or 'sql_writer_on_demand_idle_complete')
            else:
                writer_recovery_health = _sql_link_writer_recovery_health()
                row['writer_recovery_health'] = writer_recovery_health
                if bool(writer_recovery_health.get('ok', False)):
                    process_live = True
                    heartbeat_ok = True
                    row['process_live'] = True
                    row['heartbeat_ok'] = True
                    row['writer_recovered_ok'] = True
                    row['virtual_process_live'] = True
                    row['process_live_reason'] = str(
                        writer_recovery_health.get('reason') or 'sql_writer_active_progress_recovered'
                    )

        if t['name'] == 'sql_link_writer' and process_live and heartbeat_ok:
            writer_recovery_health = row.get('writer_recovery_health')
            if not isinstance(writer_recovery_health, dict):
                writer_recovery_health = _sql_link_writer_recovery_health()
            row['writer_recovery_health'] = writer_recovery_health
            if bool(writer_recovery_health.get('ok', False)):
                row['writer_recovered_ok'] = True
                row['process_live_reason'] = str(writer_recovery_health.get('reason') or 'sql_writer_active_progress_recovered')

        if safety_pause['active'] and t['name'] in safety_pause_target_names:
            row['paused_by_safety_flags'] = True
            row['safety_pause_reason'] = str(safety_pause.get('reason') or 'safety_pause_active')
            row['operator_stop_active'] = bool(safety_pause.get('operator_stop_active', False))
            row['global_halt_active'] = bool(safety_pause.get('global_halt_active', False))
            row['restart_skipped'] = 'paused_by_safety_flags'
            status.append(row)
            continue

        if _creative_pause_suppresses_target(str(t.get('name') or ''), creative_pause):
            row['paused_by_creative_cotenant_guard'] = True
            row['creative_pause_reason'] = str(creative_pause.get('reason') or 'creative_cotenant_pause_active')
            row['creative_session_level'] = str(creative_pause.get('creative_session_level') or '')
            row['creative_session_kind'] = str(creative_pause.get('creative_session_kind') or '')
            row['restart_skipped'] = 'creative_cotenant_pause_active'
            row['reason'] = str(creative_pause.get('reason') or 'creative_cotenant_pause_active')
            status.append(row)
            continue

        if t['name'] == 'execution_lane_paper':
            runtime_pause = _paper_execution_runtime_pause_state()
            row['runtime_paper_pause_state'] = runtime_pause
            if bool(runtime_pause.get('paused', False)):
                row['paused_by_runtime_gate'] = True
                row['runtime_pause_reason'] = str(runtime_pause.get('reason') or 'runtime_paper_execution_paused')
                row['restart_skipped'] = 'runtime_paper_execution_paused'
                row['reason'] = row['runtime_pause_reason']
                status.append(row)
                continue

        if process_live and heartbeat_ok:
            status.append(row)
            continue

        if t['name'] == 'all_sleeves' and parent_process_required and running <= 0 and alt_running > 0:
            cleanup_patterns = t.get('orphan_cleanup_patterns') if isinstance(t.get('orphan_cleanup_patterns'), list) else []
            cleanup = _terminate_matching_processes(
                [str(pattern) for pattern in cleanup_patterns if str(pattern or '').strip()],
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('orphan_cleanup_grace_seconds', 3.0) or 3.0),
            )
            row['orphan_cleanup'] = cleanup
            row['alt_running_after_cleanup'] = sum(
                _proc_running(p, exclude_patterns=t.get('exclude_patterns', []))
                for p in t.get('alt_patterns', [])
                if p
            )

        if t['name'] == 'all_sleeves':
            ready, reason = _all_sleeves_start_ready(str(t.get('broker', 'schwab')), bool(t.get('simulate', False)))
            if not ready:
                row['restart_skipped'] = 'startup_not_ready'
                row['reason'] = reason
                status.append(row)
                continue

        if network_outage_active and t['name'] in {'all_sleeves', 'coinbase_loop'}:
            row['restart_skipped'] = 'network_outage_active'
            status.append(row)
            continue

        stale_process_cleanup: Dict[str, Any] = {}
        if max_running > 0 and running <= 0:
            stale_process_cleanup = _terminate_matching_processes(
                [str(t['pattern'])],
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('duplicate_cleanup_grace_seconds', 1.0) or 1.0),
            )
            if stale_process_cleanup.get('matched_pids'):
                row['stale_process_cleanup'] = stale_process_cleanup
            if stale_process_cleanup.get('still_running_pids'):
                row['restart_skipped'] = 'stale_process_cleanup_pending'
                row['reason'] = 'single_instance_stale_process_cleanup_pending'
                status.append(row)
                continue

        restart_budget_repair_probe: Dict[str, Any] = {'allowed': False}
        if not _within_budget(events, t['name'], args.max_restarts_per_hour):
            restart_budget_repair_probe = _restart_budget_repair_probe(
                events=events,
                name=str(t['name']),
                row=row,
                cooldown_seconds=int(args.readonly_repair_probe_cooldown_seconds),
            )
            if bool(restart_budget_repair_probe.get('allowed', False)):
                row['restart_budget_repair_probe'] = restart_budget_repair_probe
                row['restart_budget_probe_active'] = True
            else:
                row['restart_budget_repair_probe'] = restart_budget_repair_probe
                row['restart_skipped'] = 'budget_exhausted'
                status.append(row)
                alert_severity, alert_event = _restart_budget_alert_metadata(str(t['name']), row)
                alerts.append(
                    {
                        'name': t['name'],
                        'type': 'budget_exhausted',
                        'alert': _alert(
                            alert_severity,
                            alert_event,
                            f"Restart budget exhausted for {t['name']}.",
                            suppress_seconds=max(args.alert_suppress_seconds, 60),
                        ),
                    }
                )
                continue

        restart_reason = 'process_missing' if not process_live else 'heartbeat_stale'
        if parent_process_required and running > 0 and not child_fanout_ok:
            restart_reason = str(child_fanout.get('reason') or 'child_fanout_below_floor')
            row['parent_cleanup'] = _terminate_matching_processes(
                [str(t['pattern'])],
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('orphan_cleanup_grace_seconds', 3.0) or 3.0),
            )
            if row['parent_cleanup'].get('still_running_pids'):
                row['restart_skipped'] = 'parent_cleanup_pending'
                row['restart_reason'] = restart_reason
                status.append(row)
                continue
            cleanup_patterns = t.get('orphan_cleanup_patterns') if isinstance(t.get('orphan_cleanup_patterns'), list) else []
            cleanup = _terminate_matching_processes(
                [str(pattern) for pattern in cleanup_patterns if str(pattern or '').strip()],
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('orphan_cleanup_grace_seconds', 3.0) or 3.0),
            )
            row['orphan_cleanup'] = cleanup
            row['alt_running_after_cleanup'] = sum(
                _proc_running(p, exclude_patterns=t.get('exclude_patterns', []))
                for p in t.get('alt_patterns', [])
                if p
            )
        elif parent_process_required and running <= 0 and alt_running > 0:
            restart_reason = 'parent_launcher_missing'
            cleanup_patterns = t.get('orphan_cleanup_patterns') if isinstance(t.get('orphan_cleanup_patterns'), list) else []
            cleanup = _terminate_matching_processes(
                [str(pattern) for pattern in cleanup_patterns if str(pattern or '').strip()],
                exclude_patterns=t.get('exclude_patterns', []),
                grace_seconds=float(t.get('orphan_cleanup_grace_seconds', 3.0) or 3.0),
            )
            row['orphan_cleanup'] = cleanup
            row['alt_running_after_cleanup'] = sum(
                _proc_running(p, exclude_patterns=t.get('exclude_patterns', []))
                for p in t.get('alt_patterns', [])
                if p
            )

        pid = _spawn(t['cmd'], t['log'])
        ts = datetime.now(timezone.utc).isoformat()
        evt = {
            'name': t['name'],
            'event': 'restart',
            'pid': pid,
            'timestamp_utc': ts,
            'ts_epoch': time.time(),
            'reason': restart_reason,
            'budget_repair_probe': bool(restart_budget_repair_probe.get('allowed', False)),
        }
        events.append(evt)
        restarts.append(evt)
        row['restarted_pid'] = pid
        row['restart_reason'] = evt['reason']
        status.append(row)

    if SNAPSHOT_SCRIPT.exists() and restarts:
        subprocess.run([str(SNAPSHOT_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)

    restart_window_seconds = max(int(args.restart_storm_window_seconds), 60)
    restart_storms, recent_restart_storms = _resolved_restart_storms(
        events=events,
        status_rows=status,
        restart_window_seconds=restart_window_seconds,
        restart_storm_threshold=max(int(args.restart_storm_threshold), 1),
        settle_seconds=max(int(args.restart_storm_settle_seconds), 60),
    )
    for storm in restart_storms:
        storm_isolated = bool(storm.get('quarantinable', False)) and not bool(storm.get('blocks_execution_clear', True))
        alert_severity = 'warn' if storm_isolated else 'critical'
        alert_key = 'watchdog_restart_storm_isolated' if storm_isolated else 'watchdog_restart_storm'
        alerts.append(
            {
                'name': storm['name'],
                'type': 'restart_storm',
                'count': storm['count'],
                'alert': _alert(
                    alert_severity,
                    alert_key,
                    f"Restart storm: {storm['name']} restarted {storm['count']} times in {restart_window_seconds}s.",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    events, restart_debt_forgiveness = _forgive_resolved_restart_debt(events, recent_restart_storms)
    events = sorted(events, key=lambda x: float(x.get('ts_epoch', 0)))[-800:]
    state = {
        'events': events,
        'restart_debt_forgiveness': restart_debt_forgiveness,
        'network_fail_streak': int(network_fail_streak_now),
        'storage_mode': storage_mode,
        'storage_mount_present': bool(storage_mount_present),
        'storage_external_available': bool(storage_mount_guard.get('external_available', False)),
        'storage_mount_root': str(storage_mount_guard.get('mount_root', '')),
        'storage_external_root': str(storage_mount_guard.get('external_root', '')),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    state_written = _save_state(DEFAULT_STATE_PATH, FALLBACK_STATE_PATH, state)

    storage_mount_guard_written = _write_payload(
        DEFAULT_STORAGE_MOUNT_GUARD_PATH,
        FALLBACK_STORAGE_MOUNT_GUARD_PATH,
        {
            **storage_mount_guard,
            'storage_mode_transition': storage_mode_transition,
            'storage_mount_transition': storage_mount_transition,
        },
    )
    infrabot_assignments = [
        {
            'target': str(row.get('name') or ''),
            'repair_infrabots': list(row.get('repair_infrabots') or []),
            'repair_commands': list(row.get('repair_commands') or []),
            'repair_policy': str(row.get('repair_policy') or ''),
            'active': not bool(row.get('heartbeat_ok', False)),
        }
        for row in status
        if row.get('repair_infrabots')
    ]
    watchdog_intelligence = _watchdog_intelligence_contract(
        status_rows=status,
        restarts=restarts,
        restart_storms=restart_storms,
        recent_restart_storms=recent_restart_storms,
        alerts=alerts,
        safety_pause=safety_pause,
        creative_pause=creative_pause,
        network_payload=network_payload,
    )

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'overall_status': watchdog_intelligence['overall_status'],
        'watchdog_intelligence': watchdog_intelligence,
        'status': status,
        'restarts': restarts,
        'restart_storms': restart_storms,
        'recent_restart_storms': recent_restart_storms,
        'restart_storm_isolation': _restart_storm_isolation_contract(restart_storms),
        'restart_debt_forgiveness': restart_debt_forgiveness,
        'safety_pause': safety_pause,
        'creative_cotenant_pause': creative_pause,
        'alerts': alerts,
        'max_restarts_per_hour': int(args.max_restarts_per_hour),
        'maintenance': maintenance,
        'refresh_reports': refresh_payload,
        'network': network_payload,
        'storage_mode_transition': storage_mode_transition,
        'storage_mount_transition': storage_mount_transition,
        'storage_mount_guard': {
            **storage_mount_guard,
            'out_file': str(storage_mount_guard_written),
        },
        'infrabot_assignments': infrabot_assignments,
        'state_file': str(state_written),
    }
    out_written = _write_payload(DEFAULT_OUT_PATH, FALLBACK_OUT_PATH, payload)
    payload['out_file'] = str(out_written)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"process_watchdog restarts={len(restarts)} out={out_written}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
