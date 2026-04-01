import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

DEFAULT_TOKEN_PATH = PROJECT_ROOT / 'token.json'
DEFAULT_OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'premarket_token_guard_latest.json'
DEFAULT_EVENT_DIR = PROJECT_ROOT / 'governance' / 'events'
FALLBACK_OUT_PATH = Path('/tmp/premarket_token_guard_latest.json')
FALLBACK_EVENT_PATH = Path('/tmp/premarket_token_guard_events.jsonl')
ALERT_ROUTER = PROJECT_ROOT / 'scripts' / 'pager_alert_router.py'
PY = resolve_runtime_python(PROJECT_ROOT)



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def _write_json(path: Path, fallback: Path, payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return str(path)
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return str(fallback)



def _append_jsonl(path: Path, fallback: Path, row: Dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(encoded + '\n')
        return str(path)
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open('a', encoding='utf-8') as f:
            f.write(encoded + '\n')
        return str(fallback)



def _run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()



def _alert(severity: str, event: str, message: str, suppress_seconds: int = 1800) -> Dict[str, Any]:
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
    rc, out, err = _run(cmd)
    return {
        'attempted': True,
        'rc': int(rc),
        'stdout': out[-500:],
        'stderr': err[-500:],
    }



def _split_hostport(raw: str) -> tuple[str, int]:
    value = (raw or '').strip()
    host = value
    port = 443
    if ':' in value:
        host, port_raw = value.rsplit(':', 1)
        try:
            port = int(port_raw.strip())
        except Exception:
            port = 443
    return host.strip(), port



def _probe_network(hostport: str, timeout_seconds: float) -> Dict[str, Any]:
    host, port = _split_hostport(hostport)
    if not host:
        return {'hostport': hostport, 'ok': False, 'error': 'empty_host'}
    try:
        with socket.create_connection((host, port), timeout=max(float(timeout_seconds), 0.2)):
            return {'hostport': f'{host}:{port}', 'ok': True}
    except Exception as exc:
        return {'hostport': f'{host}:{port}', 'ok': False, 'error': f'{type(exc).__name__}:{exc}'}



def _token_status(path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        'token_path': str(path),
        'exists': path.exists(),
        'size_bytes': 0,
        'age_seconds': None,
        'expires_at': '',
        'expires_in_seconds': None,
    }
    if not path.exists():
        return status

    try:
        st = path.stat()
        status['size_bytes'] = int(st.st_size)
        status['age_seconds'] = max(datetime.now(timezone.utc).timestamp() - st.st_mtime, 0.0)
    except Exception:
        pass

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        expiry_sources = [payload]
        nested = payload.get('token')
        if isinstance(nested, dict):
            expiry_sources.insert(0, nested)

        exp_value: Any = ''
        for source in expiry_sources:
            for key in ('expires_at', 'expiresAt', 'expires', 'expires_time'):
                raw = source.get(key)
                if raw not in (None, ''):
                    exp_value = raw
                    break
            if exp_value not in (None, ''):
                break

        if exp_value not in (None, ''):
            status['expires_at'] = str(exp_value)

    expires_at = str(status.get('expires_at') or '').strip()
    if expires_at:
        try:
            if expires_at.replace('.', '', 1).isdigit():
                exp_ts = float(expires_at)
            else:
                dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                exp_ts = dt.astimezone(timezone.utc).timestamp()
            status['expires_in_seconds'] = exp_ts - datetime.now(timezone.utc).timestamp()
        except Exception:
            pass

    return status



def _token_needs_refresh(
    status: Dict[str, Any],
    max_age_seconds: float,
    min_expires_seconds: float,
) -> tuple[bool, str]:
    if not bool(status.get('exists')):
        return True, 'missing_token'

    size = int(status.get('size_bytes') or 0)
    if size < 64:
        return True, 'token_too_small'

    age = status.get('age_seconds')
    if age is not None and float(age) > max(float(max_age_seconds), 0.0):
        return True, f'token_age_high:{float(age):.1f}'

    expires_floor = max(float(min_expires_seconds), 0.0)
    expires_in = status.get('expires_in_seconds')
    if expires_in is not None and float(expires_in) <= expires_floor:
        return True, f'token_expiring_soon:{float(expires_in):.1f}'

    return False, 'token_fresh'



def _auth_attempt(token_path: Path, callback_timeout_seconds: float, validate_account_probe: bool) -> Dict[str, Any]:
    api_key = os.getenv('SCHWAB_API_KEY', '').strip()
    app_secret = os.getenv('SCHWAB_SECRET', '').strip()
    callback_url = (
        os.getenv('SCHWAB_CALLBACK_URL', '').strip()
        or os.getenv('SCHWAB_REDIRECT', '').strip()
        or 'https://127.0.0.1:8182'
    )

    invalid = {'', 'YOUR_KEY_HERE', 'YOUR_SECRET_HERE', 'YOUR_REAL_KEY', 'YOUR_REAL_SECRET', '<real_key>', '<real_secret>'}
    if api_key in invalid or app_secret in invalid:
        return {
            'attempted': False,
            'ok': False,
            'reason': 'missing_credentials',
            'details': {
                'callback_url': callback_url,
            },
        }

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    os.environ['SCHWAB_AUTH_INTERACTIVE'] = '0'
    os.environ.setdefault('SCHWAB_MAX_TOKEN_AGE_SECONDS', '0')
    os.environ['SCHWAB_AUTH_CALLBACK_TIMEOUT_SECONDS'] = str(max(float(callback_timeout_seconds), 5.0))

    try:
        from core.base_trader import BaseTrader

        trader = BaseTrader(api_key=api_key, app_secret=app_secret, callback_url=callback_url, mode='shadow')
        trader.token_path = str(token_path)
        trader.authenticate()
        details: Dict[str, Any] = {
            'callback_url': callback_url,
            'interactive': False,
        }
        if validate_account_probe:
            resp = trader.client.get_account_numbers()
            status_code = int(getattr(resp, 'status_code', 0) or 0)
            details['account_probe_status_code'] = status_code
            if not (200 <= status_code < 300):
                body = (getattr(resp, 'text', '') or '')[:300]
                return {
                    'attempted': True,
                    'ok': False,
                    'reason': f'account_probe_failed:{status_code}',
                    'details': {
                        **details,
                        'account_probe_body': body,
                    },
                }
        return {
            'attempted': True,
            'ok': True,
            'reason': 'auth_success',
            'details': details,
        }
    except Exception as exc:
        return {
            'attempted': True,
            'ok': False,
            'reason': f'auth_error:{type(exc).__name__}:{exc}',
            'details': {
                'callback_url': callback_url,
                'interactive': False,
            },
        }



def main() -> int:
    parser = argparse.ArgumentParser(description='Premarket Schwab token guard with auto-refresh + alerting.')
    parser.add_argument('--token-path', default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument('--max-token-age-seconds', type=float, default=float(os.getenv('PREMARKET_TOKEN_MAX_AGE_SECONDS', '43200')))
    parser.add_argument(
        '--min-expires-seconds',
        type=float,
        default=float(os.getenv('PREMARKET_TOKEN_MIN_EXPIRES_SECONDS', '600')),
    )
    parser.add_argument('--auth-timeout-seconds', type=float, default=float(os.getenv('PREMARKET_TOKEN_AUTH_TIMEOUT_SECONDS', '30')))
    parser.add_argument('--always-auth', dest='always_auth', action='store_true', help='Always run non-interactive auth, even when token looks fresh.')
    parser.add_argument('--no-always-auth', dest='always_auth', action='store_false', help='Skip auth when token is fresh.')
    parser.add_argument('--network-host', default=os.getenv('PREMARKET_TOKEN_NETWORK_HOST', 'api.schwabapi.com:443'))
    parser.add_argument('--network-timeout-seconds', type=float, default=float(os.getenv('PREMARKET_TOKEN_NETWORK_TIMEOUT_SECONDS', '2.5')))
    parser.add_argument('--skip-network-check', action='store_true', default=os.getenv('PREMARKET_TOKEN_SKIP_NETWORK_CHECK', '0').strip() == '1')
    parser.add_argument(
        '--validate-account-probe',
        dest='validate_account_probe',
        action='store_true',
        help='Require a real authenticated account probe after token auth.',
    )
    parser.add_argument(
        '--no-validate-account-probe',
        dest='validate_account_probe',
        action='store_false',
        help='Skip the post-auth account probe.',
    )
    parser.add_argument('--alert-suppress-seconds', type=int, default=int(os.getenv('PREMARKET_TOKEN_ALERT_SUPPRESS_SECONDS', '1800')))
    parser.add_argument('--json', action='store_true')
    parser.set_defaults(
        always_auth=os.getenv('PREMARKET_TOKEN_ALWAYS_AUTH', '0').strip() == '1',
        validate_account_probe=os.getenv('PREMARKET_TOKEN_VALIDATE_ACCOUNT_PROBE', '1').strip() != '0',
    )
    args = parser.parse_args()

    now_iso = _now_iso()
    token_path = Path(args.token_path)
    before = _token_status(token_path)
    min_expires_seconds = max(float(args.min_expires_seconds), 0.0)
    needs_refresh, refresh_reason = _token_needs_refresh(
        before,
        max_age_seconds=max(args.max_token_age_seconds, 60.0),
        min_expires_seconds=min_expires_seconds,
    )

    network = {
        'checked': not bool(args.skip_network_check),
        'probe': {},
        'ok': True,
    }
    if not args.skip_network_check:
        probe = _probe_network(args.network_host, timeout_seconds=float(args.network_timeout_seconds))
        network['probe'] = probe
        network['ok'] = bool(probe.get('ok'))

    auth: Dict[str, Any] = {'attempted': False, 'ok': True, 'reason': 'not_needed'}
    if args.always_auth or needs_refresh:
        if network['ok']:
            auth = _auth_attempt(
                token_path=token_path,
                callback_timeout_seconds=float(args.auth_timeout_seconds),
                validate_account_probe=bool(args.validate_account_probe),
            )
        else:
            auth = {
                'attempted': False,
                'ok': False,
                'reason': 'network_unavailable',
            }

    after = _token_status(token_path)
    still_stale, stale_reason_after = _token_needs_refresh(
        after,
        max_age_seconds=max(args.max_token_age_seconds, 60.0),
        min_expires_seconds=min_expires_seconds,
    )

    if auth.get('attempted') and auth.get('ok') and still_stale:
        auth = {
            **auth,
            'ok': False,
            'reason': f"auth_succeeded_but_token_not_ready:{stale_reason_after}",
        }

    ok = bool(after.get('exists')) and int(after.get('size_bytes') or 0) >= 64 and bool(network['ok']) and (not still_stale)
    if auth.get('attempted') and not auth.get('ok'):
        ok = False

    alerts: list[Dict[str, Any]] = []
    if not network['ok']:
        alerts.append(
            {
                'type': 'network_unavailable',
                'alert': _alert(
                    'warn',
                    'premarket_token_network_unavailable',
                    'Premarket token guard could not reach Schwab API host.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )
    if auth.get('attempted') and not auth.get('ok'):
        alerts.append(
            {
                'type': 'auth_failed',
                'alert': _alert(
                    'critical',
                    'premarket_token_refresh_failed',
                    f"Premarket token refresh failed: {auth.get('reason', 'unknown')}",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )
    elif auth.get('attempted') and auth.get('ok'):
        alerts.append(
            {
                'type': 'auth_success',
                'alert': _alert(
                    'info',
                    'premarket_token_refresh_ok',
                    'Premarket token refresh succeeded.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    if not ok:
        alerts.append(
            {
                'type': 'token_guard_failed',
                'alert': _alert(
                    'critical',
                    'premarket_token_guard_failed',
                    f"Token not ready for premarket. before={refresh_reason} after={stale_reason_after} auth={auth.get('reason', 'n/a')}",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    payload: Dict[str, Any] = {
        'timestamp_utc': now_iso,
        'ok': bool(ok),
        'token_before': before,
        'token_after': after,
        'refresh_needed_before': bool(needs_refresh),
        'refresh_reason_before': refresh_reason,
        'refresh_needed_after': bool(still_stale),
        'refresh_reason_after': stale_reason_after,
        'network': network,
        'auth': auth,
        'validate_account_probe': bool(args.validate_account_probe),
        'alerts': alerts,
    }

    out_file = _write_json(DEFAULT_OUT_PATH, FALLBACK_OUT_PATH, payload)
    event_path = DEFAULT_EVENT_DIR / f"premarket_token_guard_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    events_file = _append_jsonl(event_path, FALLBACK_EVENT_PATH, payload)
    payload['out_file'] = out_file
    payload['events_file'] = events_file

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"premarket_token_guard ok={int(bool(ok))} out={out_file}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
