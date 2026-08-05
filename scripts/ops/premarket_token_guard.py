import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from scripts.brokers.schwab.common import (
    credentials_ready,
    schwab_credentials_from_env,
    token_needs_refresh as common_token_needs_refresh,
    token_status as common_token_status,
)

DEFAULT_TOKEN_PATH = PROJECT_ROOT / 'token.json'
DEFAULT_OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'premarket_token_guard_latest.json'
DEFAULT_BROKER_READINESS_PATH = PROJECT_ROOT / 'governance' / 'health' / 'broker_readiness_latest.json'
DEFAULT_EVENT_DIR = PROJECT_ROOT / 'governance' / 'events'
FALLBACK_OUT_PATH = Path('/tmp/premarket_token_guard_latest.json')
FALLBACK_BROKER_READINESS_PATH = Path('/tmp/broker_readiness_latest.json')
FALLBACK_EVENT_PATH = Path('/tmp/premarket_token_guard_events.jsonl')
ALERT_ROUTER = PROJECT_ROOT / 'scripts' / 'pager_alert_router.py'
PY = resolve_runtime_python(PROJECT_ROOT)



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str, default: str = '0') -> bool:
    return str(os.getenv(name, default)).strip().lower() in {'1', 'true', 'yes', 'on'}


def _browser_auth_disabled() -> bool:
    return _env_flag('PREMARKET_TOKEN_BROWSER_AUTH_DISABLED', '0') or _env_flag('SCHWAB_AUTH_BROWSER_DISABLED', '0') or not _env_flag('SCHWAB_AUTH_ALLOW_BROWSER_OPEN', '1')



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
    return common_token_status(path)



def _token_needs_refresh(
    status: Dict[str, Any],
    max_age_seconds: float,
    min_expires_seconds: float,
) -> tuple[bool, str]:
    return common_token_needs_refresh(
        status,
        min_expires_seconds=min_expires_seconds,
        max_age_seconds=max_age_seconds,
        ready_reason='token_fresh',
    )


def _token_warning_level(age_seconds: float | None, *, max_age_seconds: float) -> str:
    if age_seconds is None:
        return 'unknown'
    age = max(float(age_seconds), 0.0)
    max_age = max(float(max_age_seconds), 1.0)
    if age >= max_age:
        return 'critical'
    if age >= max_age * 0.75:
        return 'warn'
    if age >= max_age * 0.5:
        return 'watch'
    return 'fresh'



def _auth_attempt(token_path: Path, callback_timeout_seconds: float, validate_account_probe: bool) -> Dict[str, Any]:
    if _browser_auth_disabled():
        return {
            'attempted': False,
            'ok': False,
            'reason': 'browser_auth_disabled',
            'details': {
                'method': 'client_auth',
                'browser_disabled': True,
            },
        }

    api_key = os.getenv('SCHWAB_API_KEY', '').strip()
    app_secret = os.getenv('SCHWAB_SECRET', '').strip()
    callback_url = (
        os.getenv('SCHWAB_CALLBACK_URL', '').strip()
        or os.getenv('SCHWAB_REDIRECT', '').strip()
        or 'https://127.0.0.1:8182'
    )

    if not credentials_ready(schwab_credentials_from_env()):
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


def _write_token_atomic(token_path: Path, payload: Dict[str, Any]) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = token_path.with_suffix(token_path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')
    try:
        mode = token_path.stat().st_mode & 0o777
        os.chmod(tmp, mode)
    except Exception:
        pass
    tmp.replace(token_path)


def _direct_refresh_token_grant(token_path: Path, *, min_extension_seconds: float = 300.0) -> Dict[str, Any]:
    api_key = os.getenv('SCHWAB_API_KEY', '').strip()
    app_secret = os.getenv('SCHWAB_SECRET', '').strip()
    if not credentials_ready(schwab_credentials_from_env()):
        return {'attempted': False, 'ok': False, 'reason': 'missing_credentials', 'details': {'method': 'refresh_token_grant'}}

    try:
        from authlib.integrations.httpx_client import OAuth2Client
        from schwab.auth import TOKEN_ENDPOINT
    except Exception as exc:
        return {
            'attempted': False,
            'ok': False,
            'reason': f'refresh_grant_import_error:{type(exc).__name__}:{exc}',
            'details': {'method': 'refresh_token_grant'},
        }

    try:
        # schwab-py writes token updates directly, so tolerate a tiny read race
        # with any older loop that may still be shutting down.
        wrapped: Dict[str, Any] | None = None
        last_exc: Exception | None = None
        for _ in range(5):
            try:
                wrapped = json.loads(token_path.read_text(encoding='utf-8'))
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.15)
        if wrapped is None:
            raise last_exc or RuntimeError('token_read_failed')

        old_token = dict(wrapped.get('token') or {})
        refresh_token = str(old_token.get('refresh_token') or '').strip()
        if not refresh_token:
            return {
                'attempted': False,
                'ok': False,
                'reason': 'missing_refresh_token',
                'details': {'method': 'refresh_token_grant'},
            }

        before_expires_at = float(old_token.get('expires_at') or 0.0)
        before_expires_in = before_expires_at - time.time()
        oauth = OAuth2Client(
            api_key,
            client_secret=app_secret,
            token=old_token,
            token_endpoint=TOKEN_ENDPOINT,
        )
        new_token = dict(
            oauth.refresh_token(
                TOKEN_ENDPOINT,
                refresh_token=refresh_token,
                auth=(api_key, app_secret),
            )
        )
        if not str(new_token.get('access_token') or '').strip():
            return {
                'attempted': True,
                'ok': False,
                'reason': 'refresh_returned_no_access_token',
                'details': {'method': 'refresh_token_grant'},
            }
        if not str(new_token.get('refresh_token') or '').strip():
            new_token['refresh_token'] = refresh_token

        after_expires_at = float(new_token.get('expires_at') or (time.time() + float(new_token.get('expires_in') or 0.0)))
        after_expires_in = after_expires_at - time.time()
        min_extension = max(float(min_extension_seconds), 0.0)
        if after_expires_in <= max(before_expires_in, 0.0) + min_extension:
            return {
                'attempted': True,
                'ok': False,
                'reason': 'refresh_did_not_extend_enough',
                'details': {
                    'method': 'refresh_token_grant',
                    'before_expires_in_seconds': round(before_expires_in, 3),
                    'after_expires_in_seconds': round(after_expires_in, 3),
                    'min_extension_seconds': round(min_extension, 3),
                },
            }

        _write_token_atomic(
            token_path,
            {
                'creation_timestamp': int(wrapped.get('creation_timestamp') or time.time()),
                'token': new_token,
            },
        )
        return {
            'attempted': True,
            'ok': True,
            'reason': 'refresh_token_grant_success',
            'details': {
                'method': 'refresh_token_grant',
                'before_expires_in_seconds': round(before_expires_in, 3),
                'after_expires_in_seconds': round(after_expires_in, 3),
                'refresh_token_changed': new_token.get('refresh_token') != refresh_token,
            },
        }
    except Exception as exc:
        return {
            'attempted': True,
            'ok': False,
            'reason': f'refresh_grant_error:{type(exc).__name__}:{exc}',
            'details': {'method': 'refresh_token_grant'},
        }



def main() -> int:
    parser = argparse.ArgumentParser(description='Premarket Schwab token guard with auto-refresh + alerting.')
    parser.add_argument('--token-path', default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument('--max-token-age-seconds', type=float, default=float(os.getenv('PREMARKET_TOKEN_MAX_AGE_SECONDS', '43200')))
    parser.add_argument(
        '--min-expires-seconds',
        type=float,
        default=float(os.getenv('PREMARKET_TOKEN_MIN_EXPIRES_SECONDS', '1500')),
    )
    parser.add_argument(
        '--ready-min-expires-seconds',
        type=float,
        default=float(os.getenv('PREMARKET_TOKEN_READY_MIN_EXPIRES_SECONDS', '900')),
        help='Hard readiness floor; the higher min-expires floor only triggers early refresh.',
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
    parser.add_argument(
        '--skip-refresh-token-grant',
        action='store_true',
        default=os.getenv('PREMARKET_TOKEN_SKIP_REFRESH_TOKEN_GRANT', '0').strip() == '1',
        help='Disable direct OAuth refresh-token renewal before falling back to client auth.',
    )
    parser.add_argument(
        '--refresh-token-min-extension-seconds',
        type=float,
        default=float(os.getenv('PREMARKET_TOKEN_REFRESH_MIN_EXTENSION_SECONDS', '300')),
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
    ready_min_expires_seconds = max(float(args.ready_min_expires_seconds), 0.0)
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
            if needs_refresh and not bool(args.skip_refresh_token_grant):
                auth = _direct_refresh_token_grant(
                    token_path,
                    min_extension_seconds=float(args.refresh_token_min_extension_seconds),
                )
            if not auth.get('attempted') or not auth.get('ok'):
                if _browser_auth_disabled():
                    auth = {
                        'attempted': False,
                        'ok': False,
                        'reason': 'browser_auth_disabled',
                        'details': {
                            'method': 'client_auth',
                            'browser_disabled': True,
                        },
                        'refresh_grant': auth,
                    }
                else:
                    fallback_auth = _auth_attempt(
                        token_path=token_path,
                        callback_timeout_seconds=float(args.auth_timeout_seconds),
                        validate_account_probe=bool(args.validate_account_probe),
                    )
                    auth = {
                        **fallback_auth,
                        'refresh_grant': auth,
                    }
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
    not_ready_after, ready_reason_after = _token_needs_refresh(
        after,
        max_age_seconds=max(args.max_token_age_seconds, 60.0),
        min_expires_seconds=ready_min_expires_seconds,
    )

    if auth.get('attempted') and auth.get('ok') and not_ready_after:
        auth = {
            **auth,
            'ok': False,
            'reason': f"auth_succeeded_but_token_not_ready:{ready_reason_after}",
        }

    ok = bool(after.get('exists')) and int(after.get('size_bytes') or 0) >= 64 and bool(network['ok']) and (not not_ready_after)
    if auth.get('attempted') and not auth.get('ok') and not_ready_after:
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
                    'critical' if not_ready_after else 'warn',
                    'premarket_token_refresh_failed' if not_ready_after else 'premarket_token_refresh_deferred',
                    (
                        f"Premarket token refresh failed: {auth.get('reason', 'unknown')}"
                        if not_ready_after
                        else f"Premarket token refresh did not extend the token yet, but the lease is still above the readiness floor: {auth.get('reason', 'unknown')}"
                    ),
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
                    f"Token not ready for premarket. before={refresh_reason} after={ready_reason_after} auth={auth.get('reason', 'n/a')}",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    account_probe_status = None
    if isinstance(auth.get('details'), dict):
        probe_code = auth['details'].get('account_probe_status_code')
        account_probe_status = int(probe_code) if probe_code not in (None, '') else None
    broker_readiness = {
        'timestamp_utc': now_iso,
        'ready_for_open': bool(ok),
        'token_warning_level': _token_warning_level(after.get('age_seconds'), max_age_seconds=max(args.max_token_age_seconds, 60.0)),
        'token_age_seconds': after.get('age_seconds'),
        'token_expires_in_seconds': after.get('expires_in_seconds'),
        'network_ok': bool(network['ok']),
        'auth_ok': bool(auth.get('ok', False)) if auth.get('attempted') else True,
        'auth_attempted': bool(auth.get('attempted', False)),
        'account_probe_status_code': account_probe_status,
        'preflight_checks': {
            'token_exists': bool(after.get('exists')),
            'token_size_ok': int(after.get('size_bytes') or 0) >= 64,
            'token_ready_for_open': not bool(not_ready_after),
            'network_ok': bool(network['ok']),
            'auth_ok': bool(auth.get('ok', False)) if auth.get('attempted') else True,
            'refresh_needed_after': bool(still_stale),
            'readiness_refresh_needed_after': bool(not_ready_after),
        },
        'warnings': [
            item
            for item in [
                ('network_unavailable' if not network['ok'] else ''),
                # A successful refresh supersedes the pre-refresh warning. Keep
                # it visible only while the condition remains unresolved.
                (refresh_reason if needs_refresh and (still_stale or not_ready_after) else ''),
                (stale_reason_after if still_stale else ''),
                (ready_reason_after if not_ready_after else ''),
                (str(auth.get('reason') or '') if auth.get('attempted') and not auth.get('ok') else ''),
            ]
            if item
        ],
    }

    payload: Dict[str, Any] = {
        'timestamp_utc': now_iso,
        'ok': bool(ok),
        'token_before': before,
        'token_after': after,
        'refresh_needed_before': bool(needs_refresh),
        'refresh_reason_before': refresh_reason,
        'refresh_needed_after': bool(still_stale),
        'refresh_reason_after': stale_reason_after,
        'ready_min_expires_seconds': ready_min_expires_seconds,
        'token_ready_after': not bool(not_ready_after),
        'ready_reason_after': ready_reason_after,
        'network': network,
        'auth': auth,
        'validate_account_probe': bool(args.validate_account_probe),
        'alerts': alerts,
        'broker_readiness': broker_readiness,
    }

    out_file = _write_json(DEFAULT_OUT_PATH, FALLBACK_OUT_PATH, payload)
    broker_readiness_file = _write_json(DEFAULT_BROKER_READINESS_PATH, FALLBACK_BROKER_READINESS_PATH, broker_readiness)
    event_path = DEFAULT_EVENT_DIR / f"premarket_token_guard_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    events_file = _append_jsonl(event_path, FALLBACK_EVENT_PATH, payload)
    payload['out_file'] = out_file
    payload['broker_readiness_file'] = broker_readiness_file
    payload['events_file'] = events_file

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"premarket_token_guard ok={int(bool(ok))} out={out_file}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
