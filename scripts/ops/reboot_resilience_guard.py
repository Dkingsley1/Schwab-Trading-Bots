import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

ALERT_ROUTER = PROJECT_ROOT / 'scripts' / 'pager_alert_router.py'
PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'reboot_resilience_latest.json'
FALLBACK_OUT_PATH = Path('/tmp/reboot_resilience_latest.json')

DEFAULT_REQUIRED_LABELS = [
    'com.dankingsley.all_sleeves',
    'com.dankingsley.shadow_watchdog',
    'com.dankingsley.caffeinate_guard',
    'com.dankingsley.ops.watchdog',
    'com.dankingsley.failover_hot_standby',
]


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()



def _write_payload(path: Path, fallback: Path, payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return str(path)
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return str(fallback)



def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or '').split(',') if x.strip()]



def _alert(severity: str, event: str, message: str, suppress_seconds: int = 900) -> Dict[str, Any]:
    if not ALERT_ROUTER.exists() or not PY.exists():
        return {'attempted': False, 'reason': 'alert_router_missing'}

    rc, out, err = _run(
        [
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
    )
    return {
        'attempted': True,
        'rc': int(rc),
        'stdout': out[-500:],
        'stderr': err[-500:],
    }



def _launchagent_path(label: str) -> Path:
    return Path.home() / 'Library' / 'LaunchAgents' / f'{label}.plist'



def _is_loaded(domain: str, label: str) -> bool:
    rc, _out, _err = _run(['launchctl', 'print', f'{domain}/{label}'])
    return rc == 0



def _recover_label(domain: str, label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        'label': label,
        'plist': str(_launchagent_path(label)),
        'loaded_before': False,
        'loaded_after': False,
        'actions': [],
        'ok': False,
    }

    plist = _launchagent_path(label)
    loaded_before = _is_loaded(domain, label)
    row['loaded_before'] = bool(loaded_before)

    rc_en, out_en, err_en = _run(['launchctl', 'enable', f'{domain}/{label}'])
    row['actions'].append({'action': 'enable', 'rc': int(rc_en), 'stdout': out_en[-200:], 'stderr': err_en[-200:]})

    if plist.exists():
        rc_bootout, out_bootout, err_bootout = _run(['launchctl', 'bootout', domain, str(plist)])
        row['actions'].append(
            {'action': 'bootout', 'rc': int(rc_bootout), 'stdout': out_bootout[-200:], 'stderr': err_bootout[-200:]}
        )

        rc_bootstrap, out_bootstrap, err_bootstrap = _run(['launchctl', 'bootstrap', domain, str(plist)])
        row['actions'].append(
            {
                'action': 'bootstrap',
                'rc': int(rc_bootstrap),
                'stdout': out_bootstrap[-200:],
                'stderr': err_bootstrap[-200:],
            }
        )
    else:
        row['actions'].append({'action': 'bootstrap', 'rc': 1, 'stderr': 'plist_missing', 'stdout': ''})

    rc_kick, out_kick, err_kick = _run(['launchctl', 'kickstart', '-k', f'{domain}/{label}'])
    row['actions'].append({'action': 'kickstart', 'rc': int(rc_kick), 'stdout': out_kick[-200:], 'stderr': err_kick[-200:]})

    loaded_after = _is_loaded(domain, label)
    row['loaded_after'] = bool(loaded_after)
    row['ok'] = bool(loaded_after)
    return row



def main() -> int:
    parser = argparse.ArgumentParser(description='Reboot resilience guard for launchd runtime stack.')
    parser.add_argument(
        '--required-labels',
        default=os.getenv('REBOOT_GUARD_REQUIRED_LABELS', ','.join(DEFAULT_REQUIRED_LABELS)),
        help='Comma-separated LaunchAgent labels to keep loaded.',
    )
    parser.add_argument(
        '--critical-labels',
        default=os.getenv('REBOOT_GUARD_CRITICAL_LABELS', ''),
        help='Comma-separated critical labels. Defaults to required labels.',
    )
    parser.add_argument('--alert-suppress-seconds', type=int, default=int(os.getenv('REBOOT_GUARD_ALERT_SUPPRESS_SECONDS', '900')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    uid = os.getuid()
    domain = f'gui/{uid}'
    required = _split_csv(args.required_labels)
    critical = _split_csv(args.critical_labels) if args.critical_labels.strip() else list(required)

    recovered: List[Dict[str, Any]] = []
    healthy: List[Dict[str, Any]] = []

    for label in required:
        if _is_loaded(domain, label):
            healthy.append({'label': label, 'loaded_before': True, 'loaded_after': True, 'ok': True, 'actions': []})
            continue
        recovered.append(_recover_label(domain, label))

    all_rows = healthy + recovered
    failed = [r for r in all_rows if not bool(r.get('ok'))]
    failed_critical = [r for r in failed if r.get('label') in critical]

    alerts: List[Dict[str, Any]] = []
    if failed:
        labels = ','.join(str(r.get('label', 'unknown')) for r in failed)
        alerts.append(
            {
                'type': 'recovery_failed',
                'alert': _alert(
                    'critical' if failed_critical else 'warn',
                    'reboot_resilience_recovery_failed',
                    f'Reboot resilience failed to recover: {labels}',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )
    elif recovered:
        labels = ','.join(str(r.get('label', 'unknown')) for r in recovered)
        alerts.append(
            {
                'type': 'recovery_applied',
                'alert': _alert(
                    'warn',
                    'reboot_resilience_recovery_applied',
                    f'Reboot resilience reloaded: {labels}',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    ok = len(failed_critical) == 0
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'domain': domain,
        'required_labels': required,
        'critical_labels': critical,
        'healthy': healthy,
        'recovered': recovered,
        'failed': failed,
        'alerts': alerts,
    }
    out_file = _write_payload(DEFAULT_OUT_PATH, FALLBACK_OUT_PATH, payload)
    payload['out_file'] = out_file

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"reboot_resilience_guard ok={int(bool(ok))} failed={len(failed)} out={out_file}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
