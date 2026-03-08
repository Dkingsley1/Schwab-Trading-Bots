import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
PREFLIGHT = PROJECT_ROOT / 'scripts' / 'shadow_preflight.py'
OUT = PROJECT_ROOT / 'governance' / 'health' / 'preflight_autofix_latest.json'
ALERT_LATEST = PROJECT_ROOT / 'governance' / 'alerts' / 'preflight_critical_latest.json'
ALERT_EVENTS = PROJECT_ROOT / 'governance' / 'alerts' / 'preflight_critical_events.jsonl'
CRITICAL_CHECKS = {
    'symbol_groups_non_empty',
    'schwab_credentials_present',
    'token_present',
    'storage_route_writable',
    'disk_free',
    'global_halt_not_set',
}


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()
    except Exception as exc:
        return 1, '', str(exc)


def _parse_checks(payload: dict) -> list[dict]:
    rows = payload.get('checks') if isinstance(payload.get('checks'), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            'name': str(row.get('name', 'unknown')),
            'ok': bool(row.get('ok', False)),
            'details': str(row.get('details', '')),
        })
    return out


def _find_pids(pattern: str, exclude_patterns: list[str] | None = None) -> list[int]:
    rc, out, _ = _run(['ps', '-axo', 'pid,command'])
    if rc != 0:
        return []
    excludes = [x for x in (exclude_patterns or []) if x]
    hits: list[int] = []
    for line in out.splitlines():
        s = line.strip()
        if not s or pattern not in s:
            continue
        if any(ex in s for ex in excludes):
            continue
        parts = s.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        if pid != os.getpid():
            hits.append(pid)
    return sorted(set(hits))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=True) + '\n')


def _kill_pids(pids: list[int]) -> list[int]:
    killed: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 15)
            killed.append(pid)
        except Exception:
            continue
    return killed


def main() -> int:
    parser = argparse.ArgumentParser(description='Run preflight and emit actionable auto-fix suggestions.')
    parser.add_argument('--broker', choices=['schwab', 'coinbase'], default=os.getenv('DATA_BROKER', 'schwab'))
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--allow-running', action='store_true')
    parser.add_argument('--apply-kill-duplicates', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    cmd = [str(PY), str(PREFLIGHT), '--broker', args.broker, '--json']
    if args.simulate:
        cmd.append('--simulate')
    if args.allow_running:
        cmd.append('--allow-running')

    # Keep symbol inputs explicit so suggestions match current env.
    core = os.getenv('SHADOW_SYMBOLS_CORE', '')
    vol = os.getenv('SHADOW_SYMBOLS_VOLATILE', '')
    defensive = os.getenv('SHADOW_SYMBOLS_DEFENSIVE', '')
    extra = os.getenv('SHADOW_SYMBOLS_COMMOD_FX_INTL', '')
    if core:
        cmd.extend(['--symbols-core', core])
    if vol:
        cmd.extend(['--symbols-volatile', vol])
    if defensive:
        cmd.extend(['--symbols-defensive', defensive + (',' + extra if extra else '')])

    rc, out, err = _run(cmd)
    payload = {}
    try:
        payload = json.loads(out) if out else {}
    except Exception:
        payload = {'ok': False, 'checks': []}

    checks = _parse_checks(payload)
    failed = [c for c in checks if not c['ok']]

    suggestions: list[dict] = []
    applied: list[dict] = []

    for item in failed:
        name = item['name']
        if name == 'schwab_credentials_present':
            suggestions.append({
                'issue': name,
                'severity': 'critical',
                'fix': 'Set real Schwab credentials in your shell/profile (not placeholders).',
                'command': 'export SCHWAB_API_KEY="<real_key>" && export SCHWAB_SECRET="<real_secret>"',
            })
        elif name == 'token_present':
            suggestions.append({
                'issue': name,
                'severity': 'critical',
                'fix': 'Generate/refresh token.json for Schwab auth.',
                'command': 'ls -lh token.json && echo "refresh token if missing/expired"',
            })
        elif name == 'symbol_groups_non_empty':
            suggestions.append({
                'issue': name,
                'severity': 'critical',
                'fix': 'Symbol groups were empty. Auto-load defaults via runtime profile/env loader before start.',
                'command': 'source scripts/ops/load_runtime_env.sh sim --quiet && ./scripts/ops/opsctl.sh start-sim --force-restart',
            })
        elif name == 'disk_free':
            suggestions.append({
                'issue': name,
                'severity': 'high',
                'fix': 'Run retention prune before startup.',
                'command': './.venv312/bin/python scripts/data_retention_policy.py --apply --json',
            })
        elif name == 'storage_route_writable':
            suggestions.append({
                'issue': name,
                'severity': 'critical',
                'fix': 'Storage route was not writable. Verify SSD mount or local fallback permissions.',
                'command': './.venv312/bin/python scripts/ops/storage_failback_sync.py --json',
            })
        elif name == 'no_duplicate_parallel_launcher':
            pids = _find_pids(
                'scripts/run_parallel_shadows.py',
                exclude_patterns=[
                    'scripts/shadow_watchdog.py',
                    'scripts/ops/preflight_autofix.py',
                    'scripts/shadow_preflight.py',
                ],
            )
            killed = _kill_pids(pids) if (args.apply_kill_duplicates and pids) else []
            suggestions.append({
                'issue': name,
                'severity': 'medium',
                'fix': 'Stop duplicate shadow launcher before starting a new one.',
                'command': ('kill ' + ' '.join(str(x) for x in pids)) if pids else 'ps -axo pid,command | grep run_parallel_shadows.py',
            })
            if killed:
                applied.append({'issue': name, 'killed_pids': killed})
        elif name == 'no_duplicate_coinbase_loop':
            pids = _find_pids('scripts/run_shadow_training_loop.py --broker coinbase', exclude_patterns=['--profile crypto_futures'])
            killed = _kill_pids(pids) if (args.apply_kill_duplicates and pids) else []
            suggestions.append({
                'issue': name,
                'severity': 'medium',
                'fix': 'Stop duplicate Coinbase loop before starting a new one.',
                'command': ('kill ' + ' '.join(str(x) for x in pids)) if pids else 'ps -axo pid,command | grep "--broker coinbase"',
            })
            if killed:
                applied.append({'issue': name, 'killed_pids': killed})

    critical_failed = [c for c in failed if c.get('name') in CRITICAL_CHECKS]

    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'broker': args.broker,
        'simulate': bool(args.simulate),
        'preflight_ok': bool(payload.get('ok', False)),
        'failed_checks': failed,
        'suggestions': suggestions,
        'applied': applied,
        'stderr': err,
        'preflight_rc': int(rc),
        'critical_failed': critical_failed,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding='utf-8')

    ALERT_LATEST.parent.mkdir(parents=True, exist_ok=True)

    if critical_failed:
        alert_payload = {
            'timestamp_utc': result['timestamp_utc'],
            'broker': result['broker'],
            'simulate': result['simulate'],
            'failed_checks': critical_failed,
            'suggestions': [s for s in suggestions if s.get('issue') in {c.get('name') for c in critical_failed}],
            'source': str(OUT),
        }
        ALERT_LATEST.write_text(json.dumps(alert_payload, ensure_ascii=True, indent=2), encoding='utf-8')
        _append_jsonl(ALERT_EVENTS, alert_payload)
        print(
            f"[ALERT] preflight_critical count={len(critical_failed)} file={ALERT_LATEST}",
            file=sys.stderr,
        )
    else:
        clear_payload = {
            'timestamp_utc': result['timestamp_utc'],
            'broker': result['broker'],
            'simulate': result['simulate'],
            'failed_checks': [],
            'suggestions': [],
            'cleared': True,
            'source': str(OUT),
        }
        ALERT_LATEST.write_text(json.dumps(clear_payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        print(f"preflight_autofix ok={result['preflight_ok']} failed={len(failed)} suggestions={len(suggestions)}")
        for s in suggestions:
            print(f" - {s['issue']}: {s['fix']}")
            print(f"   cmd: {s['command']}")

    return 0 if result['preflight_ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
