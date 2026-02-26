import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'


def _run(step: str, cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return {
        'step': step,
        'cmd': cmd,
        'rc': int(p.returncode),
        'ok': p.returncode == 0,
        'stdout_tail': '\n'.join((p.stdout or '').splitlines()[-40:]),
        'stderr_tail': '\n'.join((p.stderr or '').splitlines()[-40:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run full promotion pipeline with one artifact.')
    parser.add_argument('--apply-retirement', action='store_true', default=True)
    parser.add_argument('--run-master-update', action='store_true', default=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    steps: list[tuple[str, list[str], bool]] = [
        ('walk_forward_validate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'walk_forward_validate.py')], True),
        ('walk_forward_promotion_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'walk_forward_promotion_gate.py')], False),
        ('promotion_readiness_summary', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_readiness_summary.py'), '--json'], True),
        ('promotion_bottleneck_focus', [str(PY), str(PROJECT_ROOT / 'scripts' / 'promotion_bottleneck_focus.py'), '--json'], True),
        ('new_bot_graduation_gate', [str(PY), str(PROJECT_ROOT / 'scripts' / 'new_bot_graduation_gate.py'), '--json'], False),
        ('leak_overfit_guard', [str(PY), str(PROJECT_ROOT / 'scripts' / 'leak_overfit_guard.py'), '--json'], False),
    ]

    if args.apply_retirement:
        steps.append(
            ('retire_persistent_losers', [
                str(PY),
                str(PROJECT_ROOT / 'scripts' / 'retire_persistent_losers.py'),
                '--lookback-days', '14',
                '--min-fail-days', '5',
                '--min-no-improvement-streak', '2',
                '--max-retire-per-run', '6',
                '--apply',
                '--json',
            ], True)
        )

    if args.run_master_update:
        steps.append(('run_master_bot', [str(PY), str(PROJECT_ROOT / 'scripts' / 'run_master_bot.py'), '--require-canary-gate'], False))

    results = []
    hard_fail = False
    for step, cmd, required_zero in steps:
        row = _run(step, cmd)
        results.append(row)
        if required_zero and row['rc'] != 0:
            hard_fail = True

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': not hard_fail,
        'hard_fail': hard_fail,
        'steps': results,
    }

    out_latest = PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_pipeline_latest.json'
    out_hist = PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_pipeline_history.jsonl'
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    with out_hist.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=True) + '\n')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"promotion_pipeline ok={payload['ok']} hard_fail={payload['hard_fail']}")

    return 0 if payload['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
