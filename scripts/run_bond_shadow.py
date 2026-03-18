import argparse
import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
SHADOW_LOOP = PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'
LOAD_RUNTIME_ENV = PROJECT_ROOT / 'scripts' / 'ops' / 'load_runtime_env.sh'

DEFAULT_BOND_SYMBOLS = (
    'TLT,IEF,SHY,TIP,LQD,HYG'
)


def _runtime_profile(simulate: bool) -> str:
    profile = os.getenv('BOT_RUNTIME_PROFILE', 'sim' if simulate else 'live').strip().lower()
    return profile if profile in {'sim', 'live'} else ('sim' if simulate else 'live')


def _bootstrap_runtime_env(base_env: dict[str, str], profile: str) -> dict[str, str]:
    if not LOAD_RUNTIME_ENV.exists():
        return base_env
    source_cmd = (
        f"source {shlex.quote(str(LOAD_RUNTIME_ENV))} {shlex.quote(profile)} --quiet >/dev/null 2>&1 && env -0"
    )
    result = subprocess.run(
        ['/bin/zsh', '-lc', source_cmd],
        cwd=str(PROJECT_ROOT),
        env=base_env,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return base_env
    merged = base_env.copy()
    for chunk in result.stdout.split(b'\0'):
        if not chunk or b'=' not in chunk:
            continue
        key, value = chunk.split(b'=', 1)
        merged[key.decode('utf-8', 'ignore')] = value.decode('utf-8', 'ignore')
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description='Run dedicated bond shadow masterbot profile.')
    parser.add_argument('--broker', default=os.getenv('DATA_BROKER', 'schwab'), choices=['schwab', 'coinbase'])
    parser.add_argument('--simulate', action='store_true', help='Use simulated market feed.')
    parser.add_argument('--symbols', default=os.getenv('BOND_SYMBOLS', DEFAULT_BOND_SYMBOLS))
    parser.add_argument('--interval-seconds', type=int, default=int(os.getenv('BOND_SHADOW_INTERVAL', '90')))
    parser.add_argument('--max-iterations', type=int, default=int(os.getenv('BOND_SHADOW_MAX_ITERS', '0')))
    parser.add_argument('--auto-retrain', action='store_true', default=False)
    args = parser.parse_args()

    if not VENV_PY.exists():
        print(f'ERROR: missing venv python: {VENV_PY}')
        return 2
    if not SHADOW_LOOP.exists():
        print(f'ERROR: missing shadow loop script: {SHADOW_LOOP}')
        return 2

    env = _bootstrap_runtime_env(os.environ.copy(), _runtime_profile(args.simulate))
    env['MARKET_DATA_ONLY'] = '1'
    env['ALLOW_ORDER_EXECUTION'] = '0'
    env['SHADOW_PROFILE'] = 'bond'
    env['SHADOW_DOMAIN'] = 'equities'
    env.setdefault('SHADOW_THRESHOLD_SHIFT', '+0.02')

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        '--broker',
        args.broker,
        '--symbols',
        args.symbols,
        '--interval-seconds',
        str(args.interval_seconds),
        '--max-iterations',
        str(args.max_iterations),
    ]
    if args.simulate:
        cmd.append('--simulate')
    if args.auto_retrain:
        cmd.append('--auto-retrain')

    print('Starting bond shadow profile...')
    print('Symbols:', args.symbols)
    print('Command:', ' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
