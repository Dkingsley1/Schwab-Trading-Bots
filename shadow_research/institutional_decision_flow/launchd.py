from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any


LABEL = "com.dkingsley.schwabtradingbot.institutional-decision-flow-shadow"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_plist(
    project_root: Path = PROJECT_ROOT,
    *,
    interval_seconds: int = 300,
) -> dict[str, Any]:
    root = project_root.resolve()
    python = root / ".venv314" / "bin" / "python"
    if not python.is_file():
        python = Path(sys.executable).resolve()
    log_dir = root / "governance" / "logs"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "shadow_research.institutional_decision_flow.runner",
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "StartInterval": max(int(interval_seconds), 60),
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 15,
        "ThrottleInterval": 30,
        "StandardOutPath": str(log_dir / "institutional_decision_flow_shadow.out.log"),
        "StandardErrorPath": str(log_dir / "institutional_decision_flow_shadow.err.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "INSTITUTIONAL_DECISION_FLOW_AUTHORITY": "shadow_read_only",
        },
    }


def _destination() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def install(project_root: Path, *, interval_seconds: int) -> Path:
    root = project_root.resolve()
    (root / "governance" / "logs").mkdir(parents=True, exist_ok=True)
    destination = _destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        plistlib.dumps(build_plist(root, interval_seconds=interval_seconds), sort_keys=True)
    )
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(destination)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["launchctl", "bootstrap", domain, str(destination)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"{domain}/{LABEL}"], check=True)
    return destination


def uninstall() -> Path:
    destination = _destination()
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(destination)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if destination.exists():
        destination.unlink()
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the read-only institutional decision-flow LaunchAgent."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--install", action="store_true")
    action.add_argument("--uninstall", action="store_true")
    action.add_argument("--print-plist", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.install:
        destination = install(args.project_root, interval_seconds=args.interval_seconds)
        print(f"installed={destination} label={LABEL}")
        return 0
    if args.uninstall:
        destination = uninstall()
        print(f"uninstalled={destination} label={LABEL}")
        return 0
    print(
        plistlib.dumps(
            build_plist(args.project_root, interval_seconds=args.interval_seconds),
            sort_keys=True,
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
