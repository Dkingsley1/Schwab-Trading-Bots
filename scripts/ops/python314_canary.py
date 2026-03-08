import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "python314_canary_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_VENV = PROJECT_ROOT / ".venv314"
DEFAULT_SKIP = "numba,llvmlite,mlx,mlx-metal,mlx-lm"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _tail(text: str, n: int = 8) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _step(name: str, cmd: list[str], accepted_rc: set[int] | None = None) -> dict:
    accepted = accepted_rc or {0}
    rc, out, err = _run(cmd)
    combined = f"{out}\n{err}".strip()
    hard_fail = any(
        marker in combined
        for marker in (
            "ModuleNotFoundError",
            "ImportError:",
            "No module named",
            "Traceback (most recent call last)",
        )
    )
    ok = (rc in accepted) and (not hard_fail)
    return {
        "name": name,
        "ok": ok,
        "rc": rc,
        "command": " ".join(shlex.quote(x) for x in cmd),
        "accepted_rc": sorted(accepted),
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _python_minor(python_bin: str) -> str:
    rc, out, _ = _run([python_bin, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
    if rc == 0 and out.strip():
        return out.strip()
    return ""


def _normalize_lock_lines(lock_file: Path) -> list[str]:
    out: list[str] = []
    for raw in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#"):
            continue
        out.append(line)
    return out


def _filtered_requirements(lock_file: Path, out_file: Path, skip_packages: set[str], relaxed: bool) -> Path:
    rows: list[str] = []
    skip = {x.strip().lower() for x in skip_packages if x.strip()}
    for line in _normalize_lock_lines(lock_file):
        pkg = line.split("==", 1)[0].strip().lower()
        if pkg in skip:
            continue
        if relaxed:
            rows.append(pkg)
        else:
            rows.append(line)
    dedup = sorted(set(rows))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(dedup) + "\n", encoding="utf-8")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Python 3.14 canary bootstrap + smoke checks.")
    parser.add_argument("--python-bin", default=os.getenv("PY314_BIN", "python3.14"))
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--refresh-deps", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    venv_dir = Path(args.venv).resolve()
    venv_py = _venv_python(venv_dir)
    lock_file = Path(args.lock_file).resolve()
    out_file = Path(args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    py_minor = _python_minor(args.python_bin)
    is_py314_plus = py_minor.startswith("3.14") or py_minor.startswith("3.15") or py_minor.startswith("3.16")

    steps: list[dict] = []
    bootstrap_ok = True
    install_needed = (not venv_py.exists()) or args.refresh_deps

    if not venv_py.exists():
        steps.append(
            _step(
                "create_venv",
                [args.python_bin, "-m", "venv", str(venv_dir)],
                accepted_rc={0},
            )
        )
        bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

    if (not args.skip_install) and bootstrap_ok and (install_needed or (not (venv_dir / ".bootstrapped").exists())):
        steps.append(
            _step(
                "upgrade_installer",
                [str(venv_py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
            )
        )
        bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

        if lock_file.exists() and bootstrap_ok:
            lock_step = _step(
                "install_lockfile",
                [str(venv_py), "-m", "pip", "install", "-r", str(lock_file)],
            )
            steps.append(lock_step)
            bootstrap_ok = bootstrap_ok and lock_step["ok"]

            if (not lock_step["ok"]) and is_py314_plus:
                skip_raw = os.getenv("PY314_CANARY_SKIP_PACKAGES", DEFAULT_SKIP)
                skip_set = {x.strip() for x in skip_raw.split(",") if x.strip()}

                filtered_file = PROJECT_ROOT / "governance" / "health" / "python314_canary_requirements_filtered.txt"
                _filtered_requirements(lock_file, filtered_file, skip_set, relaxed=False)
                filtered_step = _step(
                    "install_lockfile_filtered",
                    [str(venv_py), "-m", "pip", "install", "-r", str(filtered_file)],
                )
                filtered_step["skipped_packages"] = sorted(skip_set)
                filtered_step["filtered_lock_file"] = str(filtered_file)
                steps.append(filtered_step)
                bootstrap_ok = filtered_step["ok"]

                if not bootstrap_ok:
                    relaxed_file = PROJECT_ROOT / "governance" / "health" / "python314_canary_requirements_relaxed.txt"
                    _filtered_requirements(lock_file, relaxed_file, skip_set, relaxed=True)
                    relaxed_step = _step(
                        "install_lockfile_relaxed",
                        [str(venv_py), "-m", "pip", "install", "-r", str(relaxed_file)],
                    )
                    relaxed_step["skipped_packages"] = sorted(skip_set)
                    relaxed_step["relaxed_lock_file"] = str(relaxed_file)
                    steps.append(relaxed_step)
                    bootstrap_ok = relaxed_step["ok"]

        elif not lock_file.exists():
            steps.append(
                {
                    "name": "install_lockfile",
                    "ok": False,
                    "rc": 1,
                    "command": f"missing lock file: {lock_file}",
                    "accepted_rc": [0],
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            bootstrap_ok = False

        if bootstrap_ok:
            (venv_dir / ".bootstrapped").write_text(_now_utc(), encoding="utf-8")

    if venv_py.exists():
        steps.append(_step("pip_check", [str(venv_py), "-m", "pip", "check"]))
        bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

    smoke: list[dict] = []
    if bootstrap_ok:
        smoke.append(_step("session_ready_check", [str(venv_py), "scripts/session_ready_check.py", "--json"], accepted_rc={0, 1, 2}))
        smoke.append(_step("walk_forward_validate", [str(venv_py), "scripts/walk_forward_validate.py"], accepted_rc={0, 2}))
        smoke.append(_step("walk_forward_promotion_gate", [str(venv_py), "scripts/walk_forward_promotion_gate.py"], accepted_rc={0, 2}))
        smoke.append(_step("new_bot_graduation_gate", [str(venv_py), "scripts/new_bot_graduation_gate.py", "--json"], accepted_rc={0, 2}))
        smoke.append(_step("leak_overfit_guard", [str(venv_py), "scripts/leak_overfit_guard.py", "--json"], accepted_rc={0, 2}))

    smoke_ok = all(x["ok"] for x in smoke) if smoke else False

    py_ver = {"rc": 1, "stdout": "", "stderr": ""}
    if venv_py.exists():
        rc, out, err = _run([str(venv_py), "--version"])
        py_ver = {"rc": rc, "stdout": out, "stderr": err}

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(bootstrap_ok and smoke_ok),
        "python_bin_requested": args.python_bin,
        "python_minor_requested": py_minor,
        "venv_python": str(venv_py),
        "venv_exists": venv_py.exists(),
        "python_version": (py_ver["stdout"] or py_ver["stderr"]).strip(),
        "lock_file": str(lock_file),
        "refresh_deps": bool(args.refresh_deps),
        "skip_install": bool(args.skip_install),
        "bootstrap_ok": bool(bootstrap_ok),
        "smoke_ok": bool(smoke_ok),
        "bootstrap_steps": steps,
        "smoke_steps": smoke,
    }

    out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"python314_canary ok={str(payload['ok']).lower()} venv={venv_py}")
        print(f"python_version={payload['python_version'] or 'unknown'}")
        print(f"report={out_file}")

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
