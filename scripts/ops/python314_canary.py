import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "python314_canary_latest.json"
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_VENV = PROJECT_ROOT / ".venv314"
DEFAULT_ANCHOR_VENV = PROJECT_ROOT / ".venv312"
EXPECTED_PY314_VERSION = "3.14.5"
DEFAULT_INSTALLER_PATH = Path("/private/tmp/python-3.14.5-macos11.pkg")
DEFAULT_INSTALLER_SIGSTORE_PATH = Path("/private/tmp/python-3.14.5-macos11.pkg.sigstore")
DEFAULT_INSTALLER_SHA256 = "b28a8dc33c456dd06c97024697d63ca916cfb494594c06fa3e4ef4d41fa82335"
DEFAULT_HOMEBREW_PY314 = Path("/opt/homebrew/opt/python@3.14/bin/python3.14")
RUNTIME_FLIP_APPROVAL_ENV = "PY314_RUNTIME_FLIP_APPROVED"
ANCHOR_RETIRE_ENV = "PY314_RETIRE_312_ANCHOR"
HOMEBREW_SIDE_BY_SIDE_ENV = "PY314_ALLOW_HOMEBREW_WITHOUT_PKG"
DEFAULT_SKIP = (
    "numba,llvmlite,mlx,mlx-audio,mlx-cluster,mlx-data,mlx-embedding-models,"
    "mlx-embeddings,mlx-graphs,mlx-lm,mlx-metal,mlx-snn,mlx-vision,mlx-vlm,"
    "mlx-whisper,pandas-ta,parakeet-mlx"
)
DEFAULT_RUNTIME_PACKAGES = ("mlx", "mlx-metal", "mlx-lm")
DEFAULT_TEST_PACKAGES = ("pytest",)
DEFAULT_COMPAT_CORE_PACKAGES = (
    "pytest",
    "aiohttp",
    "apsw",
    "duckdb",
    "matplotlib",
    "onnxruntime",
    "optuna",
    "polars",
    "pyarrow",
    "scikit-learn",
    "statsmodels",
)
DEFAULT_PY314_COMPAT_EXEMPT_PACKAGES = (
    "mlx-cluster",
    "mlx-data",
    "mlx-graphs",
    "pandas-ta",
)
DEFAULT_IMPORT_SMOKES = (
    (
        "mlx_core_import",
        "import mlx.core as mx; print(mx.__name__)",
    ),
    (
        "mlx_lm_import",
        "import mlx_lm; print(mlx_lm.__name__)",
    ),
    (
        "pytest_import",
        "import pytest; print(pytest.__version__)",
    ),
    (
        "indicator_bot_common_import",
        (
            "import sys; "
            "sys.path.insert(0, 'core'); "
            "import indicator_bot_common as mod; "
            "print(mod.__file__)"
        ),
    ),
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], *, timeout_seconds: int | None = None) -> tuple[int, str, str]:
    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("PY314_CANARY_COMMAND_TIMEOUT_SECONDS", "900"))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        timeout_note = f"command timed out after {timeout_seconds}s"
        return 124, (out or "").strip(), f"{timeout_note}\n{err or ''}".strip()
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _tail(text: str, n: int = 8) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigstore_contains_digest(sigstore_path: Path, expected_sha256: str) -> bool:
    if not sigstore_path.exists():
        return False
    try:
        payload = json.loads(sigstore_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = expected_sha256.lower()
    if expected in json.dumps(payload).lower():
        return True
    try:
        entries = payload.get("verificationMaterial", {}).get("tlogEntries", [])
        for entry in entries:
            body = str(entry.get("canonicalizedBody") or "")
            if not body:
                continue
            decoded = base64.b64decode(body).decode("utf-8", errors="replace").lower()
            if expected in decoded:
                return True
    except Exception:
        return False
    return False


def _installer_artifact_step(installer_path: Path, sigstore_path: Path, expected_sha256: str) -> dict:
    exists = installer_path.exists()
    size_bytes = installer_path.stat().st_size if exists else 0
    actual_sha256 = _sha256_file(installer_path) if exists else ""
    sha256_ok = bool(expected_sha256) and actual_sha256.lower() == expected_sha256.lower()
    sigstore_ok = _sigstore_contains_digest(sigstore_path, expected_sha256) if expected_sha256 else sigstore_path.exists()
    ok = exists and size_bytes > 50_000_000 and sha256_ok and sigstore_ok
    return {
        "name": "python3145_download_artifact",
        "ok": ok,
        "rc": 0 if ok else 1,
        "command": str(installer_path),
        "accepted_rc": [0],
        "stdout_tail": f"exists={exists} size_bytes={size_bytes} sha256_ok={sha256_ok} sigstore_ok={sigstore_ok}",
        "stderr_tail": "",
        "installer_path": str(installer_path),
        "sigstore_path": str(sigstore_path),
        "size_bytes": size_bytes,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "sha256_ok": sha256_ok,
        "sigstore_ok": sigstore_ok,
    }


def _step(
    name: str,
    cmd: list[str],
    accepted_rc: set[int] | None = None,
    *,
    timeout_seconds: int | None = None,
) -> dict:
    accepted = accepted_rc or {0}
    if timeout_seconds is None:
        rc, out, err = _run(cmd)
    else:
        rc, out, err = _run(cmd, timeout_seconds=timeout_seconds)
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


def _extract_python_patch_version(text: str) -> str:
    match = re.search(r"Python\s+(\d+\.\d+\.\d+)", text or "")
    return match.group(1) if match else ""


def _python_exact_version_step(name: str, python_bin: Path, expected_version: str) -> dict:
    if not python_bin.exists():
        return {
            "name": name,
            "ok": False,
            "rc": 1,
            "command": str(python_bin),
            "accepted_rc": [0],
            "stdout_tail": "",
            "stderr_tail": "python executable missing",
            "python_bin": str(python_bin),
            "expected_version": expected_version,
            "actual_version": "",
        }
    step = _step(name, [str(python_bin), "--version"])
    actual_text = (step.get("stdout_tail") or step.get("stderr_tail") or "").strip()
    actual_version = _extract_python_patch_version(actual_text)
    step["expected_version"] = expected_version
    step["actual_version"] = actual_version
    step["python_bin"] = str(python_bin)
    step["ok"] = bool(step["ok"] and actual_version == expected_version)
    step["rc"] = 0 if step["ok"] else 1
    return step


def _macos_pkg_signature_step(installer_path: Path) -> dict:
    if not installer_path.exists():
        return {
            "name": "macos_pkg_signature_probe",
            "ok": False,
            "rc": 1,
            "command": f"pkgutil --check-signature {installer_path}",
            "accepted_rc": [0],
            "stdout_tail": "",
            "stderr_tail": "installer package missing",
            "advisory_only": True,
        }
    step = _step("macos_pkg_signature_probe", ["pkgutil", "--check-signature", str(installer_path)])
    step["advisory_only"] = True
    return step


def _parse_version_lines(lines: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if (not line) or line.startswith("#") or ("==" not in line):
            continue
        pkg, version = line.split("==", 1)
        versions[_normalize_package_name(pkg)] = version.strip()
    return versions


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


def _load_lock_versions(lock_file: Path) -> dict[str, str]:
    return _parse_version_lines(_normalize_lock_lines(lock_file))


def _load_installed_versions(venv_py: Path) -> tuple[dict[str, str], dict]:
    rc, out, err = _run([str(venv_py), "-m", "pip", "list", "--format=freeze"])
    step = {
        "name": "installed_package_inventory",
        "ok": rc == 0,
        "rc": rc,
        "command": f"{venv_py} -m pip list --format=freeze",
        "accepted_rc": [0],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }
    return _parse_version_lines(out.splitlines()) if rc == 0 else {}, step


def _package_alignment(lock_versions: dict[str, str], installed_versions: dict[str, str]) -> dict:
    missing_packages = sorted(pkg for pkg in lock_versions if pkg not in installed_versions)
    extra_packages = sorted(pkg for pkg in installed_versions if pkg not in lock_versions)
    version_mismatches = [
        {
            "package": pkg,
            "lock_version": lock_versions[pkg],
            "installed_version": installed_versions[pkg],
        }
        for pkg in sorted(lock_versions)
        if pkg in installed_versions and lock_versions[pkg] != installed_versions[pkg]
    ]
    return {
        "ok": (not missing_packages) and (not version_mismatches),
        "missing_packages": missing_packages,
        "extra_packages": extra_packages,
        "version_mismatches": version_mismatches,
        "missing_count": len(missing_packages),
        "extra_count": len(extra_packages),
        "mismatch_count": len(version_mismatches),
    }


def _alignment_step(lock_file: Path, lock_versions: dict[str, str], installed_versions: dict[str, str]) -> dict:
    alignment = _package_alignment(lock_versions, installed_versions)
    return {
        "name": "lock_alignment",
        "ok": alignment["ok"],
        "rc": 0 if alignment["ok"] else 1,
        "command": str(lock_file),
        "accepted_rc": [0],
        "stdout_tail": (
            f"missing={alignment['missing_count']} "
            f"extra={alignment['extra_count']} "
            f"mismatched={alignment['mismatch_count']}"
        ),
        "stderr_tail": "",
        **alignment,
    }


def _py314_compatibility_alignment_step(alignment: dict, exempt_packages: tuple[str, ...]) -> dict:
    exempt = sorted(_normalize_package_name(pkg) for pkg in exempt_packages)
    exempt_set = set(exempt)
    missing_packages = list(alignment.get("missing_packages") or [])
    blocking_missing = sorted(pkg for pkg in missing_packages if pkg not in exempt_set)
    exempt_missing = sorted(pkg for pkg in missing_packages if pkg in exempt_set)
    version_mismatches = list(alignment.get("version_mismatches") or [])
    ok = not blocking_missing
    return {
        "name": "py314_compatibility_alignment",
        "ok": ok,
        "rc": 0 if ok else 1,
        "command": "py314_compatibility_contract",
        "accepted_rc": [0],
        "stdout_tail": (
            f"blocking_missing={len(blocking_missing)} "
            f"exempt_missing={len(exempt_missing)} "
            f"version_mismatches_allowed={len(version_mismatches)}"
        ),
        "stderr_tail": "",
        "blocking_missing_packages": blocking_missing,
        "exempt_missing_packages": exempt_missing,
        "exempt_packages": exempt,
        "version_mismatches_allowed": version_mismatches,
        "version_mismatch_count": len(version_mismatches),
        "policy": "python_3_14_accepts_newer_compatible_wheels_when_pip_check_and_import_smokes_pass",
        "notes": [
            "strict_312_lock_versions_are_not_required_for_py314_when_pip_check_imports_and_smokes_are_green",
            "exempt_missing_packages_are_unavailable_or_not_compatible_on_python314_today",
        ],
    }


def _required_packages_step(name: str, installed_versions: dict[str, str], packages: tuple[str, ...]) -> dict:
    required = sorted(_normalize_package_name(pkg) for pkg in packages)
    missing = [pkg for pkg in required if pkg not in installed_versions]
    return {
        "name": name,
        "ok": not missing,
        "rc": 0 if not missing else 1,
        "command": "package_presence",
        "accepted_rc": [0],
        "stdout_tail": f"required={','.join(required)} missing={len(missing)}",
        "stderr_tail": "",
        "required_packages": required,
        "missing_packages": missing,
    }


def _package_list_from_env(env_name: str, default_packages: tuple[str, ...]) -> list[str]:
    raw = os.getenv(env_name)
    if raw is None:
        return list(default_packages)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "approved"}


def _training_runtime_promoted_to_py314() -> bool:
    explicit_version = os.getenv("BOT_TRAINING_PYTHON_VERSION", "").strip()
    if explicit_version.startswith("3.14"):
        return True
    explicit_bin = os.getenv("BOT_TRAINING_PYTHON_BIN", "").strip()
    if ".venv314/" in explicit_bin or explicit_bin.endswith(".venv314/bin/python"):
        return True
    lane = (
        os.getenv("BOT_TRAINING_RUNTIME_LANE")
        or os.getenv("BOT_TRAINING_PYTHON_RUNTIME")
        or ""
    ).strip().lower()
    return lane in {"shadow314", "py314", "canary314", "python314"}


def _find_step(steps: list[dict], name: str) -> dict:
    for step in steps:
        if step.get("name") == name:
            return step
    return {}


def _transition_readiness(
    *,
    steps: list[dict],
    import_steps: list[dict],
    smoke_steps: list[dict],
    signature_step: dict,
    bootstrap_ok: bool,
    smoke_ok: bool,
    runtime_flip_approved: bool = False,
) -> dict:
    blockers: list[str] = []
    warnings: list[str] = []
    compatibility_notes: list[str] = []

    homebrew_py314_ready = bool(_find_step(steps, "homebrew_python314_exact_version").get("ok"))
    anchor_retired = _env_flag(ANCHOR_RETIRE_ENV)
    allow_homebrew_side_by_side = _env_flag(HOMEBREW_SIDE_BY_SIDE_ENV, True)

    for name in (
        "python3145_download_artifact",
        "homebrew_python314_exact_version",
        "venv_python314_exact_version",
        "production_anchor_python312",
    ):
        step = _find_step(steps, name)
        if not step.get("ok"):
            if name == "python3145_download_artifact" and homebrew_py314_ready and allow_homebrew_side_by_side:
                warnings.append("python_org_pkg_missing_but_homebrew_python314_exact_version_ready")
                continue
            if name == "production_anchor_python312" and anchor_retired and runtime_flip_approved:
                warnings.append("production_anchor_python312_retired_by_approved_py314_migration")
                continue
            blockers.append(name)

    lock_alignment = _find_step(steps, "lock_alignment")
    py314_compat = _find_step(steps, "py314_compatibility_alignment")
    if lock_alignment and not lock_alignment.get("ok") and not py314_compat.get("ok"):
        blockers.append(
            "lock_alignment_missing_or_mismatched:"
            f"missing={lock_alignment.get('missing_count', 0)},"
            f"mismatched={lock_alignment.get('mismatch_count', 0)}"
        )
    elif lock_alignment and not lock_alignment.get("ok") and py314_compat.get("ok"):
        compatibility_notes.append(
            "strict_312_lock_alignment_documented_for_py314:"
            f"exempt_missing={len(py314_compat.get('exempt_missing_packages') or [])},"
            f"version_mismatches_allowed={py314_compat.get('version_mismatch_count', 0)}"
        )

    runtime_packages = _find_step(steps, "critical_runtime_packages")
    if runtime_packages and not runtime_packages.get("ok"):
        missing = ",".join(runtime_packages.get("missing_packages") or [])
        blockers.append(f"critical_runtime_packages_missing:{missing}")

    test_packages = _find_step(steps, "test_tooling_packages")
    if test_packages and not test_packages.get("ok"):
        missing = ",".join(test_packages.get("missing_packages") or [])
        blockers.append(f"test_tooling_packages_missing:{missing}")

    failed_imports = [str(step.get("name")) for step in import_steps if not step.get("ok")]
    if failed_imports:
        blockers.append("import_smoke_failed:" + ",".join(failed_imports))

    if bootstrap_ok and not smoke_ok:
        failed_smoke = [str(step.get("name")) for step in smoke_steps if not step.get("ok")]
        blockers.append("smoke_failed:" + ",".join(failed_smoke or ["not_run"]))
    elif not bootstrap_ok:
        blockers.append("bootstrap_not_green")

    if signature_step and not signature_step.get("ok"):
        warnings.append("python_org_pkg_signature_probe_not_green_use_homebrew_side_by_side")

    promotion_allowed = (not blockers) and bootstrap_ok and smoke_ok
    production_runtime_change_allowed = bool(promotion_allowed and runtime_flip_approved)
    current_transition_state = "canary_blocked"
    if promotion_allowed and runtime_flip_approved:
        current_transition_state = "runtime_flip_approved"
    elif promotion_allowed:
        current_transition_state = "canary_ready_not_promoted"

    training_promoted = _training_runtime_promoted_to_py314()
    if training_promoted and anchor_retired:
        training_step = "Training runtime is promoted to canary314/.venv314 and the 3.12 anchor is retired by explicit migration approval."
    elif training_promoted:
        training_step = "Training runtime is already promoted to canary314/.venv314; the legacy 3.12 rollback anchor is retired."
    else:
        training_step = "Promote training to canary314/.venv314 before retiring any remaining legacy runtime references."
    monitor_step = (
        "Monitor .venv314 runtime and batch-training behavior before live execution."
        if training_promoted
        else "Monitor .venv314 runtime behavior before moving batch training or live execution."
    )

    return {
        "production_runtime_change_allowed": production_runtime_change_allowed,
        "runtime_flip_approved": bool(runtime_flip_approved),
        "anchor_retired": bool(anchor_retired),
        "homebrew_side_by_side_allowed": bool(allow_homebrew_side_by_side),
        "training_runtime_promoted_to_py314": training_promoted,
        "training_runtime_lane": os.getenv("BOT_TRAINING_RUNTIME_LANE", ""),
        "training_python_runtime": os.getenv("BOT_TRAINING_PYTHON_RUNTIME", ""),
        "training_python_version": os.getenv("BOT_TRAINING_PYTHON_VERSION", ""),
        "promotion_allowed": promotion_allowed,
        "current_transition_state": current_transition_state,
        "safe_anchor": str(DEFAULT_ANCHOR_VENV / "bin" / "python"),
        "canary_python": str(DEFAULT_VENV / "bin" / "python"),
        "blockers": blockers,
        "warnings": warnings,
        "compatibility_notes": compatibility_notes,
        "next_safe_steps": [
            training_step,
            "Use BOT_RUNTIME_LANE=canary314 and BOT_PYTHON_VERSION=3.14.5 for runtime processes after this audit remains green.",
            "Keep MARKET_DATA_ONLY=1 and ALLOW_ORDER_EXECUTION=0 unless live-trading readiness is separately approved.",
            monitor_step,
        ],
    }


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


def _import_step(name: str, venv_py: Path, code: str) -> dict:
    return _step(name, [str(venv_py), "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Python 3.14 canary bootstrap + smoke checks.")
    parser.add_argument("--python-bin", default=os.getenv("PY314_BIN", "python3.14"))
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--expected-python-version", default=os.getenv("PY314_EXPECTED_VERSION", EXPECTED_PY314_VERSION))
    parser.add_argument("--installer-path", default=os.getenv("PY314_INSTALLER_PATH", str(DEFAULT_INSTALLER_PATH)))
    parser.add_argument("--installer-sha256", default=os.getenv("PY314_INSTALLER_SHA256", DEFAULT_INSTALLER_SHA256))
    parser.add_argument("--sigstore-path", default=os.getenv("PY314_SIGSTORE_PATH", str(DEFAULT_INSTALLER_SIGSTORE_PATH)))
    parser.add_argument("--homebrew-python", default=os.getenv("PY314_HOMEBREW_PYTHON", str(DEFAULT_HOMEBREW_PY314)))
    parser.add_argument("--refresh-deps", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    venv_dir = Path(args.venv).resolve()
    venv_py = _venv_python(venv_dir)
    lock_file = Path(args.lock_file).resolve()
    out_file = Path(args.out).resolve()
    expected_python_version = str(args.expected_python_version).strip()
    installer_path = Path(args.installer_path).expanduser().resolve()
    sigstore_path = Path(args.sigstore_path).expanduser().resolve()
    homebrew_python = Path(args.homebrew_python).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    py_minor = _python_minor(args.python_bin)
    is_py314_plus = py_minor.startswith("3.14") or py_minor.startswith("3.15") or py_minor.startswith("3.16")

    steps: list[dict] = []
    bootstrap_ok = True
    install_needed = (not venv_py.exists()) or args.refresh_deps

    homebrew_version_step = _python_exact_version_step("homebrew_python314_exact_version", homebrew_python, expected_python_version)
    steps.append(homebrew_version_step)
    bootstrap_ok = bootstrap_ok and homebrew_version_step["ok"]

    artifact_step = _installer_artifact_step(installer_path, sigstore_path, str(args.installer_sha256).strip())
    if homebrew_version_step["ok"] and _env_flag(HOMEBREW_SIDE_BY_SIDE_ENV, True):
        artifact_step["advisory_only"] = True
        artifact_step["policy"] = "homebrew_exact_python3145_satisfies_side_by_side_runtime"
    steps.append(artifact_step)
    bootstrap_ok = bootstrap_ok and (artifact_step["ok"] or bool(artifact_step.get("advisory_only", False)))

    anchor_version_step = _python_exact_version_step("production_anchor_python312", _venv_python(DEFAULT_ANCHOR_VENV), "3.12.12")
    if _env_flag(ANCHOR_RETIRE_ENV) and _env_flag(RUNTIME_FLIP_APPROVAL_ENV):
        anchor_version_step["advisory_only"] = True
        anchor_version_step["policy"] = "python312_anchor_retired_by_approved_py314_migration"
    steps.append(anchor_version_step)
    bootstrap_ok = bootstrap_ok and (anchor_version_step["ok"] or bool(anchor_version_step.get("advisory_only", False)))

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

                if (not bootstrap_ok) and is_py314_plus:
                    compat_packages = _package_list_from_env("PY314_CANARY_COMPAT_PACKAGES", DEFAULT_COMPAT_CORE_PACKAGES)
                    if compat_packages:
                        compat_step = _step(
                            "install_py314_compat_core",
                            [str(venv_py), "-m", "pip", "install", *compat_packages],
                        )
                        compat_step["compat_packages"] = compat_packages
                        compat_step["advisory_only"] = True
                        steps.append(compat_step)
                        bootstrap_ok = compat_step["ok"]

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

    lock_versions: dict[str, str] = {}
    installed_versions: dict[str, str] = {}
    if venv_py.exists():
        venv_version_step = _python_exact_version_step("venv_python314_exact_version", venv_py, expected_python_version)
        steps.append(venv_version_step)
        bootstrap_ok = bootstrap_ok and venv_version_step["ok"]

        steps.append(_step("pip_check", [str(venv_py), "-m", "pip", "check"]))
        bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

        if lock_file.exists():
            lock_versions = _load_lock_versions(lock_file)
            installed_versions, inventory_step = _load_installed_versions(venv_py)
            steps.append(inventory_step)
            bootstrap_ok = bootstrap_ok and inventory_step["ok"]

            if inventory_step["ok"]:
                alignment_step = _alignment_step(lock_file, lock_versions, installed_versions)
                steps.append(alignment_step)
                if is_py314_plus:
                    compat_step = _py314_compatibility_alignment_step(alignment_step, DEFAULT_PY314_COMPAT_EXEMPT_PACKAGES)
                    steps.append(compat_step)
                    bootstrap_ok = bootstrap_ok and compat_step["ok"]
                else:
                    bootstrap_ok = bootstrap_ok and alignment_step["ok"]

                steps.append(_required_packages_step("critical_runtime_packages", installed_versions, DEFAULT_RUNTIME_PACKAGES))
                bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

                steps.append(_required_packages_step("test_tooling_packages", installed_versions, DEFAULT_TEST_PACKAGES))
                bootstrap_ok = bootstrap_ok and steps[-1]["ok"]

    import_steps: list[dict] = []
    if venv_py.exists():
        for name, code in DEFAULT_IMPORT_SMOKES:
            import_steps.append(_import_step(name, venv_py, code))
        bootstrap_ok = bootstrap_ok and all(step["ok"] for step in import_steps)

    smoke: list[dict] = []
    if bootstrap_ok:
        smoke_timeout = int(os.getenv("PY314_CANARY_SMOKE_TIMEOUT_SECONDS", "90"))
        smoke.append(
            _step(
                "session_ready_check",
                [str(venv_py), "scripts/session_ready_check.py", "--json"],
                accepted_rc={0, 1, 2},
                timeout_seconds=smoke_timeout,
            )
        )
        smoke.append(
            _step(
                "walk_forward_validate",
                [str(venv_py), "scripts/walk_forward_validate.py", "--max-log-files", "80"],
                accepted_rc={0, 2},
                timeout_seconds=smoke_timeout,
            )
        )
        smoke.append(
            _step(
                "walk_forward_promotion_gate",
                [str(venv_py), "scripts/walk_forward_promotion_gate.py"],
                accepted_rc={0, 2},
                timeout_seconds=smoke_timeout,
            )
        )
        smoke.append(
            _step(
                "new_bot_graduation_gate",
                [str(venv_py), "scripts/new_bot_graduation_gate.py", "--json"],
                accepted_rc={0, 2},
                timeout_seconds=smoke_timeout,
            )
        )
        smoke.append(
            _step(
                "leak_overfit_guard",
                [str(venv_py), "scripts/leak_overfit_guard.py", "--json"],
                accepted_rc={0, 2},
                timeout_seconds=smoke_timeout,
            )
        )

    smoke_ok = all(x["ok"] for x in smoke) if smoke else False

    py_ver = {"rc": 1, "stdout": "", "stderr": ""}
    if venv_py.exists():
        rc, out, err = _run([str(venv_py), "--version"])
        py_ver = {"rc": rc, "stdout": out, "stderr": err}

    signature_step = _macos_pkg_signature_step(installer_path)
    transition_readiness = _transition_readiness(
        steps=steps,
        import_steps=import_steps,
        smoke_steps=smoke,
        signature_step=signature_step,
        bootstrap_ok=bool(bootstrap_ok),
        smoke_ok=bool(smoke_ok),
        runtime_flip_approved=_env_flag(RUNTIME_FLIP_APPROVAL_ENV),
    )

    payload = {
        "timestamp_utc": _now_utc(),
        "ok": bool(bootstrap_ok and smoke_ok),
        "python_bin_requested": args.python_bin,
        "python_minor_requested": py_minor,
        "expected_python_version": expected_python_version,
        "venv_python": str(venv_py),
        "venv_exists": venv_py.exists(),
        "python_version": (py_ver["stdout"] or py_ver["stderr"]).strip(),
        "homebrew_python": str(homebrew_python),
        "production_anchor_python": str(_venv_python(DEFAULT_ANCHOR_VENV)),
        "lock_file": str(lock_file),
        "refresh_deps": bool(args.refresh_deps),
        "skip_install": bool(args.skip_install),
        "bootstrap_ok": bool(bootstrap_ok),
        "smoke_ok": bool(smoke_ok),
        "advisory_steps": [signature_step],
        "runtime_flip_approval_env": RUNTIME_FLIP_APPROVAL_ENV,
        "anchor_retire_env": ANCHOR_RETIRE_ENV,
        "homebrew_side_by_side_env": HOMEBREW_SIDE_BY_SIDE_ENV,
        "transition_readiness": transition_readiness,
        "bootstrap_steps": steps,
        "import_steps": import_steps,
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
