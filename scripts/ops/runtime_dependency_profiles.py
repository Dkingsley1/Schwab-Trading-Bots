#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = PROJECT_ROOT / "config" / "requirements.lock.txt"
DEFAULT_OUT = (
    PROJECT_ROOT / "governance" / "health" / "runtime_dependency_profiles_latest.json"
)
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "config" / "runtime_profiles"
PROFILE_ORDER = ("live", "research", "media", "ops")

PROFILE_RULES = {
    "live": {
        "apsw",
        "duckdb",
        "duckdb-engine",
        "numpy",
        "orjson",
        "pandas",
        "polars",
        "polars-runtime-32",
        "pyarrow",
        "redis",
        "requests",
        "schwab-py",
        "sqlalchemy",
        "urllib3",
        "websockets",
        "zstandard",
        "adbc-driver-manager",
        "adbc-driver-sqlite",
    },
    "research": {
        "arch",
        "datasets",
        "duckdb",
        "duckdb-engine",
        "empyrical-reloaded",
        "huggingface_hub",
        "joblib",
        "llvmlite",
        "mlx",
        "mlx-audio",
        "mlx-data",
        "mlx-embedding-models",
        "mlx-embeddings",
        "mlx-lm",
        "mlx-metal",
        "mlx-vlm",
        "numba",
        "numpy",
        "onnx",
        "onnxruntime",
        "optuna",
        "pandas",
        "polars",
        "pyarrow",
        "quantstats",
        "safetensors",
        "scikit-learn",
        "scipy",
        "sentencepiece",
        "statsmodels",
        "sympy",
        "ta",
        "tiktoken",
        "tokenizers",
        "torch",
        "transformers",
        "xgboost",
    },
    "media": {
        "aiofiles",
        "beautifulsoup4",
        "miniaudio",
        "mlx-audio",
        "mlx-whisper",
        "opencv-python",
        "parakeet-mlx",
        "pillow",
        "regex",
        "soupsieve",
    },
    "ops": {
        "apscheduler",
        "click",
        "colorlog",
        "fastapi",
        "flask",
        "httpx",
        "loguru",
        "prometheus_client",
        "psutil",
        "pydantic",
        "pydantic-settings",
        "rich",
        "sentry-sdk",
        "structlog",
        "tenacity",
        "typer",
        "uvicorn",
        "uvloop",
        "watchfiles",
    },
}


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _load_lock_rows(lock_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#") or ("==" not in line):
            continue
        package, version = line.split("==", 1)
        rows.append((_normalize_package_name(package), version.strip()))
    return rows


def _infer_profiles(package: str) -> list[str]:
    matches = [
        profile for profile, packages in PROFILE_RULES.items() if package in packages
    ]
    if matches:
        return matches
    if package.startswith(("mlx", "torch", "transformers", "datasets")):
        return ["research"]
    if package in {"aiohttp", "requests", "httpx", "curl-cffi"}:
        return ["live", "ops"]
    return ["ops"]


def _profile_text(lock_path: Path, profile: str, rows: list[tuple[str, str]]) -> str:
    header = [
        f"# generated from {lock_path.name}",
        f"# profile={profile}",
        "",
    ]
    lines = header + [f"{package}=={version}" for package, version in sorted(rows)]
    return "\n".join(lines).rstrip() + "\n"


def build_payload(
    lock_path: Path, profile_dir: Path, *, apply: bool = False
) -> dict[str, Any]:
    rows = _load_lock_rows(lock_path)
    grouped: dict[str, list[tuple[str, str]]] = {
        profile: [] for profile in PROFILE_ORDER
    }
    package_profiles: dict[str, list[str]] = {}
    overlap_packages: list[str] = []
    for package, version in rows:
        profiles = _infer_profiles(package)
        package_profiles[package] = list(profiles)
        if len(profiles) > 1:
            overlap_packages.append(package)
        for profile in profiles:
            grouped.setdefault(profile, []).append((package, version))

    profile_files: dict[str, str] = {}
    profile_counts: dict[str, int] = {}
    profile_drift: list[dict[str, Any]] = []
    applied_profiles: list[str] = []
    for profile in PROFILE_ORDER:
        target = profile_dir / f"{profile}.lock.txt"
        expected = _profile_text(lock_path, profile, grouped.get(profile, []))
        observed = target.read_text(encoding="utf-8") if target.exists() else ""
        drifted = observed != expected
        if drifted:
            profile_drift.append(
                {
                    "profile": profile,
                    "path": str(target),
                    "reason": "content_mismatch" if target.exists() else "missing",
                }
            )
            if apply:
                profile_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(expected, encoding="utf-8")
                applied_profiles.append(profile)
        profile_files[profile] = str(target)
        profile_counts[profile] = len(grouped.get(profile, []))

    remaining_drift = [] if apply else profile_drift
    profiles_nonempty = all(
        profile_counts.get(profile, 0) > 0 for profile in PROFILE_ORDER
    )
    ok = profiles_nonempty and not remaining_drift
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "ok": ok,
        "overall_status": "ready" if ok else "degraded",
        "apply_requested": apply,
        "applied": bool(applied_profiles),
        "applied_profile_count": len(applied_profiles),
        "applied_profiles": applied_profiles,
        "lock_file": str(lock_path),
        "profile_dir": str(profile_dir),
        "profile_counts": profile_counts,
        "profile_files": profile_files,
        "profile_drift_count": len(remaining_drift),
        "profile_drift": remaining_drift,
        "observed_profile_drift_count": len(profile_drift),
        "overlap_package_count": len(overlap_packages),
        "overlap_packages": sorted(overlap_packages)[:80],
        "package_profiles": package_profiles,
        "recommendations": [
            "promote package upgrades through profile-specific lock reviews before touching the monolithic lock",
            "treat live.lock.txt as the smallest safe blast radius for broker-facing/runtime upgrades",
            "use research.lock.txt and media.lock.txt for exploratory upgrades before merging into live dependencies",
            "run with --apply only inside an explicitly accepted dependency candidate when profile drift is reported",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split the monolithic requirements lock into runtime-specific profile locks."
    )
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write drifted profile locks. Without this flag the command is a read-only audit.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.lock_file).expanduser(),
        Path(args.profile_dir).expanduser(),
        apply=bool(args.apply),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_dependency_profiles "
            f"live={int(payload['profile_counts'].get('live', 0))} "
            f"research={int(payload['profile_counts'].get('research', 0))} "
            f"media={int(payload['profile_counts'].get('media', 0))} "
            f"ops={int(payload['profile_counts'].get('ops', 0))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
