#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CANDIDATES = PROJECT_ROOT / "config" / "library_candidate_routes_v1.json"
DEFAULT_ACTIVATION_PROFILES = PROJECT_ROOT / "config" / "library_activation_profiles_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "dependency_activation_smoke_latest.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "exports" / "reports" / "operator" / "dependency_activation_smoke_latest.md"

IMPORT_MODULE_OVERRIDES = {
    "pyportfolioopt": "pypfopt",
    "riskfolio-lib": "riskfolio",
    "alibi-detect": "alibi_detect",
    "sqlite-vec": "sqlite_vec",
    "qdrant-client": "qdrant_client",
    "python-louvain": "community",
    "scikit-network": "sknetwork",
    "causal-learn": "causallearn",
    "salib": "SALib",
    "pandas-datareader": "pandas_datareader",
    "edgartools": "edgar",
    "sec-edgar-downloader": "sec_edgar_downloader",
    "sec-cik-mapper": "sec_cik_mapper",
    "great-expectations": "great_expectations",
    "jsonpath-ng": "jsonpath_ng",
    "opentelemetry-api": "opentelemetry",
    "opentelemetry-sdk": "opentelemetry.sdk",
    "opentelemetry-exporter-otlp": "opentelemetry.exporter.otlp",
    "opentelemetry-instrumentation": "opentelemetry.instrumentation",
    "opentelemetry-instrumentation-fastapi": "opentelemetry.instrumentation.fastapi",
    "opentelemetry-instrumentation-httpx": "opentelemetry.instrumentation.httpx",
    "opentelemetry-instrumentation-requests": "opentelemetry.instrumentation.requests",
    "opentelemetry-instrumentation-sqlalchemy": "opentelemetry.instrumentation.sqlalchemy",
    "opentelemetry-instrumentation-sqlite3": "opentelemetry.instrumentation.sqlite3",
    "prometheus-fastapi-instrumentator": "prometheus_fastapi_instrumentator",
    "import-linter": "importlinter",
    "pip-audit": "pip_audit",
    "cyclonedx-bom": "cyclonedx_py",
    "detect-secrets": "detect_secrets",
    "pip-licenses": "piplicenses",
    "pytest-benchmark": "pytest_benchmark",
    "pytest-socket": "pytest_socket",
    "pytest-rerunfailures": "pytest_rerunfailures",
    "hydra-core": "hydra",
    "python-decouple": "decouple",
    "mlx-diffuser": "mlx_diffuser",
}


def _norm_package(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("_", "-")
    return normalized


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = _norm_package(dist.metadata.get("Name", ""))
        if name:
            versions[name] = str(dist.version)
    return versions


def _activation_profiles_for_candidate(candidate: dict[str, Any], activation: dict[str, Any]) -> list[str]:
    package = _norm_package(str(candidate.get("package") or ""))
    lane = str(candidate.get("lane") or "").strip()
    overrides = activation.get("package_profile_overrides")
    normalized_overrides = {_norm_package(str(key)): value for key, value in overrides.items()} if isinstance(overrides, dict) else {}
    if package in normalized_overrides:
        return _ordered_profiles(_string_list(normalized_overrides.get(package)), activation)
    profile_lanes = activation.get("profile_lanes") if isinstance(activation.get("profile_lanes"), dict) else {}
    profiles = [
        str(profile)
        for profile, lanes in profile_lanes.items()
        if lane in _string_list(lanes)
    ]
    if not profiles:
        family = str(candidate.get("runtime_family") or "python").strip().lower()
        profiles = ["research"] if family == "mlx" else ["ops"]
    return _ordered_profiles(profiles, activation)


def _ordered_profiles(profiles: list[str], activation: dict[str, Any]) -> list[str]:
    order = _string_list(activation.get("profile_order")) or ["live", "ops", "research", "media"]
    rank = {profile: index for index, profile in enumerate(order)}
    return sorted(ordered_unique(profiles), key=lambda profile: rank.get(profile, len(rank)))


def _activation_batch(package: str, activation: dict[str, Any]) -> str:
    batches = activation.get("initial_activation_batches")
    if not isinstance(batches, dict):
        return ""
    normalized = _norm_package(package)
    for batch, packages in batches.items():
        if normalized in {_norm_package(item) for item in _string_list(packages)}:
            return str(batch or "").strip()
    return ""


def _module_name(package: str) -> str:
    normalized = _norm_package(package)
    return IMPORT_MODULE_OVERRIDES.get(normalized, normalized.replace("-", "_"))


def _candidate_rows(project_root: Path, candidates_path: Path, activation_path: Path) -> list[dict[str, Any]]:
    candidates_payload = load_json(candidates_path)
    activation = load_json(activation_path)
    raw_rows = candidates_payload.get("candidate_libraries") if isinstance(candidates_payload.get("candidate_libraries"), list) else []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        package = _norm_package(str(raw.get("package") or ""))
        if not package:
            continue
        candidate = dict(raw)
        candidate["package"] = package
        candidate["runtime_family"] = str(raw.get("runtime_family") or "python").strip().lower()
        profiles = _activation_profiles_for_candidate(candidate, activation)
        rows.append(
            {
                "package": package,
                "module": _module_name(package),
                "lane": str(raw.get("lane") or "").strip(),
                "runtime_family": candidate["runtime_family"],
                "priority": str(raw.get("priority") or "medium").strip().lower(),
                "promotion_gate": str(raw.get("promotion_gate") or "compatibility_smoke_then_canary").strip(),
                "activation_profiles": profiles,
                "initial_activation_batch": _activation_batch(package, activation),
            }
        )
    return rows


def _selected_rows(rows: list[dict[str, Any]], *, batch: str = "all", profile: str = "") -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    batch = str(batch or "all").strip()
    profile = str(profile or "").strip()
    for row in rows:
        if batch not in {"", "all"} and str(row.get("initial_activation_batch") or "") != batch:
            continue
        if profile and profile not in _string_list(row.get("activation_profiles")):
            continue
        selected.append(row)
    return selected


def _import_smoke(module: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        importlib.import_module(module)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    return {
        "ok": True,
        "error": "",
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    batch: str = "all",
    profile: str = "",
    import_smoke: bool = False,
    require_installed: bool = False,
    installed_versions: dict[str, str] | None = None,
    candidates_path: Path | None = None,
    activation_path: Path | None = None,
) -> dict[str, Any]:
    candidates_path = candidates_path or project_root / "config" / "library_candidate_routes_v1.json"
    activation_path = activation_path or project_root / "config" / "library_activation_profiles_v1.json"
    installed = {
        _norm_package(name): str(version)
        for name, version in (installed_versions if isinstance(installed_versions, dict) else _load_installed_versions()).items()
        if _norm_package(name)
    }
    rows = _selected_rows(_candidate_rows(project_root, candidates_path, activation_path), batch=batch, profile=profile)
    smoke_rows: list[dict[str, Any]] = []
    for row in rows:
        package = str(row.get("package") or "")
        installed_version = installed.get(package)
        status = "pending_install"
        import_result = {"ok": None, "error": "", "elapsed_ms": 0.0}
        if installed_version:
            status = "installed_pending_import_smoke" if import_smoke else "installed"
            if import_smoke:
                import_result = _import_smoke(str(row.get("module") or ""))
                status = "import_ok" if import_result.get("ok") else "import_failed"
        smoke_rows.append(
            {
                **row,
                "installed_version": installed_version,
                "status": status,
                "import_smoke": import_result,
            }
        )
    failed = [row for row in smoke_rows if str(row.get("status") or "") == "import_failed"]
    pending = [row for row in smoke_rows if str(row.get("status") or "") == "pending_install"]
    installed_count = len(smoke_rows) - len(pending)
    if failed or (require_installed and pending):
        status = "blocked"
    elif pending:
        status = "pending_install"
    else:
        status = "ready"
    activation_batches: dict[str, int] = {}
    activation_profiles: dict[str, int] = {}
    for row in smoke_rows:
        batch_name = str(row.get("initial_activation_batch") or "unbatched")
        activation_batches[batch_name] = activation_batches.get(batch_name, 0) + 1
        for activation_profile in _string_list(row.get("activation_profiles")):
            activation_profiles[activation_profile] = activation_profiles.get(activation_profile, 0) + 1
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status != "blocked",
        "overall_status": status,
        "selection": {
            "batch": batch or "all",
            "profile": profile,
            "import_smoke": import_smoke,
            "require_installed": require_installed,
        },
        "summary": {
            "selected_candidate_count": len(smoke_rows),
            "installed_candidate_count": installed_count,
            "pending_install_count": len(pending),
            "import_failed_count": len(failed),
            "activation_batch_counts": dict(sorted(activation_batches.items())),
            "activation_profile_counts": dict(sorted(activation_profiles.items())),
        },
        "candidate_smoke_rows": smoke_rows,
        "recommended_actions": ordered_unique(
            [
                "install selected activation batch in a sandbox before mutating config/requirements.lock.txt" if pending else "",
                "fix failed imports before any lock promotion" if failed else "",
                "freeze lock and rerun library-utilization-router after successful smoke" if installed_count else "",
            ]
        ),
        "artifact_paths": {
            "json": str(DEFAULT_OUT),
            "markdown": str(DEFAULT_MARKDOWN),
            "candidate_routes": str(candidates_path),
            "activation_profiles": str(activation_path),
        },
        "control_contract": {
            "does_not_install_packages": True,
            "activation_requires_smoke_before_lock_mutation": True,
            "candidate_only_is_not_missing_runtime": True,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    lines = [
        "# Dependency Activation Smoke",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        "",
        "## Selection",
        "",
        f"- Batch: `{selection.get('batch', 'all')}`",
        f"- Profile: `{selection.get('profile', '') or 'any'}`",
        f"- Import smoke: `{selection.get('import_smoke', False)}`",
        "",
        "## Summary",
        "",
        f"- Selected candidates: `{summary.get('selected_candidate_count', 0)}`",
        f"- Installed candidates: `{summary.get('installed_candidate_count', 0)}`",
        f"- Pending install: `{summary.get('pending_install_count', 0)}`",
        f"- Import failures: `{summary.get('import_failed_count', 0)}`",
        f"- Batch counts: `{json.dumps(summary.get('activation_batch_counts') or {}, sort_keys=True)}`",
        "",
        "## Recommended Actions",
        "",
    ]
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], *, out_path: Path = DEFAULT_OUT, markdown_path: Path = DEFAULT_MARKDOWN) -> None:
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke selected staged dependency activation batches without installing packages.")
    parser.add_argument("--batch", default="all", help="Activation batch to smoke, or all.")
    parser.add_argument("--profile", default="", help="Optional activation profile filter: live, ops, research, or media.")
    parser.add_argument("--import-smoke", action="store_true", help="Import installed selected packages.")
    parser.add_argument("--require-installed", action="store_true", help="Block if selected candidates are not installed.")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--exit-zero", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        batch=args.batch,
        profile=args.profile,
        import_smoke=args.import_smoke,
        require_installed=args.require_installed,
    )
    write_outputs(payload, out_path=Path(args.out_file), markdown_path=Path(args.markdown_file))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "dependency_activation_smoke "
            f"status={payload.get('overall_status')} "
            f"selected={summary.get('selected_candidate_count', 0)} "
            f"pending={summary.get('pending_install_count', 0)} "
            f"failed={summary.get('import_failed_count', 0)}"
        )
    return 0 if args.exit_zero or bool(payload.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
