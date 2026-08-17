#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.bot_profitability_scalability import build_control_payload
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from core.bot_profitability_scalability import build_control_payload
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "bot_profitability_scalability_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_profitability_scalability_latest.json"
DEFAULT_MANIFEST_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "bot_organization"
    / "bot_profitability_scalability_latest.json"
)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _tail_plain_rows(path: Path, *, maximum_rows: int, maximum_bytes: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > maximum_bytes:
                handle.seek(max(size - maximum_bytes, 0))
                handle.readline()
            lines = deque(handle, maxlen=max(maximum_rows, 1))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in lines:
        try:
            row = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _tail_gzip_rows(path: Path, *, maximum_rows: int) -> list[dict[str, Any]]:
    lines: deque[bytes] = deque(maxlen=max(maximum_rows, 1))
    try:
        with gzip.open(path, "rb") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
    except (OSError, EOFError):
        return []
    rows: list[dict[str, Any]] = []
    for raw in lines:
        try:
            row = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _source_paths(
    project_root: Path,
    patterns: Iterable[Any],
    *,
    maximum_files: int,
) -> list[Path]:
    discovered: dict[str, Path] = {}
    for pattern in patterns:
        for path in project_root.glob(str(pattern or "")):
            if path.is_file():
                discovered[str(path.resolve())] = path.resolve()
    return sorted(
        discovered.values(),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )[: max(maximum_files, 1)]


def _load_paper_rows(
    project_root: Path,
    policy: dict[str, Any],
    *,
    maximum_files: int | None = None,
    maximum_rows_per_file: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = policy.get("inputs") if isinstance(policy.get("inputs"), dict) else {}
    file_limit = maximum_files or int(inputs.get("maximum_source_files", 24) or 24)
    row_limit = maximum_rows_per_file or int(inputs.get("maximum_rows_per_file", 75000) or 75000)
    byte_limit = int(inputs.get("maximum_bytes_per_plain_file", 67108864) or 67108864)
    paths = _source_paths(
        project_root,
        inputs.get("paper_trade_globs") if isinstance(inputs.get("paper_trade_globs"), list) else [],
        maximum_files=file_limit,
    )
    rows: list[dict[str, Any]] = []
    sources = []
    for path in paths:
        loaded = (
            _tail_gzip_rows(path, maximum_rows=row_limit)
            if path.suffix == ".gz"
            else _tail_plain_rows(path, maximum_rows=row_limit, maximum_bytes=byte_limit)
        )
        rows.extend(loaded)
        sources.append(
            {
                "path": str(path),
                "row_count": len(loaded),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows, {
        "source_file_count": len(paths),
        "parsed_row_count": len(rows),
        "maximum_source_files": file_limit,
        "maximum_rows_per_file": row_limit,
        "sources": sources,
    }


def _candidate_binding(production_excellence: dict[str, Any]) -> dict[str, Any]:
    candidate = (
        production_excellence.get("candidate")
        if isinstance(production_excellence.get("candidate"), dict)
        else {}
    )
    windows = (
        candidate.get("scope_windows_started_utc")
        if isinstance(candidate.get("scope_windows_started_utc"), dict)
        else {}
    )
    relevant = [
        str(windows.get(scope) or "").strip()
        for scope in ("data", "execution", "promotion", "strategy")
        if str(windows.get(scope) or "").strip()
    ]
    cutoff = max(relevant) if relevant else ""
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "generation": int(candidate.get("generation", 0) or 0),
        "candidate_ready": bool(candidate.get("candidate_ready", False)),
        "candidate_drift": bool(candidate.get("candidate_drift", False)),
        "cutoff_utc": cutoff,
        "scope_windows_started_utc": windows,
        "bound": bool(candidate.get("candidate_ready", False) and not candidate.get("candidate_drift", False) and cutoff),
    }


def _runtime_process_inventory(markers: Iterable[Any]) -> dict[str, Any]:
    marker_rows = [str(marker or "").strip() for marker in markers if str(marker or "").strip()]
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"runtime_loop_process_count": 0, "processes": [], "inventory_error": "ps_failed"}
    if proc.returncode != 0:
        return {
            "runtime_loop_process_count": 0,
            "processes": [],
            "inventory_error": (proc.stderr or proc.stdout or "ps_failed").strip()[:500],
        }
    rows = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        if not any(marker in command for marker in marker_rows):
            continue
        rows.append(
            {
                "pid": int(pid_text) if pid_text.isdigit() else 0,
                "marker": next((marker for marker in marker_rows if marker in command), ""),
                "command_excerpt": command[:500],
            }
        )
    return {"runtime_loop_process_count": len(rows), "processes": rows, "inventory_error": ""}


def _runtime_evidence(
    project_root: Path,
    policy: dict[str, Any],
    source_rows: list[dict[str, Any]],
    extraction_scan: dict[str, Any],
) -> dict[str, Any]:
    separation = (
        ((policy.get("scalability") or {}).get("catalog_process_separation") or {})
        if isinstance(policy.get("scalability"), dict)
        else {}
    )
    process_inventory = _runtime_process_inventory(
        separation.get("runtime_process_markers")
        if isinstance(separation.get("runtime_process_markers"), list)
        else []
    )
    checkpoints = sorted(project_root.glob("governance/shadow*/runtime_checkpoint.json"))
    idempotency_registries = sorted(
        project_root.glob("governance/health/order_idempotency_*_latest.json")
    )
    explicit_identity_count = sum(
        1
        for row in source_rows
        if str(row.get("decision_id") or (row.get("metadata") or {}).get("decision_id") or row.get("message_id") or "").strip()
    )
    coverage = explicit_identity_count / max(len(source_rows), 1)
    return {
        **process_inventory,
        "runtime_checkpoint_count": len(checkpoints),
        "runtime_checkpoint_paths": [str(path) for path in checkpoints[:100]],
        "order_idempotency_registry_count": len(idempotency_registries),
        "order_idempotency_registry_paths": [str(path) for path in idempotency_registries[:100]],
        "decision_identity_coverage_ratio": round(coverage, 8),
        "duplicate_source_row_count": int(extraction_scan.get("duplicate_row_count", 0) or 0),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    maximum_files: int | None = None,
    maximum_rows_per_file: int | None = None,
    process_inventory: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    policy = load_json(config_path)
    inputs = policy.get("inputs") if isinstance(policy.get("inputs"), dict) else {}
    artifact_paths = {
        name: _resolve(project_root, raw_path)
        for name, raw_path in inputs.items()
        if name
        not in {
            "paper_trade_globs",
            "maximum_source_files",
            "maximum_rows_per_file",
            "maximum_bytes_per_plain_file",
            "bot_hierarchy",
        }
        and isinstance(raw_path, str)
    }
    artifacts = {
        name: load_json(path)
        for name, path in artifact_paths.items()
    }
    hierarchy_path = _resolve(project_root, inputs.get("bot_hierarchy"))
    hierarchy = load_json(hierarchy_path)
    assignments = [
        row
        for row in hierarchy.get("assignments", [])
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    ]
    production_excellence = artifacts.get("production_excellence", {})
    candidate_binding = _candidate_binding(production_excellence)
    source_rows, source_scan = _load_paper_rows(
        project_root,
        policy,
        maximum_files=maximum_files,
        maximum_rows_per_file=maximum_rows_per_file,
    )
    from core.bot_profitability_scalability import extract_bot_observations

    extraction = extract_bot_observations(
        source_rows,
        known_bot_ids={str(row.get("bot_id") or "") for row in assignments},
        candidate_cutoff_utc=str(candidate_binding.get("cutoff_utc") or ""),
    )
    runtime = _runtime_evidence(
        project_root,
        policy,
        source_rows,
        extraction.get("scan") if isinstance(extraction.get("scan"), dict) else {},
    )
    if process_inventory is not None:
        runtime = {**runtime, **process_inventory}
    health, manifest = build_control_payload(
        policy,
        assignments,
        extraction.get("observations") if isinstance(extraction.get("observations"), list) else [],
        artifacts,
        runtime,
    )
    timestamp = iso_now()
    source_receipt = hashlib.sha256(
        json.dumps(
            {
                "policy_sha256": _sha256(config_path),
                "hierarchy_sha256": _sha256(hierarchy_path),
                "artifact_sha256": {
                    name: _sha256(path) for name, path in sorted(artifact_paths.items())
                },
                "sources": source_scan.get("sources", []),
                "candidate_binding": candidate_binding,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "timestamp_utc": timestamp,
        **manifest,
        "candidate_binding": candidate_binding,
        "source_scan": source_scan,
        "extraction_scan": extraction.get("scan", {}),
        "runtime_evidence": runtime,
        "source_receipt_sha256": source_receipt,
    }
    health = {
        "timestamp_utc": timestamp,
        **health,
        "candidate_binding": candidate_binding,
        "source_scan": {
            "source_file_count": source_scan.get("source_file_count", 0),
            "parsed_row_count": source_scan.get("parsed_row_count", 0),
            "candidate_observation_count": sum(
                1
                for row in extraction.get("observations", [])
                if isinstance(row, dict) and row.get("candidate_bound")
            ),
            "duplicate_row_count": (extraction.get("scan") or {}).get("duplicate_row_count", 0),
            "unattributed_row_count": (extraction.get("scan") or {}).get("unattributed_row_count", 0),
        },
        "runtime_evidence_summary": {
            "runtime_loop_process_count": runtime.get("runtime_loop_process_count", 0),
            "runtime_checkpoint_count": runtime.get("runtime_checkpoint_count", 0),
            "order_idempotency_registry_count": runtime.get("order_idempotency_registry_count", 0),
            "decision_identity_coverage_ratio": runtime.get("decision_identity_coverage_ratio", 0.0),
        },
        "source_receipt_sha256": source_receipt,
        "manifest_path": str(DEFAULT_MANIFEST_OUT_PATH if project_root == PROJECT_ROOT else project_root / "governance" / "bot_organization" / DEFAULT_MANIFEST_OUT_PATH.name),
    }
    return health, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the candidate-bound bot profitability and scalability control."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-rows-per-file", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config or project_root / "config" / DEFAULT_CONFIG_PATH.name
    if not config_path.is_absolute():
        config_path = project_root / config_path
    out_path = args.out_file or project_root / "governance" / "health" / DEFAULT_OUT_PATH.name
    if not out_path.is_absolute():
        out_path = project_root / out_path
    manifest_out = args.manifest_out or (
        project_root / "governance" / "bot_organization" / DEFAULT_MANIFEST_OUT_PATH.name
    )
    if not manifest_out.is_absolute():
        manifest_out = project_root / manifest_out
    health, manifest = build_payload(
        project_root,
        config_path=config_path,
        maximum_files=args.max_files,
        maximum_rows_per_file=args.max_rows_per_file,
    )
    health["manifest_path"] = str(manifest_out)
    write_payload(manifest_out, manifest)
    write_payload(out_path, health)
    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(
            "bot_profitability_scalability_control "
            f"status={health['overall_status']} control_grade={health['control_grade']} "
            f"evidence_grade={health['economic_and_scale_evidence_grade']} "
            f"ranked={health['ranked_bot_count']} selected={health['planned_active_bot_count']}"
        )
    return 0 if health.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
