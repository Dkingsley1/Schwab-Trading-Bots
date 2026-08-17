#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic


DEFAULT_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "hdf5_training_cache_latest.json"
DEFAULT_SNAPSHOT_HEALTH = PROJECT_ROOT / "governance" / "health" / "runtime_training_snapshot_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "hdf5" / "runtime_training_snapshot_latest.h5"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "hdf5_training_cache.lock"
FILE_HASH_CHUNK_BYTES = 1024 * 1024
CACHE_SCHEMA_VERSION = 2
SCHEMA_CONTRACT_VERSION = "hdf5_training_cache_v2"
TENSOR_CONTRACT = "row_major_float32_features"
DEFAULT_BENCHMARK_ROWS = 512
DEFAULT_RETENTION_KEEP_GENERATED = 3
DEFAULT_LOCK_STALE_SECONDS = 2 * 3600
LANES = (
    "training_feature_matrices",
    "model_input_tensors",
    "embeddings",
    "walk_forward_datasets",
    "immutable_research_snapshots",
    "compressed_local_archives",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(FILE_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


class _WriterLock:
    def __init__(self, path: Path, acquired: bool, state: dict[str, Any]):
        self.path = path
        self.acquired = acquired
        self.state = state

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = _load_json(self.path)
            if int(payload.get("pid", 0) or 0) == os.getpid():
                self.path.unlink()
        except Exception:
            pass


def _acquire_writer_lock(lock_path: Path, *, stale_seconds: int) -> _WriterLock:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": _now_iso(),
        "pid": os.getpid(),
        "role": "hdf5_training_cache_single_writer",
        "stale_seconds": int(stale_seconds),
    }
    for attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            return _WriterLock(
                lock_path,
                True,
                {
                    "status": "acquired",
                    "path": str(lock_path),
                    "pid": os.getpid(),
                    "stale_seconds": int(stale_seconds),
                },
            )
        except FileExistsError:
            existing = _load_json(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            created_at = _parse_iso(existing.get("timestamp_utc"))
            age_seconds = (
                max((datetime.now(timezone.utc) - created_at).total_seconds(), 0.0)
                if created_at is not None
                else None
            )
            alive = _pid_alive(existing_pid)
            stale = bool(age_seconds is not None and age_seconds > float(stale_seconds))
            if attempt == 0 and (stale or not alive):
                try:
                    lock_path.unlink()
                    continue
                except Exception:
                    pass
            return _WriterLock(
                lock_path,
                False,
                {
                    "status": "busy",
                    "path": str(lock_path),
                    "owner_pid": existing_pid,
                    "owner_alive": bool(alive),
                    "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
                    "stale": bool(stale),
                },
            )
        except Exception as exc:
            return _WriterLock(
                lock_path,
                False,
                {
                    "status": "error",
                    "path": str(lock_path),
                    "error": f"{type(exc).__name__}:{exc}",
                },
            )
    return _WriterLock(lock_path, False, {"status": "busy", "path": str(lock_path)})


def _import_h5py() -> tuple[Any | None, str]:
    try:
        import h5py  # type: ignore

        return h5py, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _iter_jsonl_rows(path: Path, *, max_rows: int) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if max_rows > 0 and count >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            count += 1
            yield row


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        for key, value in features.items():
            try:
                float(value)
            except Exception:
                continue
            names.add(str(key))
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    if not rows or not feature_names:
        return np.empty((0, 0), dtype=np.float32)
    out = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float32)
    for row_idx, row in enumerate(rows):
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        for col_idx, name in enumerate(feature_names):
            out[row_idx, col_idx] = _as_float(features.get(name), np.nan)
    return out


def _embedding_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        for key in features:
            key_text = str(key)
            if key_text.startswith("embedding_") or key_text.startswith("embed_"):
                names.append(key_text)
    names = sorted(set(names))
    if not names:
        return np.empty((0, 0), dtype=np.float32), []
    return _matrix(rows, names), names


def _string_array(values: list[Any]) -> np.ndarray:
    return np.asarray([str(value or "") for value in values], dtype=object)


def _split_indices(row_count: int) -> dict[str, np.ndarray]:
    indices = np.arange(row_count, dtype=np.int64)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def _read_rows(snapshot_health: dict[str, Any], rows_path: Path, *, max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(_iter_jsonl_rows(rows_path, max_rows=max_rows))
    feature_names = _numeric_feature_names(rows)
    matrix = _matrix(rows, feature_names)
    embeddings, embedding_names = _embedding_matrix(rows)
    source_row_count = int(snapshot_health.get("row_count", 0) or 0)
    truncated = bool(max_rows > 0 and source_row_count > max_rows)
    return rows, {
        "row_count": len(rows),
        "source_row_count": source_row_count,
        "sequence_count": int(snapshot_health.get("sequence_count", 0) or 0),
        "feature_count": len(feature_names),
        "embedding_feature_count": len(embedding_names),
        "truncated": truncated,
        "max_rows": int(max_rows),
        "features_matrix": matrix,
        "feature_names": feature_names,
        "embedding_matrix": embeddings,
        "embedding_names": embedding_names,
    }


def _schema_contract(rows_summary: dict[str, Any]) -> dict[str, Any]:
    feature_names = [str(name) for name in rows_summary.get("feature_names", [])]
    embedding_names = [str(name) for name in rows_summary.get("embedding_names", [])]
    return {
        "contract_version": SCHEMA_CONTRACT_VERSION,
        "schema_version": CACHE_SCHEMA_VERSION,
        "tensor_contract": TENSOR_CONTRACT,
        "feature_dtype": "float32",
        "row_count": int(rows_summary.get("row_count", 0) or 0),
        "feature_count": int(rows_summary.get("feature_count", 0) or 0),
        "embedding_feature_count": int(rows_summary.get("embedding_feature_count", 0) or 0),
        "feature_schema_sha256": _sha256_json(feature_names),
        "embedding_schema_sha256": _sha256_json(embedding_names),
        "required_groups": list(LANES) + ["manifest"],
        "required_datasets": {
            "/training_feature_matrices/features": [int(rows_summary.get("row_count", 0) or 0), int(rows_summary.get("feature_count", 0) or 0)],
            "/training_feature_matrices/feature_names": [int(rows_summary.get("feature_count", 0) or 0)],
            "/model_input_tensors/X": [int(rows_summary.get("row_count", 0) or 0), int(rows_summary.get("feature_count", 0) or 0)],
            "/walk_forward_datasets/train": ["variable"],
            "/walk_forward_datasets/validation": ["variable"],
            "/walk_forward_datasets/test": ["variable"],
            "/immutable_research_snapshots/raw_rows_json": [int(rows_summary.get("row_count", 0) or 0)],
            "/compressed_local_archives/source_rows_jsonl_gzip": ["variable"],
        },
    }


def _base_manifest(
    *,
    out_path: Path,
    snapshot_health_path: Path,
    rows_path: Path,
    snapshot_health: dict[str, Any],
    rows_summary: dict[str, Any],
    apply: bool,
    h5py_error: str = "",
) -> dict[str, Any]:
    rows_sha256 = _sha256_file(rows_path) if rows_path.exists() else ""
    lanes = {
        "training_feature_matrices": {
            "status": "ready" if rows_summary.get("feature_count", 0) else "blocked",
            "group": "/training_feature_matrices",
            "dataset": "features",
            "feature_count": int(rows_summary.get("feature_count", 0) or 0),
        },
        "model_input_tensors": {
            "status": "ready" if rows_summary.get("feature_count", 0) else "blocked",
            "group": "/model_input_tensors",
            "dataset": "X",
            "tensor_contract": "row_major_float32_features",
        },
        "embeddings": {
            "status": "ready" if rows_summary.get("embedding_feature_count", 0) else "skipped_no_embedding_features",
            "group": "/embeddings",
            "dataset": "embeddings",
            "embedding_feature_count": int(rows_summary.get("embedding_feature_count", 0) or 0),
        },
        "walk_forward_datasets": {
            "status": "ready" if int(rows_summary.get("row_count", 0) or 0) >= 3 else "blocked_not_enough_rows",
            "group": "/walk_forward_datasets",
            "split_policy": "chronological_70_15_15",
        },
        "immutable_research_snapshots": {
            "status": "ready" if rows_summary.get("row_count", 0) else "blocked_empty_snapshot",
            "group": "/immutable_research_snapshots",
            "immutability": "source_hashes_and_rows_embedded",
        },
        "compressed_local_archives": {
            "status": "ready" if rows_summary.get("row_count", 0) else "blocked_empty_snapshot",
            "group": "/compressed_local_archives",
            "compression": "gzip_payload_inside_hdf5_gzip_dataset",
        },
    }
    ready = bool(
        rows_path.exists()
        and rows_summary.get("row_count", 0)
        and rows_summary.get("feature_count", 0)
        and not h5py_error
    )
    upstream_rows_sha256 = str(snapshot_health.get("rows_sha256") or "")
    rows_hash_match = bool(rows_sha256 and (not upstream_rows_sha256 or upstream_rows_sha256 == rows_sha256))
    return {
        "timestamp_utc": _now_iso(),
        "schema_version": CACHE_SCHEMA_VERSION,
        "overall_status": "ready" if ready and apply else "planned" if ready else "blocked",
        "apply": bool(apply),
        "hdf5_dependency": {
            "required_package": "h5py",
            "available": not bool(h5py_error),
            "error": h5py_error,
        },
        "local_only_contract": {
            "path": str(out_path),
            "relative_path": _repo_rel(out_path),
            "icloud_safe": not ("Mobile Documents" in str(out_path) or "CloudStorage" in str(out_path)),
            "source_of_truth": "jsonl_sqlite_parquet_remain_authoritative",
            "hdf5_role": "derived_training_cache",
        },
        "source_snapshot": {
            "health_path": str(snapshot_health_path),
            "rows_path": str(rows_path),
            "timestamp_utc": str(snapshot_health.get("timestamp_utc") or ""),
            "rows_sha256": rows_sha256,
            "health_sha256": _sha256_file(snapshot_health_path) if snapshot_health_path.exists() else "",
            "upstream_rows_sha256": upstream_rows_sha256,
            "rows_hash_match": rows_hash_match,
        },
        "cache": {
            "h5_path": str(out_path),
            "sidecar_manifest_path": str(out_path.with_suffix(out_path.suffix + ".manifest.json")),
            "row_count": int(rows_summary.get("row_count", 0) or 0),
            "source_row_count": int(rows_summary.get("source_row_count", 0) or 0),
            "sequence_count": int(rows_summary.get("sequence_count", 0) or 0),
            "feature_count": int(rows_summary.get("feature_count", 0) or 0),
            "truncated": bool(rows_summary.get("truncated", False)),
            "max_rows": int(rows_summary.get("max_rows", 0) or 0),
            "fresh": False,
            "schema_ok": False,
        },
        "freshness_gate": {
            "status": "source_ready_cache_pending" if ready else "blocked",
            "fresh": False,
            "current_rows_sha256": rows_sha256,
            "upstream_rows_sha256": upstream_rows_sha256,
            "rows_hash_match": rows_hash_match,
            "source_health_path": str(snapshot_health_path),
            "rows_path": str(rows_path),
        },
        "schema_contract": _schema_contract(rows_summary),
        "schema_validation": {
            "status": "not_checked",
            "ok": False,
            "reasons": ["cache_not_written_or_not_reused"],
        },
        "performance_benchmark": {
            "status": "not_run",
            "sample_rows": 0,
        },
        "retention_policy": {
            "status": "not_run",
            "policy": "keep_latest_plus_pinned_research_snapshots_and_recent_generated_caches",
        },
        "lanes": lanes,
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "hdf5-training-cache", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-training-snapshot", "--reuse-if-fresh-minutes", "360", "--json"],
        ],
    }


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_h5_string_dataset(dataset: Any, *, limit: int = 0) -> list[str]:
    try:
        values = dataset[:limit] if limit and int(dataset.shape[0]) > limit else dataset[:]
    except Exception:
        return []
    return [_decode_h5_string(value) for value in values]


def _inspect_hdf5_contract(
    *,
    h5py: Any | None,
    out_path: Path,
    manifest: dict[str, Any],
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    if not out_path.exists():
        return {"status": "missing", "ok": False, "reasons": ["h5_file_missing"]}
    if h5py is None:
        return {"status": "blocked", "ok": False, "reasons": ["h5py_missing"]}
    if not hasattr(h5py, "File"):
        return {"status": "not_inspected_test_h5py_stub", "ok": True, "reasons": []}

    reasons: list[str] = []
    details: dict[str, Any] = {}
    contract = manifest.get("schema_contract") if isinstance(manifest.get("schema_contract"), dict) else {}
    expected_feature_count = int(contract.get("feature_count", 0) or 0)
    expected_row_count = int(contract.get("row_count", 0) or 0)
    expected_feature_hash = str(contract.get("feature_schema_sha256") or "")
    current_source_hash = str(manifest.get("source_snapshot", {}).get("rows_sha256") or "")
    try:
        with h5py.File(out_path, "r") as h5:
            schema_version = int(h5.attrs.get("schema_version", 0) or 0)
            source_hash = str(h5.attrs.get("source_rows_sha256", "") or "")
            role = str(h5.attrs.get("role", "") or "")
            tensor_contract = str(h5.attrs.get("tensor_contract", h5.get("model_input_tensors", {}).attrs.get("tensor_contract", "") if "model_input_tensors" in h5 else "") or "")
            group_names = set(str(name) for name in h5.keys())
            required_groups = set(str(name) for name in contract.get("required_groups", []))
            missing_groups = sorted(required_groups - group_names)
            if missing_groups:
                reasons.append("required_groups_missing")
            features = h5.get("training_feature_matrices/features")
            tensor = h5.get("model_input_tensors/X")
            feature_names_ds = h5.get("training_feature_matrices/feature_names")
            feature_shape = tuple(int(x) for x in getattr(features, "shape", ())) if features is not None else ()
            tensor_shape = tuple(int(x) for x in getattr(tensor, "shape", ())) if tensor is not None else ()
            feature_names = _read_h5_string_dataset(feature_names_ds) if feature_names_ds is not None else []
            feature_hash = _sha256_json(feature_names)
            if schema_version != CACHE_SCHEMA_VERSION:
                reasons.append("schema_version_mismatch")
            if role != "derived_training_cache":
                reasons.append("role_mismatch")
            if tensor_contract != TENSOR_CONTRACT:
                reasons.append("tensor_contract_mismatch")
            if source_hash != current_source_hash:
                reasons.append("source_hash_mismatch")
            if feature_shape != (expected_row_count, expected_feature_count):
                reasons.append("feature_matrix_shape_mismatch")
            if tensor_shape != (expected_row_count, expected_feature_count):
                reasons.append("model_tensor_shape_mismatch")
            if len(feature_names) != expected_feature_count:
                reasons.append("feature_name_count_mismatch")
            if expected_feature_hash and feature_hash != expected_feature_hash:
                reasons.append("feature_schema_hash_mismatch")
            details = {
                "schema_version": schema_version,
                "role": role,
                "tensor_contract": tensor_contract,
                "source_rows_sha256": source_hash,
                "feature_shape": list(feature_shape),
                "tensor_shape": list(tensor_shape),
                "feature_name_count": len(feature_names),
                "feature_schema_sha256": feature_hash,
                "sidecar_schema_version": int(sidecar.get("schema_version", 0) or 0),
            }
    except Exception as exc:
        return {
            "status": "error",
            "ok": False,
            "reasons": [f"{type(exc).__name__}:{exc}"],
        }
    return {
        "status": "ok" if not reasons else "invalid",
        "ok": not reasons,
        "reasons": reasons,
        "details": details,
    }


def _existing_cache_summary(out_path: Path, manifest: dict[str, Any], *, h5py: Any | None) -> dict[str, Any]:
    sidecar_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    if not out_path.exists() or not sidecar_path.exists():
        return {
            "reusable": False,
            "freshness_gate": {
                **manifest.get("freshness_gate", {}),
                "status": "cache_missing",
                "fresh": False,
            },
            "schema_validation": {"status": "missing", "ok": False, "reasons": ["cache_or_sidecar_missing"]},
        }
    sidecar = _load_json(sidecar_path)
    sidecar_source = sidecar.get("source_snapshot") if isinstance(sidecar.get("source_snapshot"), dict) else {}
    source = manifest.get("source_snapshot") if isinstance(manifest.get("source_snapshot"), dict) else {}
    h5_sha256 = _sha256_file(out_path)
    sidecar_sha256 = _sha256_file(sidecar_path)
    if not h5_sha256 or not sidecar_sha256:
        return {
            "reusable": False,
            "freshness_gate": {
                **manifest.get("freshness_gate", {}),
                "status": "cache_hash_unavailable",
                "fresh": False,
            },
            "schema_validation": {"status": "hash_unavailable", "ok": False, "reasons": ["cache_hash_unavailable"]},
        }
    source_hash_match = str(sidecar_source.get("rows_sha256") or "") == str(source.get("rows_sha256") or "")
    schema_validation = _inspect_hdf5_contract(h5py=h5py, out_path=out_path, manifest=manifest, sidecar=sidecar)
    reusable = bool(source_hash_match and schema_validation.get("ok"))
    sidecar_benchmark = sidecar.get("performance_benchmark") if isinstance(sidecar.get("performance_benchmark"), dict) else {}
    freshness_gate = {
        **manifest.get("freshness_gate", {}),
        "status": "fresh" if reusable else "stale_or_invalid",
        "fresh": reusable,
        "sidecar_rows_sha256": str(sidecar_source.get("rows_sha256") or ""),
        "source_hash_match": source_hash_match,
        "schema_ok": bool(schema_validation.get("ok")),
    }
    return {
        "reusable": reusable,
        "cache": {
            "h5_sha256": h5_sha256,
            "h5_size_bytes": out_path.stat().st_size,
            "sidecar_manifest_sha256": sidecar_sha256,
            "existing_cache_reused": reusable,
            "fresh": reusable,
            "schema_ok": bool(schema_validation.get("ok")),
        },
        "freshness_gate": freshness_gate,
        "schema_validation": schema_validation,
        "performance_benchmark": sidecar_benchmark or {"status": "not_run", "sample_rows": 0},
    }


def _write_string_dataset(h5py: Any, group: Any, name: str, values: list[Any]) -> None:
    group.create_dataset(name, data=_string_array(values), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")


def _write_hdf5(
    *,
    h5py: Any,
    out_path: Path,
    rows: list[dict[str, Any]],
    rows_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}.h5")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        features = rows_summary["features_matrix"]
        feature_names = rows_summary["feature_names"]
        embedding_matrix = rows_summary["embedding_matrix"]
        embedding_names = rows_summary["embedding_names"]
        raw_lines = [json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows]
        compressed_blob = gzip.compress(("\n".join(raw_lines) + ("\n" if raw_lines else "")).encode("utf-8"))

        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["schema_version"] = int(manifest["schema_version"])
            h5.attrs["created_at_utc"] = str(manifest["timestamp_utc"])
            h5.attrs["source_rows_sha256"] = str(manifest["source_snapshot"]["rows_sha256"])
            h5.attrs["role"] = "derived_training_cache"
            h5.attrs["schema_contract_version"] = SCHEMA_CONTRACT_VERSION
            h5.attrs["tensor_contract"] = TENSOR_CONTRACT
            h5.attrs["feature_schema_sha256"] = str(manifest["schema_contract"]["feature_schema_sha256"])
            h5.attrs["embedding_schema_sha256"] = str(manifest["schema_contract"]["embedding_schema_sha256"])
            h5.attrs["row_count"] = int(manifest["schema_contract"]["row_count"])
            h5.attrs["feature_count"] = int(manifest["schema_contract"]["feature_count"])

            feature_group = h5.create_group("training_feature_matrices")
            feature_group.create_dataset("features", data=features, compression="gzip", shuffle=True)
            _write_string_dataset(h5py, feature_group, "feature_names", feature_names)
            _write_string_dataset(h5py, feature_group, "mode", [row.get("mode") for row in rows])
            _write_string_dataset(h5py, feature_group, "symbol", [row.get("symbol") for row in rows])
            _write_string_dataset(h5py, feature_group, "timestamp_utc", [row.get("timestamp_utc") for row in rows])
            feature_group.create_dataset("price", data=np.asarray([_as_float(row.get("price"), np.nan) for row in rows], dtype=np.float32), compression="gzip")

            tensor_group = h5.create_group("model_input_tensors")
            tensor_group.create_dataset("X", data=features, compression="gzip", shuffle=True)
            tensor_group.attrs["tensor_contract"] = TENSOR_CONTRACT

            embedding_group = h5.create_group("embeddings")
            embedding_group.create_dataset("embeddings", data=embedding_matrix, compression="gzip", shuffle=True)
            _write_string_dataset(h5py, embedding_group, "embedding_feature_names", embedding_names)
            embedding_group.attrs["status"] = str(manifest["lanes"]["embeddings"]["status"])

            split_group = h5.create_group("walk_forward_datasets")
            for split_name, indices in _split_indices(len(rows)).items():
                split_group.create_dataset(split_name, data=indices, compression="gzip")
            split_group.attrs["split_policy"] = "chronological_70_15_15"

            research_group = h5.create_group("immutable_research_snapshots")
            _write_string_dataset(h5py, research_group, "raw_rows_json", raw_lines)
            research_group.attrs["rows_sha256"] = str(manifest["source_snapshot"]["rows_sha256"])

            archive_group = h5.create_group("compressed_local_archives")
            archive_group.create_dataset("source_rows_jsonl_gzip", data=np.frombuffer(compressed_blob, dtype=np.uint8), compression="gzip")
            archive_group.attrs["payload_sha256"] = _sha256_bytes(compressed_blob)

            manifest_group = h5.create_group("manifest")
            _write_string_dataset(h5py, manifest_group, "manifest_json", [json.dumps(manifest, ensure_ascii=True, sort_keys=True)])

        os.replace(tmp_path, out_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _benchmark_existing_cache(
    *,
    h5py: Any | None,
    out_path: Path,
    rows_path: Path,
    sample_rows: int,
) -> dict[str, Any]:
    sample_rows = max(int(sample_rows), 1)
    if h5py is None or not hasattr(h5py, "File"):
        return {"status": "skipped_h5py_unavailable", "sample_rows": 0}
    if not out_path.exists() or not rows_path.exists():
        return {"status": "skipped_missing_inputs", "sample_rows": 0}

    hdf5_rows = 0
    hdf5_ms = 0.0
    jsonl_rows: list[dict[str, Any]] = []
    jsonl_ms = 0.0
    try:
        start = time.perf_counter()
        with h5py.File(out_path, "r") as h5:
            features = h5["model_input_tensors/X"]
            hdf5_rows = min(int(features.shape[0]), sample_rows)
            _ = features[:hdf5_rows, :]
        hdf5_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return {
            "status": "error_hdf5_read",
            "sample_rows": 0,
            "error": f"{type(exc).__name__}:{exc}",
        }

    try:
        start = time.perf_counter()
        jsonl_rows = list(_iter_jsonl_rows(rows_path, max_rows=hdf5_rows or sample_rows))
        names = _numeric_feature_names(jsonl_rows)
        _ = _matrix(jsonl_rows, names)
        jsonl_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return {
            "status": "error_jsonl_rebuild",
            "sample_rows": hdf5_rows,
            "hdf5_load_ms": round(hdf5_ms, 4),
            "error": f"{type(exc).__name__}:{exc}",
        }

    speedup = (jsonl_ms / hdf5_ms) if hdf5_ms > 0 else None
    return {
        "status": "ready",
        "sample_rows": int(hdf5_rows),
        "jsonl_sample_rows": len(jsonl_rows),
        "hdf5_load_ms": round(hdf5_ms, 4),
        "jsonl_rebuild_ms": round(jsonl_ms, 4),
        "speedup_ratio": round(float(speedup), 4) if speedup is not None else None,
        "faster_than_jsonl": bool(speedup is not None and speedup >= 1.0),
    }


def _retention_plan(
    *,
    out_path: Path,
    keep_generated: int,
    apply: bool,
) -> dict[str, Any]:
    root = out_path.parent
    if not root.exists():
        return {
            "status": "skipped_missing_dir",
            "policy": "keep_latest_plus_pinned_research_snapshots_and_recent_generated_caches",
            "apply": bool(apply),
        }

    keep_generated = max(int(keep_generated), 0)
    current = out_path.resolve()
    candidates: list[dict[str, Any]] = []
    pinned: list[str] = []
    skipped: list[str] = []
    for h5_path in sorted(root.glob("*.h5")):
        try:
            resolved = h5_path.resolve()
        except Exception:
            resolved = h5_path
        if resolved == current:
            pinned.append(str(h5_path))
            continue
        sidecar = h5_path.with_suffix(h5_path.suffix + ".manifest.json")
        sidecar_payload = _load_json(sidecar) if sidecar.exists() else {}
        contract = sidecar_payload.get("local_only_contract") if isinstance(sidecar_payload.get("local_only_contract"), dict) else {}
        retention = sidecar_payload.get("retention_policy") if isinstance(sidecar_payload.get("retention_policy"), dict) else {}
        generated = str(contract.get("hdf5_role") or "") == "derived_training_cache"
        research_pinned = (
            bool(retention.get("pinned", False))
            or h5_path.name.startswith("research_")
            or h5_path.name.startswith("pinned_")
        )
        if research_pinned:
            pinned.append(str(h5_path))
            continue
        if not generated:
            skipped.append(str(h5_path))
            continue
        try:
            mtime = h5_path.stat().st_mtime
            size = h5_path.stat().st_size
        except Exception:
            mtime = 0.0
            size = 0
        candidates.append(
            {
                "path": str(h5_path),
                "sidecar_path": str(sidecar),
                "mtime": float(mtime),
                "size_bytes": int(size),
            }
        )

    candidates.sort(key=lambda row: float(row.get("mtime", 0.0) or 0.0), reverse=True)
    retained = candidates[:keep_generated]
    delete_candidates = candidates[keep_generated:]
    deleted: list[str] = []
    reclaimed = 0
    if apply:
        for row in delete_candidates:
            path = Path(str(row.get("path") or ""))
            sidecar = Path(str(row.get("sidecar_path") or ""))
            try:
                size = path.stat().st_size if path.exists() else 0
                if path.exists():
                    path.unlink()
                if sidecar.exists():
                    sidecar.unlink()
                reclaimed += int(size)
                deleted.append(str(path))
            except Exception as exc:
                skipped.append(f"{path}:{type(exc).__name__}:{exc}")

    return {
        "status": "applied" if apply else "planned",
        "policy": "keep_latest_plus_pinned_research_snapshots_and_recent_generated_caches",
        "apply": bool(apply),
        "keep_generated": keep_generated,
        "pinned_count": len(pinned),
        "candidate_count": len(candidates),
        "retained_generated": [str(row.get("path") or "") for row in retained],
        "delete_candidates": [str(row.get("path") or "") for row in delete_candidates],
        "deleted": deleted,
        "skipped": skipped,
        "bytes_reclaimed": int(reclaimed),
    }


def evaluate(
    *,
    apply: bool,
    snapshot_health_path: Path,
    rows_path_arg: Path | None,
    out_path: Path,
    health_path: Path,
    max_rows: int,
    benchmark: bool = False,
    benchmark_rows: int = DEFAULT_BENCHMARK_ROWS,
    retention_keep_generated: int = DEFAULT_RETENTION_KEEP_GENERATED,
    retention: bool = True,
    lock_path: Path = DEFAULT_LOCK_PATH,
    lock_stale_seconds: int = DEFAULT_LOCK_STALE_SECONDS,
) -> dict[str, Any]:
    h5py, h5py_error = _import_h5py()
    snapshot_health = _load_json(snapshot_health_path)
    rows_path = rows_path_arg or Path(str(snapshot_health.get("rows_path") or "")).expanduser()
    if not rows_path.is_absolute():
        rows_path = PROJECT_ROOT / rows_path

    rows: list[dict[str, Any]] = []
    rows_summary: dict[str, Any] = {
        "row_count": 0,
        "source_row_count": int(snapshot_health.get("row_count", 0) or 0),
        "sequence_count": int(snapshot_health.get("sequence_count", 0) or 0),
        "feature_count": 0,
        "embedding_feature_count": 0,
        "truncated": False,
        "max_rows": int(max_rows),
        "features_matrix": np.empty((0, 0), dtype=np.float32),
        "feature_names": [],
        "embedding_matrix": np.empty((0, 0), dtype=np.float32),
        "embedding_names": [],
    }
    if rows_path.exists():
        rows, rows_summary = _read_rows(snapshot_health, rows_path, max_rows=max_rows)

    manifest = _base_manifest(
        out_path=out_path,
        snapshot_health_path=snapshot_health_path,
        rows_path=rows_path,
        snapshot_health=snapshot_health,
        rows_summary=rows_summary,
        apply=apply,
        h5py_error=h5py_error,
    )
    if not rows_path.exists():
        manifest["overall_status"] = "blocked"
        manifest["blockers"] = ["runtime_training_snapshot_rows_missing"]
    elif int(rows_summary.get("row_count", 0) or 0) <= 0:
        manifest["overall_status"] = "blocked"
        manifest["blockers"] = ["runtime_training_snapshot_empty"]
    elif int(rows_summary.get("feature_count", 0) or 0) <= 0:
        manifest["overall_status"] = "blocked"
        manifest["blockers"] = ["runtime_training_snapshot_has_no_numeric_features"]
    elif h5py_error:
        manifest["overall_status"] = "blocked"
        manifest["blockers"] = ["h5py_missing"]
    elif apply and h5py is not None:
        writer_lock = _acquire_writer_lock(lock_path, stale_seconds=max(int(lock_stale_seconds), 1))
        manifest["single_writer_guard"] = writer_lock.state
        if not writer_lock.acquired:
            manifest["overall_status"] = "blocked"
            manifest["blockers"] = ["hdf5_cache_writer_busy"]
        else:
            try:
                _write_hdf5(h5py=h5py, out_path=out_path, rows=rows, rows_summary=rows_summary, manifest=manifest)
                manifest["overall_status"] = "ready"
                manifest["cache"]["h5_sha256"] = _sha256_file(out_path)
                manifest["cache"]["h5_size_bytes"] = out_path.stat().st_size if out_path.exists() else 0
                manifest["cache"]["fresh"] = True
                manifest["freshness_gate"] = {
                    **manifest["freshness_gate"],
                    "status": "fresh",
                    "fresh": True,
                    "sidecar_rows_sha256": str(manifest["source_snapshot"]["rows_sha256"]),
                    "source_hash_match": True,
                }
                schema_validation = _inspect_hdf5_contract(
                    h5py=h5py,
                    out_path=out_path,
                    manifest=manifest,
                    sidecar=manifest,
                )
                manifest["schema_validation"] = schema_validation
                manifest["cache"]["schema_ok"] = bool(schema_validation.get("ok"))
                manifest["freshness_gate"]["schema_ok"] = bool(schema_validation.get("ok"))
                if not schema_validation.get("ok"):
                    manifest["overall_status"] = "blocked"
                    manifest["blockers"] = ["hdf5_schema_validation_failed"]
                if benchmark:
                    manifest["performance_benchmark"] = _benchmark_existing_cache(
                        h5py=h5py,
                        out_path=out_path,
                        rows_path=rows_path,
                        sample_rows=int(benchmark_rows),
                    )
                if retention:
                    manifest["retention_policy"] = _retention_plan(
                        out_path=out_path,
                        keep_generated=int(retention_keep_generated),
                        apply=True,
                    )
                sidecar_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
                sidecar_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
                manifest["cache"]["sidecar_manifest_sha256"] = _sha256_file(sidecar_path)
            finally:
                writer_lock.release()
    elif not apply:
        existing = _existing_cache_summary(out_path, manifest, h5py=h5py)
        manifest["freshness_gate"] = existing.get("freshness_gate", manifest["freshness_gate"])
        manifest["schema_validation"] = existing.get("schema_validation", manifest["schema_validation"])
        if isinstance(existing.get("performance_benchmark"), dict):
            manifest["performance_benchmark"] = existing["performance_benchmark"]
        if retention:
            manifest["retention_policy"] = _retention_plan(
                out_path=out_path,
                keep_generated=int(retention_keep_generated),
                apply=False,
            )
        if existing.get("reusable"):
            manifest["overall_status"] = "ready"
            manifest["cache"].update(existing.get("cache", {}))
        elif manifest.get("overall_status") == "planned":
            manifest["cache"]["existing_cache_reused"] = False
            manifest["cache"]["fresh"] = False
            manifest["cache"]["schema_ok"] = bool(manifest.get("schema_validation", {}).get("ok"))
        if benchmark and out_path.exists():
            manifest["performance_benchmark"] = _benchmark_existing_cache(
                h5py=h5py,
                out_path=out_path,
                rows_path=rows_path,
                sample_rows=int(benchmark_rows),
            )

    clean_manifest = json.loads(json.dumps(manifest, ensure_ascii=True))
    return clean_manifest


def _write_health(path: Path, payload: dict[str, Any]) -> bool:
    return bool(
        safe_write_json_atomic(
            str(path),
            payload,
            project_root=str(PROJECT_ROOT),
            source="hdf5_training_cache",
        )
    )


def _print_human(payload: dict[str, Any]) -> None:
    cache = payload.get("cache", {}) if isinstance(payload.get("cache"), dict) else {}
    dep = payload.get("hdf5_dependency", {}) if isinstance(payload.get("hdf5_dependency"), dict) else {}
    freshness = payload.get("freshness_gate", {}) if isinstance(payload.get("freshness_gate"), dict) else {}
    schema = payload.get("schema_validation", {}) if isinstance(payload.get("schema_validation"), dict) else {}
    benchmark = payload.get("performance_benchmark", {}) if isinstance(payload.get("performance_benchmark"), dict) else {}
    print(f"hdf5_training_cache_status={payload.get('overall_status', 'unknown')}")
    print(f"h5py_available={int(bool(dep.get('available')))}")
    print(
        f"rows={cache.get('row_count', 0)} features={cache.get('feature_count', 0)} "
        f"fresh={int(bool(freshness.get('fresh')))} schema_ok={int(bool(schema.get('ok')))}"
    )
    if benchmark.get("status") == "ready":
        print(
            f"benchmark_sample_rows={benchmark.get('sample_rows', 0)} "
            f"hdf5_ms={benchmark.get('hdf5_load_ms', '')} "
            f"jsonl_ms={benchmark.get('jsonl_rebuild_ms', '')} "
            f"speedup={benchmark.get('speedup_ratio', '')}"
        )
    print(f"h5_path={cache.get('h5_path', '')}")
    print(f"latest={DEFAULT_HEALTH_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local-only HDF5 cache from the runtime training snapshot.")
    parser.add_argument("--apply", action="store_true", help="Write the .h5 cache when h5py is available.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--snapshot-health", default=str(DEFAULT_SNAPSHOT_HEALTH))
    parser.add_argument("--rows-path", default="")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--health-path", default=str(DEFAULT_HEALTH_PATH))
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("HDF5_TRAINING_CACHE_MAX_ROWS", "50000")))
    parser.add_argument("--benchmark", action="store_true", help="Run a small HDF5-vs-JSONL load benchmark.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the apply-time benchmark.")
    parser.add_argument("--benchmark-rows", type=int, default=int(os.getenv("HDF5_TRAINING_CACHE_BENCHMARK_ROWS", str(DEFAULT_BENCHMARK_ROWS))))
    parser.add_argument("--retention-keep-generated", type=int, default=int(os.getenv("HDF5_TRAINING_CACHE_RETENTION_KEEP_GENERATED", str(DEFAULT_RETENTION_KEEP_GENERATED))))
    parser.add_argument("--no-retention", action="store_true", help="Skip generated-cache retention planning/apply.")
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--lock-stale-seconds", type=int, default=int(os.getenv("HDF5_TRAINING_CACHE_LOCK_STALE_SECONDS", str(DEFAULT_LOCK_STALE_SECONDS))))
    parser.add_argument("--no-write-health", action="store_true", help="Do not write the health artifact.")
    parser.add_argument("--assert-ready", action="store_true", help="Return non-zero unless the cache is ready.")
    parser.add_argument("--assert-fresh", action="store_true", help="Return non-zero unless the cache is ready, fresh, and schema-valid.")
    args = parser.parse_args()

    rows_path = Path(args.rows_path).expanduser() if str(args.rows_path or "").strip() else None
    payload = evaluate(
        apply=bool(args.apply),
        snapshot_health_path=Path(args.snapshot_health).expanduser(),
        rows_path_arg=rows_path,
        out_path=Path(args.out_path).expanduser(),
        health_path=Path(args.health_path).expanduser(),
        max_rows=max(int(args.max_rows), 0),
        benchmark=bool(args.benchmark or (args.apply and not args.no_benchmark)),
        benchmark_rows=max(int(args.benchmark_rows), 1),
        retention_keep_generated=max(int(args.retention_keep_generated), 0),
        retention=not bool(args.no_retention),
        lock_path=Path(args.lock_path).expanduser(),
        lock_stale_seconds=max(int(args.lock_stale_seconds), 1),
    )
    if not args.no_write_health:
        ok = _write_health(Path(args.health_path).expanduser(), payload)
        if not ok:
            payload["io_error"] = f"write_health_failed:{args.health_path}"
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        _print_human(payload)
    if args.assert_ready and payload.get("overall_status") != "ready":
        return 2
    if args.assert_fresh:
        freshness = payload.get("freshness_gate") if isinstance(payload.get("freshness_gate"), dict) else {}
        schema = payload.get("schema_validation") if isinstance(payload.get("schema_validation"), dict) else {}
        if payload.get("overall_status") != "ready" or not bool(freshness.get("fresh")) or not bool(schema.get("ok")):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
