#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "host_self_benchmark_latest.json"
PROTECTED_PREFIXES = ("/Volumes/VIDEO",)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_benchmark_dir(project_root: Path) -> Path:
    health = project_root / "governance" / "health"
    resolved = str(health.resolve())
    if any(resolved == prefix or resolved.startswith(f"{prefix}/") for prefix in PROTECTED_PREFIXES):
        return Path(tempfile.gettempdir())
    health.mkdir(parents=True, exist_ok=True)
    return health


def _jsonl_parse_benchmark(rows: int = 5000) -> dict[str, Any]:
    sample = json.dumps({"ts": "2026-05-20T00:00:00+00:00", "symbol": "BTC", "value": 123.456, "flags": ["a", "b"]})
    lines = [sample for _ in range(max(rows, 1))]
    started = time.perf_counter()
    parsed = 0
    for line in lines:
        json.loads(line)
        parsed += 1
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "rows": parsed,
        "elapsed_seconds": round(elapsed, 6),
        "rows_per_second": round(parsed / elapsed, 3),
    }


def _sqlite_latency_benchmark(directory: Path, rows: int = 1500) -> dict[str, Any]:
    started = time.perf_counter()
    path = directory / f".host_self_benchmark_{int(time.time() * 1000)}.sqlite3"
    insert_elapsed = 0.0
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, payload TEXT)")
            insert_started = time.perf_counter()
            conn.executemany("INSERT INTO bench(payload) VALUES (?)", [(f"row-{idx}",) for idx in range(rows)])
            conn.commit()
            insert_elapsed = time.perf_counter() - insert_started
            conn.execute("SELECT COUNT(*) FROM bench").fetchone()
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                path.with_name(path.name + suffix).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "rows": rows,
        "elapsed_seconds": round(elapsed, 6),
        "insert_commit_seconds": round(insert_elapsed, 6),
        "rows_per_second": round(rows / max(insert_elapsed, 1e-9), 3),
        "benchmark_dir": str(directory),
        "protected_volume_safe": not any(str(directory).startswith(prefix) for prefix in PROTECTED_PREFIXES),
    }


def _storage_write_latency(directory: Path, bytes_to_write: int = 1024 * 1024) -> dict[str, Any]:
    path = directory / f".host_self_benchmark_{int(time.time() * 1000)}.bin"
    payload = b"0" * max(bytes_to_write, 1)
    started = time.perf_counter()
    try:
        path.write_bytes(payload)
        with path.open("rb") as handle:
            handle.read(128)
    finally:
        try:
            path.unlink()
        except Exception:
            pass
    elapsed = max(time.perf_counter() - started, 1e-9)
    mb = bytes_to_write / (1024.0 * 1024.0)
    return {
        "bytes": bytes_to_write,
        "elapsed_seconds": round(elapsed, 6),
        "mb_per_second": round(mb / elapsed, 3),
        "benchmark_dir": str(directory),
        "protected_volume_safe": not any(str(directory).startswith(prefix) for prefix in PROTECTED_PREFIXES),
    }


def _mlx_proxy_benchmark(mlx_router: dict[str, Any]) -> dict[str, Any]:
    caps = mlx_router.get("runtime_caps") if isinstance(mlx_router.get("runtime_caps"), dict) else {}
    return {
        "status": "available" if mlx_router else "missing",
        "primary_mode": caps.get("profile", "unknown"),
        "max_concurrent_jobs": _safe_int(caps.get("max_concurrent_mlx_jobs"), 0),
        "compile_mode": caps.get("compile_mode", "unknown"),
        "note": "Uses the MLX router readiness artifact; avoids launching a heavy GPU job during system pressure.",
    }


def _safe_collector_rate(storage: dict[str, Any], governor: dict[str, Any]) -> dict[str, Any]:
    budgets = governor.get("budgets") if isinstance(governor.get("budgets"), dict) else {}
    collectors = budgets.get("collectors") if isinstance(budgets.get("collectors"), dict) else {}
    enforcement = storage.get("collector_intake_enforcement_audit") if isinstance(storage.get("collector_intake_enforcement_audit"), dict) else {}
    ratio = _safe_float(collectors.get("max_active_ratio") or enforcement.get("active_ratio_limit"), 0.2)
    return {
        "max_active_ratio": round(ratio, 3),
        "source": "autonomic_governor" if collectors else "ingestion_storage_control",
        "safe_to_raise": ratio < 0.35 and str(storage.get("overall_status", "")).lower() not in {"blocked", "critical"},
    }


def _historical_writer_throughput(writer: dict[str, Any]) -> dict[str, Any]:
    effectiveness = writer.get("drain_effectiveness") if isinstance(writer.get("drain_effectiveness"), dict) else {}
    catch_up = writer.get("catch_up_wave_controller") if isinstance(writer.get("catch_up_wave_controller"), dict) else {}
    waves = _safe_int(catch_up.get("waves_run"), 0)
    merged = _safe_int(effectiveness.get("merged_rows"), 0)
    return {
        "source": "latest_writer_cycle",
        "merged_rows_latest_cycle": merged,
        "waves_run_latest_cycle": waves,
        "rows_per_wave_latest_cycle": round(merged / max(waves, 1), 3) if merged else 0.0,
        "status": effectiveness.get("status", "unknown"),
    }


def _limits_from_benchmark(jsonl: dict[str, Any], sqlite: dict[str, Any], write: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
    body = host.get("body_map") if isinstance(host.get("body_map"), dict) else {}
    cpu = body.get("cpu_topology") if isinstance(body.get("cpu_topology"), dict) else {}
    pcores = _safe_int(cpu.get("performance_core_count") or cpu.get("recommended_primary_compute_lanes"), 1)
    sqlite_rps = _safe_float(sqlite.get("rows_per_second"), 0.0)
    jsonl_rps = _safe_float(jsonl.get("rows_per_second"), 0.0)
    selected = min(max(pcores - 1, 1), 7)
    if sqlite_rps < 10000 or jsonl_rps < 50000:
        selected = min(selected, 3)
    return {
        "recommended_p_core_preprocess_workers": selected,
        "recommended_writer_policy": "single_writer_with_bounded_preprocess",
        "recommended_collector_ratio_floor": 0.12,
        "recommended_collector_ratio_ceiling": 0.35 if sqlite_rps < 25000 else 0.55,
        "training_recommendation": "small_targeted_only_until_backlog_green",
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    host = load_json(health / "host_capability_contract_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    mlx_router = load_json(health / "mlx_intelligence_router_latest.json")
    governor = load_json(health / "autonomic_resource_governor_latest.json")
    bench_dir = _safe_benchmark_dir(project_root)
    jsonl = _jsonl_parse_benchmark()
    sqlite = _sqlite_latency_benchmark(bench_dir)
    write = _storage_write_latency(bench_dir)
    limits = _limits_from_benchmark(jsonl, sqlite, write, host)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "benchmark_scope": "lightweight_safe_microbenchmarks",
        "protected_volume_policy": {
            "never_touch": list(PROTECTED_PREFIXES),
            "benchmark_dir": str(bench_dir),
            "protected_volume_safe": not any(str(bench_dir).startswith(prefix) for prefix in PROTECTED_PREFIXES),
        },
        "writer_throughput": _historical_writer_throughput(writer),
        "sqlite_latency": sqlite,
        "jsonl_parse_speed": jsonl,
        "mlx_gpu_speed_proxy": _mlx_proxy_benchmark(mlx_router),
        "storage_write_latency": write,
        "safe_collector_rate": _safe_collector_rate(storage, governor),
        "thermal_swap_behavior": {
            "source": "host_capability_contract",
            "memory": (host.get("body_map") if isinstance(host.get("body_map"), dict) else {}).get("memory", {}),
        },
        "self_tuned_limits": limits,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight self-benchmarks so runtime limits are learned from the host instead of guessed.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "host_self_benchmark "
            f"status={payload['overall_status']} sqlite_rps={payload['sqlite_latency']['rows_per_second']} "
            f"jsonl_rps={payload['jsonl_parse_speed']['rows_per_second']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
