import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_memory_pressure() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        proc = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
        for raw in (proc.stdout or "").splitlines():
            line = raw.strip()
            low = line.lower()
            if "free percentage" in low:
                out["memory_free_pct"] = float(line.split(":", 1)[-1].strip().replace("%", ""))
            elif "available percentage" in low:
                out["memory_available_pct"] = float(line.split(":", 1)[-1].strip().replace("%", ""))
    except Exception:
        pass
    return out


def _heavy_app_cpu_sum() -> float:
    total = 0.0
    try:
        proc = subprocess.run(["/bin/ps", "-axo", "%cpu,command"], capture_output=True, text=True, check=False)
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                cpu = float(parts[0])
            except Exception:
                continue
            cmd = parts[1]
            if "Final Cut Pro" in cmd or "Logic Pro" in cmd:
                total += cpu
    except Exception:
        pass
    return round(total, 2)


def build_snapshot(project_root: Path) -> dict[str, Any]:
    cpu_count = max(os.cpu_count() or 1, 1)
    l1, l5, l15 = os.getloadavg()
    disk = shutil.disk_usage(project_root)
    free_gb = disk.free / (1024.0 ** 3)

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_count": cpu_count,
        "load1": round(l1, 3),
        "load5": round(l5, 3),
        "load15": round(l15, 3),
        "load1_per_core": round(l1 / cpu_count, 3),
        "disk_free_gb": round(free_gb, 2),
        "editing_app_cpu_sum": _heavy_app_cpu_sum(),
    }
    payload.update(_parse_memory_pressure())
    return payload


def evaluate(snapshot: dict[str, Any], *, max_load_per_core: float, min_disk_gb: float, min_memory_free_pct: float, max_editing_cpu: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if float(snapshot.get("load1_per_core", 0.0)) > max_load_per_core:
        reasons.append(f"load1_per_core_high:{snapshot.get('load1_per_core')}>{max_load_per_core}")
    if float(snapshot.get("disk_free_gb", 0.0)) < min_disk_gb:
        reasons.append(f"disk_free_low:{snapshot.get('disk_free_gb')}<{min_disk_gb}")

    free_pct = snapshot.get("memory_free_pct")
    avail_pct = snapshot.get("memory_available_pct")
    memory_pct = free_pct if free_pct is not None else avail_pct
    if memory_pct is not None and float(memory_pct) < min_memory_free_pct:
        reasons.append(f"memory_free_low:{memory_pct}<{min_memory_free_pct}")

    if float(snapshot.get("editing_app_cpu_sum", 0.0)) > max_editing_cpu:
        reasons.append(f"editing_apps_hot:{snapshot.get('editing_app_cpu_sum')}>{max_editing_cpu}")

    return len(reasons) == 0, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Resource guard for heavy jobs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-load-per-core", type=float, default=float(os.getenv("RESOURCE_GUARD_MAX_LOAD_PER_CORE", "1.80")))
    parser.add_argument("--min-disk-gb", type=float, default=float(os.getenv("RESOURCE_GUARD_MIN_DISK_GB", "20")))
    parser.add_argument("--min-memory-free-pct", type=float, default=float(os.getenv("RESOURCE_GUARD_MIN_MEMORY_FREE_PCT", "10")))
    parser.add_argument("--max-editing-cpu", type=float, default=float(os.getenv("RESOURCE_GUARD_MAX_EDITING_CPU", "180")))
    parser.add_argument("--emit-path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    snapshot = build_snapshot(project_root)
    ok, reasons = evaluate(
        snapshot,
        max_load_per_core=args.max_load_per_core,
        min_disk_gb=args.min_disk_gb,
        min_memory_free_pct=args.min_memory_free_pct,
        max_editing_cpu=args.max_editing_cpu,
    )

    payload = {
        **snapshot,
        "resource_guard_ok": ok,
        "resource_guard_reasons": reasons,
        "thresholds": {
            "max_load_per_core": args.max_load_per_core,
            "min_disk_gb": args.min_disk_gb,
            "min_memory_free_pct": args.min_memory_free_pct,
            "max_editing_cpu": args.max_editing_cpu,
        },
    }

    emit = Path(args.emit_path).resolve() if args.emit_path else (project_root / "governance" / "health" / "resource_guard_latest.json")
    emit.parent.mkdir(parents=True, exist_ok=True)
    emit.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"resource_guard_ok={ok} load1_per_core={payload['load1_per_core']} "
            f"disk_free_gb={payload['disk_free_gb']} editing_app_cpu_sum={payload['editing_app_cpu_sum']} "
            f"reasons={';'.join(reasons) if reasons else 'none'}"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
