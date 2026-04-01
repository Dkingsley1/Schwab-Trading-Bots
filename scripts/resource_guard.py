import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PAGE_SIZE_BYTES = 16384


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


def _parse_swap_usage() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        proc = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
        text = (proc.stdout or "").strip()
        if "used =" in text:
            token = text.split("used =", 1)[1].strip().split()[0]
            suffix = token[-1:].upper()
            value = float(token[:-1] if suffix in {"G", "M", "K"} else token)
            if suffix == "M":
                value /= 1024.0
            elif suffix == "K":
                value /= (1024.0 * 1024.0)
            out["swap_used_gb"] = round(value, 3)
    except Exception:
        pass
    return out


def _parse_vm_stat() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        proc = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=False)
        for raw in (proc.stdout or "").splitlines():
            line = raw.strip()
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            digits = value.strip().rstrip(".").replace(".", "").replace(",", "")
            try:
                count = float(digits)
            except Exception:
                continue
            key = label.strip().lower()
            if key == "pages throttled":
                out["pages_throttled"] = int(count)
            elif key == "pages occupied by compressor":
                out["compressor_gb"] = round((count * _PAGE_SIZE_BYTES) / (1024.0 ** 3), 3)
            elif key == "pages stored in compressor":
                out["compressed_store_gb"] = round((count * _PAGE_SIZE_BYTES) / (1024.0 ** 3), 3)
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
    payload.update(_parse_swap_usage())
    payload.update(_parse_vm_stat())
    return payload


def _memory_pressure_state(snapshot: dict[str, Any]) -> tuple[str, list[str], dict[str, float]]:
    thresholds = {
        "yellow_available_pct": float(os.getenv("RESOURCE_GUARD_MEMORY_YELLOW_AVAILABLE_PCT", "50")),
        "yellow_free_pct": float(os.getenv("RESOURCE_GUARD_MEMORY_YELLOW_FREE_PCT", "8")),
        "yellow_swap_gb": float(os.getenv("RESOURCE_GUARD_MEMORY_YELLOW_SWAP_GB", "12")),
        "yellow_swap_relax_available_pct": float(os.getenv("RESOURCE_GUARD_MEMORY_YELLOW_SWAP_RELAX_AVAILABLE_PCT", "60")),
        "red_available_pct": float(os.getenv("RESOURCE_GUARD_MEMORY_RED_AVAILABLE_PCT", "35")),
        "red_free_pct": float(os.getenv("RESOURCE_GUARD_MEMORY_RED_FREE_PCT", "4")),
        "red_swap_gb": float(os.getenv("RESOURCE_GUARD_MEMORY_RED_SWAP_GB", "18")),
        "red_throttled_pages": float(os.getenv("RESOURCE_GUARD_MEMORY_RED_THROTTLED_PAGES", "1")),
    }

    avail = snapshot.get("memory_available_pct")
    free = snapshot.get("memory_free_pct")
    swap = float(snapshot.get("swap_used_gb", 0.0) or 0.0)
    throttled = float(snapshot.get("pages_throttled", 0) or 0)
    reasons: list[str] = []

    red = False
    if throttled >= thresholds["red_throttled_pages"] and thresholds["red_throttled_pages"] > 0:
        red = True
        reasons.append(f"pages_throttled:{int(throttled)}")
    if avail is not None and float(avail) < thresholds["red_available_pct"]:
        red = True
        reasons.append(f"available_pct:{avail}<{thresholds['red_available_pct']}")
    if free is not None and float(free) < thresholds["red_free_pct"]:
        red = True
        reasons.append(f"free_pct:{free}<{thresholds['red_free_pct']}")
    if swap >= thresholds["red_swap_gb"] and ((avail is None) or float(avail) < thresholds["yellow_swap_relax_available_pct"]):
        red = True
        reasons.append(f"swap_used_gb:{swap}>{thresholds['red_swap_gb']}")
    if red:
        return "red", reasons, thresholds

    yellow = False
    if avail is not None and float(avail) < thresholds["yellow_available_pct"]:
        yellow = True
        reasons.append(f"available_pct:{avail}<{thresholds['yellow_available_pct']}")
    if free is not None and float(free) < thresholds["yellow_free_pct"]:
        yellow = True
        reasons.append(f"free_pct:{free}<{thresholds['yellow_free_pct']}")
    if swap >= thresholds["yellow_swap_gb"] and ((avail is None) or float(avail) < thresholds["yellow_swap_relax_available_pct"]):
        yellow = True
        reasons.append(f"swap_used_gb:{swap}>{thresholds['yellow_swap_gb']}")
    if yellow:
        return "yellow", reasons, thresholds
    return "green", reasons, thresholds


def _memory_pressure_kind(snapshot: dict[str, Any], state: str, reasons: list[str]) -> str:
    if str(state).strip().lower() == "green":
        return "none"
    if not reasons:
        return "unknown"
    if all(str(reason).startswith("swap_used_gb:") for reason in reasons):
        available_pct = snapshot.get("memory_available_pct")
        free_pct = snapshot.get("memory_free_pct")
        throttled = float(snapshot.get("pages_throttled", 0.0) or 0.0)
        if (
            throttled <= 0
            and available_pct is not None
            and float(available_pct) >= float(os.getenv("RESOURCE_GUARD_SWAP_ONLY_HEALTHY_AVAILABLE_PCT", "55"))
            and free_pct is not None
            and float(free_pct) >= float(os.getenv("RESOURCE_GUARD_SWAP_ONLY_HEALTHY_FREE_PCT", "12"))
        ):
            return "swap_only_with_headroom"
        return "swap_only"
    if any(str(reason).startswith("pages_throttled:") for reason in reasons):
        return "throttled"
    if any(str(reason).startswith("available_pct:") or str(reason).startswith("free_pct:") for reason in reasons):
        return "free_or_available"
    return "mixed"


def evaluate(snapshot: dict[str, Any], *, max_load_per_core: float, min_disk_gb: float, min_memory_free_pct: float, max_editing_cpu: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    load_limit = float(max_load_per_core)
    relaxed_load_limit = float(os.getenv("RESOURCE_GUARD_RELAXED_MAX_LOAD_PER_CORE", "2.4"))
    relaxed_min_available_pct = float(os.getenv("RESOURCE_GUARD_RELAXED_MIN_AVAILABLE_PCT", "35"))
    relaxed_max_editing_cpu = float(os.getenv("RESOURCE_GUARD_RELAXED_MAX_EDITING_CPU", str(max_editing_cpu)))
    available_pct = snapshot.get("memory_available_pct")
    free_pct = snapshot.get("memory_free_pct")
    relaxed_memory_pct = available_pct if available_pct is not None else free_pct
    if (
        relaxed_memory_pct is not None
        and float(relaxed_memory_pct) >= relaxed_min_available_pct
        and float(snapshot.get("editing_app_cpu_sum", 0.0)) <= relaxed_max_editing_cpu
    ):
        load_limit = max(load_limit, relaxed_load_limit)

    if float(snapshot.get("load1_per_core", 0.0)) > load_limit:
        reasons.append(f"load1_per_core_high:{snapshot.get('load1_per_core')}>{load_limit}")
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


def evaluate_optional_job(snapshot: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    state, state_reasons, state_thresholds = _memory_pressure_state(snapshot)
    pressure_kind = _memory_pressure_kind(snapshot, state, state_reasons)
    block_on_states = {
        item.strip().lower()
        for item in str(os.getenv("RESOURCE_GUARD_OPTIONAL_BLOCK_ON_MEMORY_STATES", "yellow,red")).split(",")
        if item.strip()
    }
    if state in block_on_states:
        reasons.append(f"memory_pressure_{state}")
        reasons.extend(state_reasons)

    max_load_per_core = float(os.getenv("RESOURCE_GUARD_OPTIONAL_MAX_LOAD_PER_CORE", "2.4"))
    min_disk_gb = float(os.getenv("RESOURCE_GUARD_OPTIONAL_MIN_DISK_GB", "20"))
    max_editing_cpu = float(os.getenv("RESOURCE_GUARD_OPTIONAL_MAX_EDITING_CPU", "300"))

    if float(snapshot.get("load1_per_core", 0.0)) > max_load_per_core:
        reasons.append(f"load1_per_core_high:{snapshot.get('load1_per_core')}>{max_load_per_core}")
    if float(snapshot.get("disk_free_gb", 0.0)) < min_disk_gb:
        reasons.append(f"disk_free_low:{snapshot.get('disk_free_gb')}<{min_disk_gb}")
    if float(snapshot.get("editing_app_cpu_sum", 0.0)) > max_editing_cpu:
        reasons.append(f"editing_apps_hot:{snapshot.get('editing_app_cpu_sum')}>{max_editing_cpu}")

    details = {
        "memory_pressure_state": state,
        "memory_pressure_reasons": state_reasons,
        "memory_pressure_kind": pressure_kind,
        "memory_pressure_thresholds": state_thresholds,
        "optional_job_thresholds": {
            "block_on_memory_states": sorted(block_on_states),
            "max_load_per_core": max_load_per_core,
            "min_disk_gb": min_disk_gb,
            "max_editing_cpu": max_editing_cpu,
        },
    }
    return len(reasons) == 0, reasons, details


def evaluate_refresh_job(snapshot: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    ok, reasons, details = evaluate_optional_job(snapshot)

    thresholds = {
        "allow_memory_states": [
            item.strip().lower()
            for item in str(os.getenv("RESOURCE_GUARD_REFRESH_ALLOW_MEMORY_STATES", "yellow,red")).split(",")
            if item.strip()
        ],
        "min_available_pct": float(os.getenv("RESOURCE_GUARD_REFRESH_MIN_AVAILABLE_PCT", "55")),
        "min_free_pct": float(os.getenv("RESOURCE_GUARD_REFRESH_MIN_FREE_PCT", "12")),
        "max_swap_gb": float(os.getenv("RESOURCE_GUARD_REFRESH_MAX_SWAP_GB", "32")),
        "max_load_per_core": float(os.getenv("RESOURCE_GUARD_REFRESH_MAX_LOAD_PER_CORE", "1.2")),
        "min_disk_gb": float(os.getenv("RESOURCE_GUARD_REFRESH_MIN_DISK_GB", "20")),
        "max_editing_cpu": float(os.getenv("RESOURCE_GUARD_REFRESH_MAX_EDITING_CPU", "220")),
    }

    details["refresh_job_thresholds"] = thresholds
    details["refresh_relax_applied"] = False
    details["refresh_relax_reason"] = ""

    if ok:
        return ok, reasons, details

    state = str(details.get("memory_pressure_state", "") or "").strip().lower()
    state_reasons = [str(item or "") for item in (details.get("memory_pressure_reasons") or [])]
    allow_states = set(thresholds["allow_memory_states"])
    available_pct = snapshot.get("memory_available_pct")
    free_pct = snapshot.get("memory_free_pct")
    swap_used_gb = float(snapshot.get("swap_used_gb", 0.0) or 0.0)
    load1_per_core = float(snapshot.get("load1_per_core", 0.0) or 0.0)
    disk_free_gb = float(snapshot.get("disk_free_gb", 0.0) or 0.0)
    editing_app_cpu_sum = float(snapshot.get("editing_app_cpu_sum", 0.0) or 0.0)
    pages_throttled = float(snapshot.get("pages_throttled", 0.0) or 0.0)

    swap_only_pressure = bool(state_reasons) and all(reason.startswith("swap_used_gb:") for reason in state_reasons)
    effective_available_pct = available_pct if available_pct is not None else free_pct
    effective_free_pct = free_pct if free_pct is not None else available_pct
    healthy_available = effective_available_pct is not None and float(effective_available_pct) >= thresholds["min_available_pct"]
    healthy_free = effective_free_pct is not None and float(effective_free_pct) >= thresholds["min_free_pct"]

    relax_allowed = (
        state in allow_states
        and swap_only_pressure
        and healthy_available
        and healthy_free
        and pages_throttled <= 0
        and swap_used_gb <= thresholds["max_swap_gb"]
        and load1_per_core <= thresholds["max_load_per_core"]
        and disk_free_gb >= thresholds["min_disk_gb"]
        and editing_app_cpu_sum <= thresholds["max_editing_cpu"]
    )

    if not relax_allowed:
        return ok, reasons, details

    filtered_reasons = [
        reason
        for reason in reasons
        if (not reason.startswith("memory_pressure_")) and (not reason.startswith("swap_used_gb:"))
    ]
    details["refresh_relax_applied"] = True
    details["refresh_relax_reason"] = "swap_only_pressure_with_healthy_headroom"
    return len(filtered_reasons) == 0, filtered_reasons, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Resource guard for heavy jobs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--profile", choices=["default", "optional", "refresh"], default="default")
    parser.add_argument("--max-load-per-core", type=float, default=float(os.getenv("RESOURCE_GUARD_MAX_LOAD_PER_CORE", "1.80")))
    parser.add_argument("--min-disk-gb", type=float, default=float(os.getenv("RESOURCE_GUARD_MIN_DISK_GB", "20")))
    parser.add_argument("--min-memory-free-pct", type=float, default=float(os.getenv("RESOURCE_GUARD_MIN_MEMORY_FREE_PCT", "10")))
    parser.add_argument("--max-editing-cpu", type=float, default=float(os.getenv("RESOURCE_GUARD_MAX_EDITING_CPU", "180")))
    parser.add_argument("--emit-path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    snapshot = build_snapshot(project_root)
    memory_state, memory_state_reasons, memory_thresholds = _memory_pressure_state(snapshot)

    details: dict[str, Any] = {
        "memory_pressure_state": memory_state,
        "memory_pressure_reasons": memory_state_reasons,
        "memory_pressure_kind": _memory_pressure_kind(snapshot, memory_state, memory_state_reasons),
        "memory_pressure_thresholds": memory_thresholds,
    }
    if args.profile == "optional":
        ok, reasons, optional_details = evaluate_optional_job(snapshot)
        details.update(optional_details)
    elif args.profile == "refresh":
        ok, reasons, refresh_details = evaluate_refresh_job(snapshot)
        details.update(refresh_details)
    else:
        ok, reasons = evaluate(
            snapshot,
            max_load_per_core=args.max_load_per_core,
            min_disk_gb=args.min_disk_gb,
            min_memory_free_pct=args.min_memory_free_pct,
            max_editing_cpu=args.max_editing_cpu,
        )

    payload = {
        **snapshot,
        "resource_guard_profile": args.profile,
        "resource_guard_ok": ok,
        "resource_guard_reasons": reasons,
        **details,
        "thresholds": {
            "max_load_per_core": args.max_load_per_core,
            "relaxed_max_load_per_core": float(os.getenv("RESOURCE_GUARD_RELAXED_MAX_LOAD_PER_CORE", "2.4")),
            "relaxed_min_available_pct": float(os.getenv("RESOURCE_GUARD_RELAXED_MIN_AVAILABLE_PCT", "35")),
            "relaxed_max_editing_cpu": float(os.getenv("RESOURCE_GUARD_RELAXED_MAX_EDITING_CPU", str(args.max_editing_cpu))),
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
            f"resource_guard_ok={ok} profile={args.profile} memory_pressure_state={payload.get('memory_pressure_state','unknown')} "
            f"load1_per_core={payload['load1_per_core']} "
            f"disk_free_gb={payload['disk_free_gb']} editing_app_cpu_sum={payload['editing_app_cpu_sum']} "
            f"reasons={';'.join(reasons) if reasons else 'none'}"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
