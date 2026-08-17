#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "host_capability_contract_latest.json"
PROTECTED_VOLUME_DEFAULTS = ("/Volumes/VIDEO",)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _run_capture(command: list[str], *, timeout: float = 3.0) -> str:
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def _sysctl_text(name: str) -> str:
    exe = "/usr/sbin/sysctl"
    if not Path(exe).exists():
        return ""
    return _run_capture([exe, "-n", name])


def _sysctl_int(name: str, default: int = 0) -> int:
    return _safe_int(_sysctl_text(name), default)


def _bytes_to_gb(value: Any) -> float:
    return round(max(_safe_float(value), 0.0) / (1024.0**3), 3)


def _system_profile() -> dict[str, Any]:
    system = platform.system() or "Unknown"
    release = platform.release() or ""
    version = platform.version() or ""
    machine = platform.machine() or ""
    lowered_release = release.lower()
    is_wsl = bool(system == "Linux" and ("microsoft" in lowered_release or "wsl" in lowered_release))
    normalized = "wsl" if is_wsl else system.lower()
    return {
        "os": normalized,
        "system": system,
        "release": release,
        "version": version,
        "machine": machine,
        "platform": platform.platform(),
        "is_wsl": is_wsl,
        "python": sys.version.split()[0],
    }


def _cpu_topology(system: dict[str, Any], apple_profile: dict[str, Any]) -> dict[str, Any]:
    logical = os.cpu_count() or 0
    physical = logical
    performance = 0
    efficiency = 0
    chip = ""
    topology = "symmetric"
    hard_affinity_supported = False
    core_allocator = "os_scheduler"

    if system["system"] == "Darwin":
        logical = _sysctl_int("hw.logicalcpu", logical)
        physical = _sysctl_int("hw.physicalcpu", logical)
        chip = _sysctl_text("machdep.cpu.brand_string") or str(apple_profile.get("hardware", {}).get("chip") or "")
        perf0 = _sysctl_int("hw.perflevel0.physicalcpu", 0)
        perf1 = _sysctl_int("hw.perflevel1.physicalcpu", 0)
        if perf0 or perf1:
            performance = max(perf0, perf1)
            efficiency = min(perf0, perf1)
        else:
            performance = _safe_int(apple_profile.get("performance_core_contract", {}).get("primary_performance_core_budget"), 0)
            efficiency = _safe_int(apple_profile.get("performance_core_contract", {}).get("efficiency_spillover_core_budget"), 0)
        if performance or efficiency:
            topology = "apple_silicon_p_e"
        hard_affinity_supported = False
        core_allocator = "darwin_qos_nice_taskpolicy"
    elif system["system"] == "Linux":
        hard_affinity_supported = shutil.which("taskset") is not None
        core_allocator = "taskset_cgroups_or_systemd"
        physical = logical
        chip = _linux_cpu_model()
    elif system["system"] == "Windows":
        hard_affinity_supported = True
        core_allocator = "windows_processor_affinity"
        chip = platform.processor() or ""
    else:
        chip = platform.processor() or ""

    return {
        "topology": topology,
        "chip": chip,
        "logical_core_count": logical,
        "physical_core_count": physical,
        "performance_core_count": performance,
        "efficiency_core_count": efficiency,
        "hard_affinity_supported": hard_affinity_supported,
        "core_allocator": core_allocator,
        "recommended_primary_compute_lanes": max(performance or physical or logical, 1),
    }


def _linux_cpu_model() -> str:
    try:
        for raw in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if raw.lower().startswith("model name"):
                return raw.split(":", 1)[1].strip()
    except Exception:
        return ""
    return ""


def _linux_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for raw in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                out[key.strip()] = int(digits) * 1024
    except Exception:
        return {}
    return out


def _memory_profile(system: dict[str, Any], runtime: dict[str, Any], memory_efficiency: dict[str, Any]) -> dict[str, Any]:
    total_bytes = 0
    swap_gb = 0.0
    compression = "unknown"
    pressure = str(runtime.get("memory_pressure_level") or "").strip().lower() or "unknown"
    if system["system"] == "Darwin":
        total_bytes = _sysctl_int("hw.memsize", 0)
        swap_text = _sysctl_text("vm.swapusage")
        match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", swap_text, re.IGNORECASE)
        if match:
            factor = {"M": 1 / 1024.0, "G": 1.0, "T": 1024.0}.get(match.group(2).upper(), 1.0)
            swap_gb = round(_safe_float(match.group(1)) * factor, 3)
        compression = "vm_compressor"
    elif system["system"] == "Linux":
        info = _linux_meminfo()
        total_bytes = info.get("MemTotal", 0)
        swap_total = info.get("SwapTotal", 0)
        swap_free = info.get("SwapFree", 0)
        swap_gb = _bytes_to_gb(max(swap_total - swap_free, 0))
        compression = "zram_or_swap" if Path("/sys/block/zram0").exists() else "swap"
    memory_snapshot = memory_efficiency.get("memory_snapshot") if isinstance(memory_efficiency.get("memory_snapshot"), dict) else {}
    return {
        "memory_gb": _bytes_to_gb(total_bytes),
        "swap_used_gb": swap_gb,
        "compression_behavior": compression,
        "pressure_level": pressure,
        "memory_snapshot": memory_snapshot,
    }


def _gpu_stack(system: dict[str, Any], mlx_runtime: dict[str, Any]) -> dict[str, Any]:
    runtime = mlx_runtime.get("runtime") if isinstance(mlx_runtime.get("runtime"), dict) else {}
    mlx_available = bool(importlib.util.find_spec("mlx") is not None or runtime.get("metal_available"))
    metal_available = bool(system["system"] == "Darwin" and (runtime.get("metal_available") is not False))
    cuda_available = shutil.which("nvidia-smi") is not None
    rocm_available = shutil.which("rocm-smi") is not None
    stacks = ordered_unique(
        [
            "MLX" if mlx_available else "",
            "Metal" if metal_available else "",
            "CUDA" if cuda_available else "",
            "ROCm" if rocm_available else "",
        ]
    )
    return {
        "available_stacks": stacks or ["none"],
        "mlx_available": mlx_available,
        "metal_available": metal_available,
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "compile_available": bool(runtime.get("compile_available", False)),
        "primary_gpu_stack": stacks[0] if stacks else "none",
    }


def _mount_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mount_text = _run_capture(["/sbin/mount"]) or _run_capture(["mount"])
    for raw in mount_text.splitlines():
        line = raw.strip()
        if " on " not in line:
            continue
        device, rest = line.split(" on ", 1)
        if " (" in rest:
            mountpoint, meta = rest.split(" (", 1)
            filesystem = meta.split(",", 1)[0].rstrip(")")
        else:
            parts = rest.split()
            mountpoint = parts[0] if parts else rest
            filesystem = ""
        rows.append({"device": device, "mountpoint": mountpoint, "filesystem": filesystem})
    return rows


def _df_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _run_capture(["df", "-kP"])
    for raw in text.splitlines()[1:]:
        parts = raw.split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "filesystem": parts[0],
                "size_gb": round(_safe_float(parts[1]) / 1024.0 / 1024.0, 3),
                "used_gb": round(_safe_float(parts[2]) / 1024.0 / 1024.0, 3),
                "free_gb": round(_safe_float(parts[3]) / 1024.0 / 1024.0, 3),
                "capacity": parts[4],
                "mountpoint": parts[5],
            }
        )
    return rows


def _storage_layout() -> dict[str, Any]:
    mounts = _mount_rows()
    df = _df_rows()
    fs_by_mount = {str(row.get("mountpoint")): str(row.get("filesystem") or "") for row in mounts}
    protected = [*PROTECTED_VOLUME_DEFAULTS, *os.getenv("BOT_PROTECTED_VOLUMES", "").split(",")]
    protected = ordered_unique(protected)
    project_realpath = str(PROJECT_ROOT.resolve())
    bot_logs_mount = os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")
    storage_rows: list[dict[str, Any]] = []
    for row in df:
        mountpoint = str(row.get("mountpoint") or "")
        storage_rows.append(
            {
                **row,
                "filesystem_type": fs_by_mount.get(mountpoint, ""),
                "protected": any(mountpoint == volume or mountpoint.startswith(f"{volume}/") for volume in protected),
                "is_project_mount": project_realpath.startswith(mountpoint.rstrip("/") + "/") or project_realpath == mountpoint,
                "is_bot_logs_mount": mountpoint == bot_logs_mount,
            }
        )
    return {
        "project_root": str(PROJECT_ROOT),
        "project_realpath": project_realpath,
        "bot_logs_external_mount": bot_logs_mount,
        "storage_rows": storage_rows,
        "protected_volumes": protected,
        "denylist_rules": [{"path": volume, "policy": "never_write_or_prune"} for volume in protected],
    }


def _launch_system(system: dict[str, Any]) -> dict[str, Any]:
    os_name = system["os"]
    if os_name == "darwin":
        primary = "launchd"
    elif os_name in {"linux", "wsl"}:
        primary = "systemd" if Path("/run/systemd/system").exists() else "cron"
    elif os_name == "windows":
        primary = "task_scheduler"
    else:
        primary = "unknown"
    return {
        "primary": primary,
        "supports_launchd": os_name == "darwin",
        "supports_systemd": primary == "systemd",
        "supports_cron": shutil.which("cron") is not None or shutil.which("crontab") is not None,
        "supports_task_scheduler": os_name == "windows",
    }


def _foreground_context(computer_task: dict[str, Any], resource_guard: dict[str, Any]) -> dict[str, Any]:
    session = computer_task.get("session_context") if isinstance(computer_task.get("session_context"), dict) else {}
    return {
        "source": "computer_task_intelligence",
        "creative_kind": session.get("creative_kind") or resource_guard.get("creative_session_kind") or "none",
        "creative_level": session.get("creative_level") or resource_guard.get("creative_session_level") or "none",
        "co_running_level": session.get("co_running_level") or resource_guard.get("co_running_session_level") or "none",
        "open_apps": session.get("open_apps") if isinstance(session.get("open_apps"), list) else resource_guard.get("co_running_apps", []),
        "user_coexistent_required": bool(session.get("cotenant_active") or session.get("creative_active") or resource_guard.get("co_running_session_level") not in {None, "", "none"}),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    apple_profile = load_json(health / "apple_silicon_profile_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    mlx_runtime = load_json(health / "mlx_runtime_audit_latest.json")
    memory_efficiency = load_json(health / "memory_efficiency_control_latest.json")
    computer_task = load_json(health / "computer_task_intelligence_latest.json")
    resource_guard = load_json(health / "resource_guard_latest.json")

    system = _system_profile()
    cpu = _cpu_topology(system, apple_profile)
    memory = _memory_profile(system, runtime, memory_efficiency)
    gpu = _gpu_stack(system, mlx_runtime)
    storage = _storage_layout()
    launch = _launch_system(system)
    foreground = _foreground_context(computer_task, resource_guard)

    capabilities = {
        "process_priority": system["os"] in {"darwin", "linux", "wsl"},
        "service_supervision": str(launch.get("primary") or "unknown") != "unknown",
        "memory_pressure_probe": system["os"] in {"darwin", "linux", "wsl"},
        "thermal_probe": system["os"] in {"darwin", "linux"},
        "hard_cpu_affinity": bool(cpu.get("hard_affinity_supported", False)),
        "qos_cpu_steering": system["os"] == "darwin",
        "gpu_compute": str(gpu.get("primary_gpu_stack") or "none") != "none",
        "storage_mount_awareness": bool(storage.get("storage_rows", [])),
    }
    limitations = ordered_unique(
        [
            "darwin_has_no_reliable_per_process_p_core_hard_affinity" if system["os"] == "darwin" else "",
            "wsl_launch_and_gpu_stack_need_explicit_binding" if system["os"] == "wsl" else "",
            "no_gpu_stack_detected" if gpu["primary_gpu_stack"] == "none" else "",
        ]
    )
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready" if capabilities["service_supervision"] else "advisory",
        "body_map": {
            "system": system,
            "cpu_topology": cpu,
            "memory": memory,
            "gpu_stack": gpu,
            "storage_layout": storage,
            "launch_system": launch,
            "foreground_apps_and_user_activity": foreground,
            "protected_volume_policy": {
                "enabled": True,
                "denylist": storage["protected_volumes"],
                "never_touch_video_volume": "/Volumes/VIDEO" in storage["protected_volumes"],
            },
        },
        "capabilities": capabilities,
        "limitations": limitations,
        "source_artifacts": {
            "apple_silicon_profile": str(health / "apple_silicon_profile_latest.json"),
            "runtime_throttle": str(health / "runtime_throttle_control_latest.json"),
            "mlx_runtime_audit": str(health / "mlx_runtime_audit_latest.json"),
            "computer_task_intelligence": str(health / "computer_task_intelligence_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the host capability contract body map for portable runtime governance.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    payload = build_payload(PROJECT_ROOT)
    write_payload(Path(args.out), payload)
    if args.json:
        print(__import__("json").dumps(payload, ensure_ascii=True))
    else:
        print(f"host_capability_contract status={payload['overall_status']} os={payload['body_map']['system']['os']} gpu={payload['body_map']['gpu_stack']['primary_gpu_stack']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
