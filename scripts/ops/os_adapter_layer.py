#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "os_adapter_layer_latest.json"
DEFAULT_HOST_PATH = PROJECT_ROOT / "governance" / "health" / "host_capability_contract_latest.json"


def _body(host: dict[str, Any]) -> dict[str, Any]:
    return host.get("body_map") if isinstance(host.get("body_map"), dict) else {}


def _system(host: dict[str, Any]) -> dict[str, Any]:
    body = _body(host)
    return body.get("system") if isinstance(body.get("system"), dict) else {}


def _cpu(host: dict[str, Any]) -> dict[str, Any]:
    body = _body(host)
    return body.get("cpu_topology") if isinstance(body.get("cpu_topology"), dict) else {}


def _gpu(host: dict[str, Any]) -> dict[str, Any]:
    body = _body(host)
    return body.get("gpu_stack") if isinstance(body.get("gpu_stack"), dict) else {}


def _launch(host: dict[str, Any]) -> dict[str, Any]:
    body = _body(host)
    return body.get("launch_system") if isinstance(body.get("launch_system"), dict) else {}


def _storage(host: dict[str, Any]) -> dict[str, Any]:
    body = _body(host)
    return body.get("storage_layout") if isinstance(body.get("storage_layout"), dict) else {}


def _adapter_id(os_name: str, gpu: dict[str, Any]) -> str:
    primary_gpu = str(gpu.get("primary_gpu_stack") or "none").lower()
    if os_name == "darwin":
        return "macos_apple_silicon_mlx_launchd" if primary_gpu in {"mlx", "metal"} else "macos_launchd"
    if os_name == "linux":
        return "linux_cuda_systemd" if primary_gpu == "cuda" else "linux_systemd_or_cron"
    if os_name == "wsl":
        return "wsl_portable"
    if os_name == "windows":
        return "windows_task_scheduler"
    return "portable_minimal"


def _process_priority_adapter(os_name: str, cpu: dict[str, Any]) -> dict[str, Any]:
    if os_name == "darwin":
        return {
            "supported": True,
            "adapter": "renice_taskpolicy_qos",
            "commands": ["renice", "taskpolicy"],
            "hard_affinity_supported": False,
            "policy": "performance_core_primary_with_qos_steering",
            "notes": ["macOS can bias work through QoS/nice, but it does not expose reliable hard P-core pinning per process."],
        }
    if os_name in {"linux", "wsl"}:
        return {
            "supported": True,
            "adapter": "renice_taskset_cgroups",
            "commands": ordered_unique(["renice", "taskset" if cpu.get("hard_affinity_supported") else "", "systemd-run"]),
            "hard_affinity_supported": bool(cpu.get("hard_affinity_supported", False)),
            "policy": "affinity_when_available_else_nice",
            "notes": [],
        }
    if os_name == "windows":
        return {
            "supported": True,
            "adapter": "powershell_priority_affinity",
            "commands": ["Start-Process", "SetPriorityClass", "ProcessorAffinity"],
            "hard_affinity_supported": True,
            "policy": "windows_priority_class_and_affinity",
            "notes": [],
        }
    return {"supported": False, "adapter": "none", "commands": [], "hard_affinity_supported": False, "policy": "observe_only", "notes": []}


def _service_adapter(os_name: str, launch: dict[str, Any]) -> dict[str, Any]:
    primary = str(launch.get("primary") or "unknown")
    if os_name == "darwin":
        commands = ["launchctl bootstrap", "launchctl kickstart", "launchctl bootout"]
    elif primary == "systemd":
        commands = ["systemctl --user enable", "systemctl --user start", "systemctl --user stop"]
    elif os_name in {"linux", "wsl"}:
        commands = ["crontab", "nohup"]
    elif os_name == "windows":
        commands = ["schtasks", "PowerShell ScheduledTasks"]
    else:
        commands = []
    return {"primary": primary, "commands": commands, "portable_service_manifest_required": primary != "launchd"}


def _probe_adapter(os_name: str, gpu: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_pressure": "vm_stat_and_memory_pressure" if os_name == "darwin" else "/proc/meminfo" if os_name in {"linux", "wsl"} else "platform_api",
        "thermal": "pmset_or_powermetrics" if os_name == "darwin" else "thermal_zone_or_nvidia_smi" if os_name in {"linux", "wsl"} else "platform_api",
        "gpu": "mlx_metal" if gpu.get("mlx_available") or gpu.get("metal_available") else "nvidia_smi" if gpu.get("cuda_available") else "rocm_smi" if gpu.get("rocm_available") else "none",
        "disk": "df_mount_diskutil" if os_name == "darwin" else "df_mount_lsblk" if os_name in {"linux", "wsl"} else "platform_api",
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, host: dict[str, Any] | None = None) -> dict[str, Any]:
    host_payload = host if host is not None else load_json(project_root / "governance" / "health" / "host_capability_contract_latest.json")
    system = _system(host_payload)
    cpu = _cpu(host_payload)
    gpu = _gpu(host_payload)
    launch = _launch(host_payload)
    storage = _storage(host_payload)
    os_name = str(system.get("os") or "unknown")
    adapter_id = _adapter_id(os_name, gpu)

    adapters = {
        "process_priority": _process_priority_adapter(os_name, cpu),
        "service_startup": _service_adapter(os_name, launch),
        "runtime_probes": _probe_adapter(os_name, gpu),
        "core_allocation": {
            "adapter": str(cpu.get("core_allocator") or "os_scheduler"),
            "topology": str(cpu.get("topology") or "unknown"),
            "primary_compute_lanes": int(cpu.get("recommended_primary_compute_lanes") or 1),
            "performance_core_count": int(cpu.get("performance_core_count") or 0),
            "efficiency_core_count": int(cpu.get("efficiency_core_count") or 0),
            "hard_affinity_supported": bool(cpu.get("hard_affinity_supported", False)),
        },
        "gpu_runtime": {
            "primary_stack": str(gpu.get("primary_gpu_stack") or "none"),
            "portable_backend_order": ordered_unique(
                [
                    "mlx" if gpu.get("mlx_available") else "",
                    "cuda" if gpu.get("cuda_available") else "",
                    "rocm" if gpu.get("rocm_available") else "",
                    "cpu",
                ]
            ),
        },
        "protected_storage": {
            "denylist": storage.get("protected_volumes", []),
            "rules": storage.get("denylist_rules", []),
            "enforced_policy": "denylist_paths_are_never_write_targets",
        },
    }
    capability_gaps = ordered_unique(
        [
            "host_capability_contract_missing" if not host_payload else "",
            "gpu_runtime_cpu_only" if adapters["gpu_runtime"]["primary_stack"] == "none" else "",
            "portable_service_manifest_needed" if adapters["service_startup"]["portable_service_manifest_required"] else "",
        ]
    )
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(host_payload),
        "overall_status": "ready" if host_payload and not capability_gaps else "advisory",
        "adapter_id": adapter_id,
        "os": os_name,
        "adapters": adapters,
        "capability_gaps": capability_gaps,
        "integration_contract": {
            "host_contract_required": True,
            "all_runtime_controls_should_read_adapter_before_os_specific_actions": True,
            "never_touch_protected_volumes": adapters["protected_storage"]["denylist"],
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the OS adapter layer that maps host capabilities to portable runtime actions.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"os_adapter_layer status={payload['overall_status']} adapter={payload['adapter_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
