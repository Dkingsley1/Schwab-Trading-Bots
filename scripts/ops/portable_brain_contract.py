#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.ml_backend_contract import detect_installed_backends, resolve_backend_contract
    from scripts.ops import apple_silicon_profile as apple_src
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, write_payload
else:
    from core.ml_backend_contract import detect_installed_backends, resolve_backend_contract
    from . import apple_silicon_profile as apple_src
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "portable_brain_contract_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.host_profile_override"
DEFAULT_PROOF_ROOT = PROJECT_ROOT / "governance" / "health"

GENERIC_PROFILE_PRESETS: dict[str, dict[str, str]] = {
    "portable_constrained": {
        "ASYNC_PIPELINE_WORKERS": "2",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "96",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "64",
        "RUNTIME_TRAIN_MAX_SAMPLES": "12000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "600",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "180",
    },
    "portable_balanced": {
        "ASYNC_PIPELINE_WORKERS": "4",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "160",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "96",
        "RUNTIME_TRAIN_MAX_SAMPLES": "22000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "300",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "150",
    },
    "portable_throughput": {
        "ASYNC_PIPELINE_WORKERS": "6",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "256",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "128",
        "RUNTIME_TRAIN_MAX_SAMPLES": "32000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "180",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _command_text(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def _linux_memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0
    try:
        for raw in meminfo.read_text(encoding="utf-8").splitlines():
            if not raw.startswith("MemTotal:"):
                continue
            kib = float(raw.split()[1])
            return round((kib * 1024.0) / (1024.0 ** 3), 2)
    except Exception:
        return 0.0
    return 0.0


def _windows_memory_gb() -> float:
    raw = _command_text(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
    values = [line.strip() for line in raw.splitlines() if line.strip() and "TotalPhysicalMemory" not in line]
    if not values:
        return 0.0
    try:
        return round(int(values[0]) / (1024.0 ** 3), 2)
    except Exception:
        return 0.0


def _memory_gb_for_system(system_name: str) -> float:
    if system_name == "Darwin":
        return _safe_float(apple_src._detect_hardware().get("memory_gb"), 0.0)
    if system_name == "Linux":
        return _linux_memory_gb()
    if system_name == "Windows":
        return _windows_memory_gb()
    return 0.0


def _chip_for_system(system_name: str) -> str:
    if system_name == "Darwin":
        hardware = apple_src._detect_hardware()
        return str(hardware.get("chip") or "").strip()
    if system_name == "Linux":
        model = ""
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                for raw in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if ":" not in raw:
                        continue
                    key, value = raw.split(":", 1)
                    if key.strip().lower() in {"model name", "hardware"}:
                        model = value.strip()
                        break
            except Exception:
                model = ""
        return model or platform.processor() or platform.machine()
    if system_name == "Windows":
        return platform.processor() or platform.machine()
    return platform.processor() or platform.machine()


def detect_host_hardware() -> dict[str, Any]:
    system_name = platform.system()
    machine = platform.machine()
    memory_gb = _memory_gb_for_system(system_name)
    chip = _chip_for_system(system_name)
    cpu_count = int(os.cpu_count() or 0)
    is_apple_silicon = system_name == "Darwin" and machine == "arm64"
    accelerator_hint = "cpu"
    if is_apple_silicon:
        accelerator_hint = "metal"
    elif str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip() not in {"", "-1", "none"}:
        accelerator_hint = "cuda"
    elif str(os.getenv("ROCR_VISIBLE_DEVICES", "")).strip() not in {"", "-1", "none"}:
        accelerator_hint = "rocm"
    recognized_host_and_chip = bool(system_name and machine and chip)
    return {
        "system": system_name,
        "release": platform.release(),
        "machine": machine,
        "processor": platform.processor(),
        "chip": chip,
        "memory_gb": memory_gb,
        "cpu_count": cpu_count,
        "is_apple_silicon": is_apple_silicon,
        "accelerator_hint": accelerator_hint,
        "recognized_host_and_chip": recognized_host_and_chip,
    }


def detect_host_profile(hardware: dict[str, Any]) -> str:
    if bool(hardware.get("is_apple_silicon")):
        return str(apple_src.detect_profile_tier(hardware))

    accelerator_hint = str(hardware.get("accelerator_hint") or "")
    memory_gb = _safe_float(hardware.get("memory_gb"), 0.0)
    cpu_count = int(hardware.get("cpu_count", 0) or 0)
    if accelerator_hint in {"cuda", "rocm"} or memory_gb >= 64.0 or cpu_count >= 16:
        return "portable_throughput"
    if memory_gb >= 24.0 or cpu_count >= 8:
        return "portable_balanced"
    return "portable_constrained"


def recommended_runtime_access_mode(hardware: dict[str, Any]) -> str:
    return "native" if bool(hardware.get("is_apple_silicon")) else "portable"


def recommended_ml_backend(hardware: dict[str, Any]) -> str:
    return "native_default" if bool(hardware.get("is_apple_silicon")) else "portable_auto"


def override_lines_for_host(profile: str, hardware: dict[str, Any]) -> list[str]:
    lines = [
        "# Auto-managed by scripts/ops/portable_brain_contract.py",
        f"BOT_HOST_PROFILE_SLUG={profile}",
        f"BOT_HOST_OS_FAMILY={str(hardware.get('system') or '').lower()}",
        f"BOT_HOST_ACCELERATOR_HINT={str(hardware.get('accelerator_hint') or '').lower()}",
        f"BOT_HOST_MEMORY_GB={_safe_float(hardware.get('memory_gb'), 0.0):.2f}",
        f"BOT_HOST_CPU_COUNT={int(hardware.get('cpu_count', 0) or 0)}",
        f"BOT_HOST_CHIP={str(hardware.get('chip') or '').replace(' ', '_')}",
    ]
    if not bool(hardware.get("is_apple_silicon")):
        for key, value in GENERIC_PROFILE_PRESETS.get(profile, {}).items():
            lines.append(f"{key}={value}")
    return lines


def _write_override(path: Path, profile: str, hardware: dict[str, Any]) -> bool:
    payload = "\n".join(override_lines_for_host(profile, hardware)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == payload:
        return False
    path.write_text(payload, encoding="utf-8")
    return True


def _deployment_matrix(
    *,
    native_contract: dict[str, Any],
    portable_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    portable_shadow = bool(portable_contract.get("shadow_replay_supported", False))
    portable_canary = bool(portable_contract.get("sidecar_canary_supported", False))
    return [
        {
            "platform": "macos_apple_silicon",
            "launch_supervisor": "launchd",
            "browser_launcher": "open",
            "recommended_access_mode": "native",
            "recommended_backend": str(native_contract.get("effective_backend") or ""),
            "live_trading_capable": bool(native_contract.get("live_trading_supported", False)),
            "shadow_replay_capable": bool(native_contract.get("shadow_replay_supported", False)),
        },
        {
            "platform": "linux_workstation",
            "launch_supervisor": "systemd",
            "browser_launcher": "xdg-open",
            "recommended_access_mode": "portable",
            "recommended_backend": str(portable_contract.get("effective_backend") or ""),
            "live_trading_capable": False,
            "shadow_replay_capable": portable_shadow,
            "sidecar_canary_capable": portable_canary,
        },
        {
            "platform": "windows_workstation",
            "launch_supervisor": "task_scheduler_or_service",
            "browser_launcher": "start",
            "recommended_access_mode": "portable",
            "recommended_backend": str(portable_contract.get("effective_backend") or ""),
            "live_trading_capable": False,
            "shadow_replay_capable": portable_shadow,
            "sidecar_canary_capable": portable_canary,
        },
    ]


def _cross_platform_proof_node(portable_contract: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    shadow_ready = bool(portable_contract.get("shadow_replay_supported", False))
    canary_ready = bool(portable_contract.get("sidecar_canary_supported", False))
    status = "ready" if shadow_ready else "planned"
    if str(hardware.get("system") or "") in {"Linux", "Windows"} and shadow_ready:
        status = "active_host_candidate"
    return {
        "status": status,
        "effective_backend": str(portable_contract.get("effective_backend") or ""),
        "shadow_replay_supported": shadow_ready,
        "sidecar_canary_supported": canary_ready,
        "host_candidate": str(hardware.get("system") or "").lower(),
        "recommended_next_step": (
            "run replay and parity checks on the non-Mac node before claiming live portability"
            if shadow_ready
            else "install an optional portable backend such as PyTorch or ONNX before running cross-platform proof"
        ),
    }


def build_payload(
    *,
    hardware: dict[str, Any],
    profile: str,
    override_path: Path,
    changed: bool,
    action: str,
) -> dict[str, Any]:
    installed = detect_installed_backends()
    runtime_mode = recommended_runtime_access_mode(hardware)
    ml_backend = recommended_ml_backend(hardware)
    native_contract = resolve_backend_contract("native_default", mode="native", installed=installed)
    portable_contract = resolve_backend_contract("portable_auto", mode="portable", installed=installed)
    env_overrides = (
        {}
        if bool(hardware.get("is_apple_silicon"))
        else GENERIC_PROFILE_PRESETS.get(profile, {})
    )
    cross_platform = _cross_platform_proof_node(portable_contract, hardware)
    parity_contract = {
        "nightly_proof_supported": bool(portable_contract.get("shadow_replay_supported", False)),
        "parity_focus": ("mlx_vs_portable_replay" if bool(hardware.get("is_apple_silicon")) else "portable_host_validation"),
        "backend_priority_by_os": {
            "darwin": ["mlx", "pytorch", "onnx"],
            "linux": ["onnx", "pytorch", "tensorflow"],
            "windows": ["onnx", "pytorch", "tensorflow"],
        },
        "recommended_reports": [
            "backend_parity_report",
            "shadow_replay_diff",
            "sidecar_canary_health",
        ],
    }
    portability_score = 45.0
    if bool(native_contract.get("live_trading_supported", False)):
        portability_score += 20.0
    if bool(portable_contract.get("shadow_replay_supported", False)):
        portability_score += 20.0
    if bool(portable_contract.get("sidecar_canary_supported", False)):
        portability_score += 10.0
    if bool(hardware.get("recognized_host_and_chip")):
        portability_score += 5.0
    portability_score = min(round(portability_score, 2), 100.0)

    overall_status = "ready"
    if runtime_mode == "portable" and not bool(portable_contract.get("shadow_replay_supported", False)):
        overall_status = "degraded"

    notes = [
        "the host contract now detects the operating system, chip family, memory envelope, and accelerator hint before choosing a runtime posture",
        "Apple Silicon stays MLX-native while Linux and Windows are steered toward the portable replay and sidecar contract",
        "the host-profile override file is safe to move between systems because it only carries local tuning hints and not broker secrets",
        "start_stack can auto-refresh this contract so transferred installs do not keep stale host assumptions",
    ]

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "recommended_runtime_mode": runtime_mode,
        "recommended_backend": ml_backend,
        "next_step": str(cross_platform.get("recommended_next_step") or ""),
        "action": action,
        "host_contract": {
            **hardware,
            "host_profile": profile,
            "recommended_runtime_access_mode": runtime_mode,
            "recommended_ml_backend": ml_backend,
        },
        "adaptation_contract": {
            "profile_source": (
                "apple_silicon_profile" if bool(hardware.get("is_apple_silicon")) else "portable_host_profile"
            ),
            "recommended_runtime_access_mode": runtime_mode,
            "recommended_ml_backend": ml_backend,
            "override_path": str(override_path),
            "override_exists": bool(override_path.exists()),
            "changed": bool(changed),
            "env_override_count": len(env_overrides),
            "env_overrides": env_overrides,
        },
        "native_contract": native_contract,
        "portable_contract": portable_contract,
        "cross_platform_proof_node": cross_platform,
        "parity_contract": parity_contract,
        "nightly_proof_contract": {
            "ready": bool(parity_contract.get("nightly_proof_supported", False)),
            "report_paths": {
                "backend_parity_report": str(DEFAULT_PROOF_ROOT / "backend_parity_report_latest.json"),
                "shadow_replay_diff": str(DEFAULT_PROOF_ROOT / "shadow_replay_diff_latest.json"),
                "sidecar_canary_health": str(DEFAULT_PROOF_ROOT / "sidecar_canary_health_latest.json"),
            },
            "recommended_backend_priority": parity_contract.get("backend_priority_by_os", {}).get(str(hardware.get("system") or "").lower(), []),
            "recommended_next_step": str(cross_platform.get("recommended_next_step") or ""),
        },
        "deployment_matrix": _deployment_matrix(native_contract=native_contract, portable_contract=portable_contract),
        "transfer_contract": {
            "project_local_storage": True,
            "sqlite_preferred": True,
            "artifact_store_mode": "content_addressed_json_and_sqlite",
            "path_semantics": {
                "macos": "posix",
                "linux": "posix",
                "windows": "nt",
            },
            "supervisor_mapping": {
                "macos": "launchd",
                "linux": "systemd",
                "windows": "task_scheduler_or_service",
            },
            "browser_launchers": {
                "macos": "open",
                "linux": "xdg-open",
                "windows": "start",
            },
            "scheduler_hints": {
                "macos": "launchd keepalive",
                "linux": "systemd timer plus service",
                "windows": "scheduled task plus service wrapper",
            },
        },
        "portability_score": portability_score,
        "recommended_actions": [
            "keep Apple Silicon on the native MLX lane when you want the full live-trading brain",
            "use portable mode plus the proof-node contract for Linux and Windows transfers before attempting any promotion claims",
            "refresh the host profile whenever the project moves to a different machine class so stale cache and worker ceilings do not follow it",
        ],
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the portable brain contract and host-aware tuning profile.")
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    override_path = Path(args.override_file).expanduser()
    hardware = detect_host_hardware()
    profile = detect_host_profile(hardware)
    changed = False
    if args.action == "apply":
        changed = _write_override(override_path, profile, hardware)
    payload = build_payload(
        hardware=hardware,
        profile=profile,
        override_path=override_path,
        changed=changed,
        action=args.action,
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "portable_brain_contract "
            f"overall_status={payload.get('overall_status', '')} "
            f"host_profile={payload.get('host_contract', {}).get('host_profile', '')} "
            f"portability_score={float(payload.get('portability_score', 0.0) or 0.0):.2f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
