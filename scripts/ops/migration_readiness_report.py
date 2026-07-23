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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "migration_readiness_report_latest.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _target_stack(target_os: str, current_gpu: str) -> dict[str, Any]:
    normalized = target_os.strip().lower() or "auto"
    if normalized in {"auto", "current"}:
        return {"target_os": "current", "gpu_backend": current_gpu, "service_manager": "current_adapter"}
    if normalized in {"macos", "darwin", "mac"}:
        return {"target_os": "darwin", "gpu_backend": "mlx_metal", "service_manager": "launchd"}
    if normalized in {"linux", "ubuntu", "debian"}:
        return {"target_os": "linux", "gpu_backend": "cuda_or_rocm_or_cpu", "service_manager": "systemd"}
    if normalized in {"windows", "win"}:
        return {"target_os": "windows", "gpu_backend": "cuda_or_cpu", "service_manager": "task_scheduler"}
    if normalized == "wsl":
        return {"target_os": "wsl", "gpu_backend": "cuda_or_cpu", "service_manager": "windows_scheduler_plus_wsl_bootstrap"}
    return {"target_os": normalized, "gpu_backend": "unknown", "service_manager": "unknown"}


def _migration_items(host: dict[str, Any], adapter: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    body = _as_dict(host.get("body_map"))
    system = _as_dict(body.get("system"))
    gpu = _as_dict(body.get("gpu_stack"))
    storage = _as_dict(body.get("storage_layout"))
    launch = _as_dict(body.get("launch_system"))
    current_os = str(system.get("os") or "unknown")
    target_os = str(target.get("target_os") or "current")
    items: list[dict[str, Any]] = []
    if target_os not in {"current", current_os}:
        items.append(
            {
                "area": "launch_services",
                "status": "needs_rebind",
                "current": launch.get("primary", "unknown"),
                "target": target.get("service_manager"),
                "action": "Regenerate runtime services through the OS adapter instead of copying launchd plists blindly.",
            }
        )
    if str(gpu.get("primary_gpu_stack") or "none").lower() in {"mlx", "metal"} and target_os in {"linux", "windows", "wsl"}:
        items.append(
            {
                "area": "gpu_backend",
                "status": "needs_backend_switch",
                "current": "mlx_metal",
                "target": target.get("gpu_backend"),
                "action": "Route MLX-specific workloads through portable backend selection before expecting CUDA/NVIDIA acceleration.",
            }
        )
    protected = _as_list(storage.get("protected_volumes"))
    items.append(
        {
            "area": "protected_volumes",
            "status": "ready" if "/Volumes/VIDEO" in protected else "needs_rule",
            "current": protected,
            "target": "preserve_denylist",
            "action": "Keep /Volumes/VIDEO denylisted on every host; never use it for cleanup, pruning, or benchmark writes.",
        }
    )
    items.extend(
        [
            {
                "area": "secrets_tokens",
                "status": "needs_rebind",
                "current": "local_keychain_or_env",
                "target": "new_host_secret_store",
                "action": "Rebind Schwab/Coinbase/provider credentials through the target OS secret store or .env.secrets.local.",
            },
            {
                "area": "storage_mounts",
                "status": "needs_verify",
                "current": storage.get("bot_logs_external_mount", "/Volumes/BOT_LOGS"),
                "target": "fast_local_or_external_logs_volume",
                "action": "Verify BOT_LOGS mount path, filesystem, symlinks, and SQLite write latency before starting full loops.",
            },
            {
                "area": "python_runtime",
                "status": "needs_verify",
                "current": "local .venv314",
                "target": "host-native Python 3.14.5 venv with runtime dependency profiles",
                "action": "Verify the Python 3.14.5 virtualenv and run runtime audits before launch.",
            },
        ]
    )
    if "portable_service_manifest_needed" in _as_list(adapter.get("capability_gaps")):
        items.append(
            {
                "area": "portable_service_manifest",
                "status": "recommended",
                "current": adapter.get("adapter_id", "unknown"),
                "target": target.get("service_manager"),
                "action": "Generate a neutral service manifest from workload classes so launchd/systemd/Task Scheduler can be rebuilt.",
            }
        )
    return items


def _binder_checklist(items: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    required = [
        {
            "step": "publish_current_host_contract",
            "status": "ready",
            "command": ["./scripts/ops/opsctl.sh", "host-capability", "--json"],
            "purpose": "Capture CPU, memory, GPU, launch system, storage, and protected-volume body map.",
        },
        {
            "step": "publish_target_os_adapter",
            "status": "ready",
            "command": ["./scripts/ops/opsctl.sh", "os-adapter", "--json"],
            "purpose": "Translate runtime controls to launchd/systemd/Task Scheduler and OS-specific priority APIs.",
        },
        {
            "step": "benchmark_new_host_before_full_loops",
            "status": "ready",
            "command": ["./scripts/ops/opsctl.sh", "host-self-benchmark", "--json"],
            "purpose": "Set writer, JSONL parse, SQLite, storage, collector, and MLX limits from the actual machine.",
        },
        {
            "step": "apply_autonomic_budget",
            "status": "ready",
            "command": ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"],
            "purpose": "Apply safe P-core, collector, training, report, and MLX budgets before loops start.",
        },
    ]
    rebinding = [
        {
            "step": "rebind_secrets_tokens",
            "status": "operator_required",
            "command": [],
            "purpose": "Rebind Schwab, Coinbase, provider, and keychain/env secrets on the new host.",
        },
        {
            "step": "rebind_bot_logs_storage",
            "status": "operator_required",
            "command": ["./scripts/ops/opsctl.sh", "storage-reconnect-regression-guard", "--json"],
            "purpose": "Verify BOT_LOGS mount, filesystem, symlinks, and safe fallback behavior.",
        },
        {
            "step": "rebuild_python_runtime",
            "status": "operator_required",
            "command": [],
            "purpose": "Create a host-native virtualenv and run dependency/runtime audits before services launch.",
        },
    ]
    target_os = str(target.get("target_os") or "current")
    if target_os in {"linux", "windows", "wsl"}:
        rebinding.append(
            {
                "step": "switch_gpu_backend_contract",
                "status": "operator_required",
                "command": ["./scripts/ops/opsctl.sh", "runtime-backend-switch", "portable_auto", "--json"],
                "purpose": "Move MLX-only lanes behind portable backend routing before expecting CUDA/CPU parity.",
            }
        )
    return {
        "enabled": True,
        "target_os": target_os,
        "required_preflight": required,
        "operator_rebind_steps": rebinding,
        "protected_volume_rule": {
            "path": "/Volumes/VIDEO",
            "policy": "never_write_prune_or_benchmark_even_if_mounted",
        },
        "blocking_area_count": len([item for item in items if str(item.get("status")) in {"needs_rebind", "needs_backend_switch", "needs_rule", "needs_rebuild"}]),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, target_os: str = "current") -> dict[str, Any]:
    health = project_root / "governance" / "health"
    host = load_json(health / "host_capability_contract_latest.json")
    adapter = load_json(health / "os_adapter_layer_latest.json")
    benchmark = load_json(health / "host_self_benchmark_latest.json")
    body = _as_dict(host.get("body_map"))
    gpu = _as_dict(body.get("gpu_stack"))
    current_gpu = str(gpu.get("primary_gpu_stack") or "none")
    target = _target_stack(target_os, current_gpu)
    items = _migration_items(host, adapter, target)
    binder = _binder_checklist(items, target)
    blockers = [item for item in items if str(item.get("status")) in {"needs_rebind", "needs_backend_switch", "needs_rule", "needs_rebuild"}]
    score = max(100 - len(blockers) * 12 - (0 if benchmark else 8), 0)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(host and adapter),
        "overall_status": "ready" if score >= 80 else "advisory" if score >= 55 else "needs_work",
        "readiness_score": score,
        "current_host": {
            "os": _as_dict(body.get("system")).get("os", "unknown"),
            "cpu_topology": _as_dict(body.get("cpu_topology")).get("topology", "unknown"),
            "gpu_stack": current_gpu,
            "primary_compute_lanes": _safe_int(_as_dict(body.get("cpu_topology")).get("recommended_primary_compute_lanes"), 0),
        },
        "target_stack": target,
        "migration_items": items,
        "blocking_items": blockers,
        "migration_binder": binder,
        "can_this_machine_handle_current_system": {
            "answer": "yes_with_governor" if host and adapter else "unknown_until_contracts_refresh",
            "basis": "Host contract, OS adapter, and governor budgets are available." if host and adapter else "Run host-capability and os-adapter first.",
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "host-capability", "--json"],
            ["./scripts/ops/opsctl.sh", "os-adapter", "--json"],
            ["./scripts/ops/opsctl.sh", "host-self-benchmark", "--json"],
            ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report migration readiness across macOS, Linux, Windows, WSL, GPU stacks, services, secrets, and mounts.")
    parser.add_argument("--target-os", default="current")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT, target_os=args.target_os)
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"migration_readiness status={payload['overall_status']} score={payload['readiness_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
