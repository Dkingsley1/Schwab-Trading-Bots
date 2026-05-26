#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERRIDE = PROJECT_ROOT / "config" / ".env.apple_silicon_override"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "apple_silicon_profile_latest.json"

PROFILE_PRESETS: Dict[str, Dict[str, str]] = {
    "air_safe": {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "90",
        "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "0.75",
        "SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB": "6",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "1800000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "80000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "50000",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "300",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1200",
        "BOT_LOGS_LOW_SPACE_AUTOPRUNE_MIN_FREE_GB": "40",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "8",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "88",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "8",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "88",
        "RESOURCE_GUARD_MEMORY_YELLOW_SWAP_GB": "8",
        "RESOURCE_GUARD_MEMORY_RED_SWAP_GB": "12",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
        "COINBASE_CACHE_MAX_ENTRIES": "128",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "6",
        "TRADE_BEHAVIOR_BATCH_SIZE": "768",
        "ASYNC_PIPELINE_WORKERS": "3",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "96",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "24",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "32",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "32",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "2",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "64",
        "RUNTIME_TRAIN_MAX_SAMPLES": "12000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "600",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "240",
        "RESOURCE_GUARD_CREATIVE_APP_NAMES": "Final Cut Pro,Logic Pro,Music,iTunes",
        "RESOURCE_GUARD_MUSIC_HOT_CPU_THRESHOLD": "45",
        "RESOURCE_GUARD_CREATIVE_HOT_CPU_THRESHOLD": "90",
        "RESOURCE_GUARD_BLOCK_ON_CREATIVE_SESSION_LEVELS": "active,dual_pro,hot",
        "RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS": "active,dual_pro,hot",
        "RESOURCE_GUARD_REFRESH_BLOCK_ON_CREATIVE_SESSION_LEVELS": "active,dual_pro,hot",
        "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE": "constrained",
        "MEMORY_EFFICIENCY_CREATIVE_DUAL_PROFILE": "constrained",
        "CREATIVE_AUDIO_SAMPLE_RATE_HZ": "96000",
        "LOGIC_PRO_AUDIO_SAMPLE_RATE_HZ": "96000",
        "CREATIVE_AUDIO_REQUIRE_MATCHED_IO_SAMPLE_RATE": "1",
    },
    "pro_balanced": {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "60",
        "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "720",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "1.25",
        "SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB": "10",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "2500000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "100000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "70000",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "240",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "900",
        "BOT_LOGS_LOW_SPACE_AUTOPRUNE_MIN_FREE_GB": "70",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "16",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "82",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "16",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "82",
        "RESOURCE_GUARD_MEMORY_YELLOW_SWAP_GB": "12",
        "RESOURCE_GUARD_MEMORY_RED_SWAP_GB": "18",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
        "COINBASE_CACHE_MAX_ENTRIES": "256",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "8",
        "TRADE_BEHAVIOR_BATCH_SIZE": "1024",
        "ASYNC_PIPELINE_WORKERS": "4",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "160",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "40",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "48",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "48",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "1",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "96",
        "RUNTIME_TRAIN_MAX_SAMPLES": "20000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "300",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "180",
        "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "180",
        "RESOURCE_GUARD_CREATIVE_APP_NAMES": "Final Cut Pro,Logic Pro,Music,iTunes",
        "RESOURCE_GUARD_MUSIC_HOT_CPU_THRESHOLD": "45",
        "RESOURCE_GUARD_CREATIVE_HOT_CPU_THRESHOLD": "140",
        "RESOURCE_GUARD_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot",
        "RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS": "active,dual_pro,hot",
        "RESOURCE_GUARD_REFRESH_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot",
        "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE": "constrained",
        "MEMORY_EFFICIENCY_CREATIVE_DUAL_PROFILE": "constrained",
        "CREATIVE_AUDIO_SAMPLE_RATE_HZ": "96000",
        "LOGIC_PRO_AUDIO_SAMPLE_RATE_HZ": "96000",
        "CREATIVE_AUDIO_REQUIRE_MATCHED_IO_SAMPLE_RATE": "1",
    },
    "max_throughput": {
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "45",
        "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "600",
        "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": "1.75",
        "SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB": "16",
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": "4000000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "140000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "90000",
        "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "180",
        "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "900",
        "BOT_LOGS_LOW_SPACE_AUTOPRUNE_MIN_FREE_GB": "100",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "24",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "75",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "24",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "75",
        "RESOURCE_GUARD_MEMORY_YELLOW_SWAP_GB": "16",
        "RESOURCE_GUARD_MEMORY_RED_SWAP_GB": "24",
        "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
        "COINBASE_CACHE_MAX_ENTRIES": "512",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "10",
        "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
        "ASYNC_PIPELINE_WORKERS": "6",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "256",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "64",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "72",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "72",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "1",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "96",
        "RUNTIME_TRAIN_MAX_SAMPLES": "32000",
        "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "180",
        "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "120",
        "OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS": "120",
        "RESOURCE_GUARD_CREATIVE_APP_NAMES": "Final Cut Pro,Logic Pro,Music,iTunes",
        "RESOURCE_GUARD_MUSIC_HOT_CPU_THRESHOLD": "45",
        "RESOURCE_GUARD_CREATIVE_HOT_CPU_THRESHOLD": "135",
        "RESOURCE_GUARD_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot",
        "RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot",
        "RESOURCE_GUARD_REFRESH_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot",
        "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE": "pro_balanced",
        "MEMORY_EFFICIENCY_CREATIVE_MUSIC_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE": "air_safe",
        "MEMORY_EFFICIENCY_CREATIVE_DUAL_PROFILE": "constrained",
        "CREATIVE_AUDIO_SAMPLE_RATE_HZ": "96000",
        "LOGIC_PRO_AUDIO_SAMPLE_RATE_HZ": "96000",
        "CREATIVE_AUDIO_REQUIRE_MATCHED_IO_SAMPLE_RATE": "1",
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sysctl_text(name: str) -> str:
    try:
        proc = subprocess.run(
            ["/usr/sbin/sysctl", "-n", str(name)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def _sysctl_int(name: str, default: int = 0) -> int:
    text = _sysctl_text(name)
    try:
        return int(str(text).strip())
    except Exception:
        return int(default)


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _detect_hardware() -> Dict[str, Any]:
    system_name = platform.system()
    machine = platform.machine()
    chip = _sysctl_text("machdep.cpu.brand_string") or _sysctl_text("hw.model")
    mem_text = _sysctl_text("hw.memsize")
    try:
        memory_gb = round(int(mem_text) / (1024.0 ** 3), 2) if mem_text else 0.0
    except Exception:
        memory_gb = 0.0
    is_apple_silicon = system_name == "Darwin" and machine == "arm64"
    physical_core_count = _sysctl_int("hw.physicalcpu", 0)
    logical_core_count = _sysctl_int("hw.logicalcpu", 0) or int(os.cpu_count() or 0)
    performance_core_count = _sysctl_int("hw.perflevel0.physicalcpu", 0)
    efficiency_core_count = _sysctl_int("hw.perflevel1.physicalcpu", 0)
    total_core_hint = physical_core_count or logical_core_count
    if is_apple_silicon and performance_core_count <= 0 and total_core_hint > 0:
        chip_lower = str(chip or "").lower()
        if "max" in chip_lower or "ultra" in chip_lower or memory_gb >= 64.0:
            performance_core_count = min(total_core_hint, 8)
        elif "pro" in chip_lower or memory_gb >= 24.0:
            performance_core_count = min(total_core_hint, 6)
        else:
            performance_core_count = min(total_core_hint, 4)
        efficiency_core_count = max(total_core_hint - performance_core_count, 0)
    elif performance_core_count > 0 and efficiency_core_count <= 0 and total_core_hint > 0:
        efficiency_core_count = max(total_core_hint - performance_core_count, 0)
    return {
        "system": system_name,
        "machine": machine,
        "chip": chip,
        "memory_gb": memory_gb,
        "is_apple_silicon": is_apple_silicon,
        "physical_core_count": physical_core_count,
        "logical_core_count": logical_core_count,
        "performance_core_count": performance_core_count,
        "efficiency_core_count": efficiency_core_count,
    }


def detect_profile_tier(hardware: Dict[str, Any]) -> str:
    if not bool(hardware.get("is_apple_silicon")):
        return "generic"
    chip = str(hardware.get("chip") or "").strip().lower()
    memory_gb = float(hardware.get("memory_gb", 0.0) or 0.0)
    if "ultra" in chip or "max" in chip or memory_gb >= 64.0:
        return "max_throughput"
    if "pro" in chip or memory_gb >= 24.0:
        return "pro_balanced"
    return "air_safe"


def _unified_memory_telemetry(hardware: Dict[str, Any], tier: str) -> Dict[str, Any]:
    memory_gb = float(hardware.get("memory_gb", 0.0) or 0.0)
    is_unified = bool(hardware.get("is_apple_silicon", False))
    feature_budget_ratio = 0.18 if tier == "max_throughput" else 0.12 if tier == "pro_balanced" else 0.08
    inference_budget_ratio = 0.12 if tier == "max_throughput" else 0.08 if tier == "pro_balanced" else 0.05
    competitive_advantage = "high" if is_unified and memory_gb >= 32.0 else "moderate" if is_unified else "portable_only"
    return {
        "memory_architecture": ("unified" if is_unified else "system_memory"),
        "shared_cpu_gpu_memory_pool": is_unified,
        "estimated_feature_cache_budget_gb": round(memory_gb * feature_budget_ratio, 3),
        "estimated_live_inference_budget_gb": round(memory_gb * inference_budget_ratio, 3),
        "broker_context_window_multiplier": (2.0 if is_unified and tier == "max_throughput" else 1.5 if is_unified else 1.0),
        "competitive_advantage": competitive_advantage,
        "copy_avoidance_summary": (
            "Shared CPU, GPU, and MLX memory keeps broker context, feature windows, and inference tensors on one low-copy path."
            if is_unified
            else "Portable hosts can run the stack, but they usually incur extra copy overhead between CPU and accelerator memory domains."
        ),
    }


def _creative_audio_contract(tier: str) -> Dict[str, Any]:
    profile = PROFILE_PRESETS.get(str(tier), {})
    sample_rate = int(profile.get("CREATIVE_AUDIO_SAMPLE_RATE_HZ", "96000") or 96000)
    return {
        "target_sample_rate_hz": sample_rate,
        "target_sample_rate_khz": round(sample_rate / 1000.0, 1),
        "require_matched_input_output": profile.get("CREATIVE_AUDIO_REQUIRE_MATCHED_IO_SAMPLE_RATE", "1") == "1",
        "logic_pro_sample_rate_hz": int(profile.get("LOGIC_PRO_AUDIO_SAMPLE_RATE_HZ", str(sample_rate)) or sample_rate),
        "reason": "Logic Pro and standalone audio apps should see the same 96 kHz input/output contract before heavy bot work starts.",
    }


def _performance_core_contract(hardware: Dict[str, Any], tier: str) -> Dict[str, Any]:
    profile = PROFILE_PRESETS.get(str(tier), {})
    is_apple_silicon = bool(hardware.get("is_apple_silicon", False))
    physical_core_count = _positive_int(hardware.get("physical_core_count"), 0)
    logical_core_count = _positive_int(hardware.get("logical_core_count"), 0)
    performance_core_count = _positive_int(hardware.get("performance_core_count"), 0)
    efficiency_core_count = _positive_int(hardware.get("efficiency_core_count"), 0)
    total_core_hint = physical_core_count or logical_core_count or int(os.cpu_count() or 1)
    if is_apple_silicon and performance_core_count <= 0:
        chip_lower = str(hardware.get("chip") or "").lower()
        if "max" in chip_lower or "ultra" in chip_lower or float(hardware.get("memory_gb") or 0.0) >= 64.0:
            performance_core_count = min(total_core_hint, 8)
        elif "pro" in chip_lower or float(hardware.get("memory_gb") or 0.0) >= 24.0:
            performance_core_count = min(total_core_hint, 6)
        else:
            performance_core_count = min(total_core_hint, 4)
        efficiency_core_count = max(total_core_hint - performance_core_count, 0)
    primary_budget = max(performance_core_count or min(total_core_hint, 4), 1)
    async_workers = _positive_int(profile.get("ASYNC_PIPELINE_WORKERS"), min(primary_budget, 4))
    support_spillover_workers = min(max(efficiency_core_count, 0), 2)
    return {
        "policy": "performance_core_primary" if is_apple_silicon else "portable_scheduler_default",
        "hard_affinity_supported": False,
        "macos_hard_affinity_note": "macOS does not expose portable hard P-core pinning for these shell/Python workers; this policy is expressed through env budgets, worker caps, nice/QoS, and foreground governors.",
        "scheduler_intent": "keep the SQL writer and hot backlog drain off macOS background taskpolicy unless foreground creative work is active",
        "physical_core_count": physical_core_count,
        "logical_core_count": logical_core_count,
        "primary_performance_core_count": performance_core_count,
        "primary_performance_core_budget": primary_budget,
        "efficiency_spillover_core_budget": efficiency_core_count,
        "support_spillover_workers": support_spillover_workers,
        "foreground_app_reserve": 1,
        "worker_budget_contract": {
            "system_primary_workers": primary_budget,
            "default_async_pipeline_workers": async_workers,
            "baseline_sleeve_workers": primary_budget,
            "support_spillover_workers": support_spillover_workers,
            "foreground_governors_may_shrink": True,
        },
    }


def _core_allocation_env(hardware: Dict[str, Any], tier: str) -> Dict[str, str]:
    if str(tier) not in PROFILE_PRESETS:
        return {}
    contract = _performance_core_contract(hardware, tier)
    primary_budget = _positive_int(contract.get("primary_performance_core_budget"), 1)
    efficiency_budget = _positive_int(contract.get("efficiency_spillover_core_budget"), 0)
    support_workers = _positive_int(contract.get("support_spillover_workers"), 0)
    return {
        "BOT_CPU_ALLOCATION_POLICY": str(contract.get("policy") or "performance_core_primary"),
        "BOT_CPU_HARD_AFFINITY_SUPPORTED": "0",
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
        "BOT_CPU_EFFICIENCY_SATURATION_GUARD": "1",
        "BOT_CPU_SPILLOVER_POLICY": "efficiency_core_low_priority_spillover",
        "BOT_PERFORMANCE_CORE_PRIMARY_COUNT": str(_positive_int(contract.get("primary_performance_core_count"), primary_budget)),
        "BOT_PERFORMANCE_CORE_TARGET": str(primary_budget),
        "BOT_EFFICIENCY_CORE_SPILLOVER_COUNT": str(efficiency_budget),
        "BOT_CPU_PRIMARY_WORKER_BUDGET": str(primary_budget),
        "BOT_CPU_SUPPORT_SPILLOVER_WORKERS": str(support_workers),
        "BOT_CPU_FOREGROUND_APP_RESERVE": str(_positive_int(contract.get("foreground_app_reserve"), 1)),
        "SQL_LINK_WRITER_NICE": "0",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        "OPS_SQL_WRITER_NICE": "0",
        "OPS_SQL_WRITER_BACKGROUND_POLICY": "0",
        "BACKPRESSURE_DRAINER_NICE": "0",
        "SLEEVE_WORKERS_BASELINE": str(primary_budget),
        "SLEEVE_WORKERS_DIVIDEND": str(max(min(support_workers, 2), 1)),
        "SLEEVE_WORKERS_BOND": str(max(min(support_workers, 2), 1)),
        "SLEEVE_WORKERS_FX": str(max(min(support_workers, 2), 1)),
        "SLEEVE_NICE_BASELINE": "0",
        "SLEEVE_NICE_AGGRESSIVE": "0",
        "SLEEVE_NICE_SPECIALIZED": "6",
        "SLEEVE_NICE_DIVIDEND": "8",
        "SLEEVE_NICE_DIVIDEND_CAPTURE": "8",
        "SLEEVE_NICE_BOND": "8",
        "SLEEVE_NICE_FX": "8",
    }


def _env_overrides_for_tier(tier: str, hardware: Dict[str, Any]) -> Dict[str, str]:
    profile = PROFILE_PRESETS.get(str(tier), {})
    env = dict(_core_allocation_env(hardware, tier))
    env.update(profile)
    return env


def override_lines_for_tier(tier: str, hardware: Dict[str, Any]) -> list[str]:
    env_overrides = _env_overrides_for_tier(tier, hardware)
    lines = [
        "# Auto-managed by scripts/ops/apple_silicon_profile.py",
        f"BOT_APPLE_SILICON_TIER={shlex.quote(str(tier))}",
        f"BOT_APPLE_SILICON_MEMORY_GB={shlex.quote(str(hardware.get('memory_gb', 0.0)))}",
        f"BOT_APPLE_SILICON_CHIP={shlex.quote(str(hardware.get('chip') or '').replace(' ', '_'))}",
    ]
    for key, value in env_overrides.items():
        lines.append(f"{key}={shlex.quote(str(value))}")
    return lines


def _write_override(path: Path, tier: str, hardware: Dict[str, Any]) -> bool:
    if tier == "generic":
        if path.exists():
            path.unlink()
            return True
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(override_lines_for_tier(tier, hardware)) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == payload:
        return False
    path.write_text(payload, encoding="utf-8")
    return True


def build_payload(*, action: str, tier: str, hardware: Dict[str, Any], override_path: Path, changed: bool) -> Dict[str, Any]:
    unified_memory = _unified_memory_telemetry(hardware, tier)
    performance_core_contract = _performance_core_contract(hardware, tier)
    env_overrides = _env_overrides_for_tier(tier, hardware)
    notes = []
    if tier == "air_safe":
        notes.append("favor slower refresh cadence and tighter swap ceilings for MacBook Air and low-memory Apple Silicon")
    elif tier == "pro_balanced":
        notes.append("use balanced refresh cadence and moderate retention thresholds for Apple Silicon Pro-class machines")
    elif tier == "max_throughput":
        notes.append("allow denser ingestion and retention budgets for Max or Ultra class Apple Silicon machines")
    else:
        notes.append("no Apple Silicon-specific override was applied on this hardware")
    notes.append("MLX remains the preferred live backend; profile overrides focus on storage, ingestion, and memory behavior")
    notes.append("Creative audio sessions pin the intended Logic Pro input/output contract to 96 kHz so runtime guards do not treat 48 kHz as the default.")
    notes.append("Apple Silicon hosts use a performance-core-primary contract: detected P cores are the main worker budget, while efficiency cores are reserved for low-priority spillover and support work.")
    return {
        "timestamp_utc": _now_utc(),
        "ok": True,
        "action": action,
        "hardware": hardware,
        "detected_tier": detect_profile_tier(hardware),
        "applied_tier": tier,
        "changed": bool(changed),
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "env_overrides": env_overrides,
        "unified_memory_telemetry": unified_memory,
        "creative_audio_contract": _creative_audio_contract(tier),
        "performance_core_contract": performance_core_contract,
        "notes": notes,
    }


def _write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and optionally apply an Apple Silicon runtime/storage profile.")
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--tier", default="", help="Optionally pin air_safe|pro_balanced|max_throughput")
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    hardware = _detect_hardware()
    detected_tier = detect_profile_tier(hardware)
    requested_tier = str(args.tier or "").strip().lower()
    tier = requested_tier if requested_tier in PROFILE_PRESETS else detected_tier
    override_path = Path(args.override_file).expanduser()

    changed = False
    if args.action == "apply":
        changed = _write_override(override_path, tier, hardware)

    payload = build_payload(
        action=args.action,
        tier=tier,
        hardware=hardware,
        override_path=override_path,
        changed=changed,
    )
    _write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "apple_silicon_profile "
            f"tier={payload['applied_tier']} "
            f"memory_gb={hardware.get('memory_gb', 0.0)} "
            f"changed={int(bool(payload['changed']))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
