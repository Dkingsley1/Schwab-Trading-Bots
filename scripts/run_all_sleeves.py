import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

VENV_PY = resolve_runtime_python(PROJECT_ROOT)
PARALLEL_SHADOWS = PROJECT_ROOT / "scripts" / "run_parallel_shadows.py"
DIVIDEND_SHADOW = PROJECT_ROOT / "scripts" / "run_dividend_shadow.py"
DIVIDEND_CAPTURE_SHADOW = PROJECT_ROOT / "scripts" / "run_dividend_capture_shadow.py"
BOND_SHADOW = PROJECT_ROOT / "scripts" / "run_bond_shadow.py"
FX_SHADOW = PROJECT_ROOT / "scripts" / "run_fx_shadow.py"
SPECIALIZED_SLEEVE_SHADOW = PROJECT_ROOT / "scripts" / "run_specialized_sleeve_shadow.py"
AGGRESSIVE_MODES = PROJECT_ROOT / "scripts" / "run_parallel_aggressive_modes.py"
EXECUTION_LANE = PROJECT_ROOT / "scripts" / "run_execution_lane.py"
HALT_FLAG_PATH = PROJECT_ROOT / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
PREFLIGHT_SCRIPT = PROJECT_ROOT / "scripts" / "shadow_preflight.py"
DEBUG_SNAPSHOT_SCRIPT = PROJECT_ROOT / "scripts" / "collect_debug_snapshot.sh"
CAPTURE_CONFIG_SCRIPT = PROJECT_ROOT / "scripts" / "capture_run_config.py"
PAPER_TRADE_LOCK_PATH = PROJECT_ROOT / "governance" / "health" / "PAPER_TRADE_LOCK.flag"
LAUNCHER_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "all_sleeves_launcher_latest.json"
PROCESS_FANOUT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.process_fanout_guard_override"
SLEEVE_LAUNCHER_REPAIR_INFRABOTS = (
    {
        "name": "sleeve_launcher_parent_watchdog",
        "responsibility": "restart the all-sleeves parent when it is missing and startup gates allow a read-only sleeve restart",
        "command": [str(VENV_PY), "scripts/ops/process_watchdog.py", "--json"],
    },
    {
        "name": "sleeve_child_recycler",
        "responsibility": "clean orphaned sleeve children and rebuild the child fanout when the parent is hollow or a child exits",
        "command": [str(VENV_PY), "scripts/ops/process_watchdog.py", "--json"],
    },
    {
        "name": "sleeve_preflight_repair_guard",
        "responsibility": "refresh storage, halt, auth, and runtime preflight evidence before restarting the launcher",
        "command": ["./scripts/ops/opsctl.sh", "post-restart-settle", "--apply", "--json"],
    },
    {
        "name": "sleeve_backpressure_guard",
        "responsibility": "keep backlog drain and storage governor work ahead of sleeve widening after a launcher repair",
        "command": ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
    },
    {
        "name": "sleeve_fanout_integrity_infrabot",
        "responsibility": "grade the parent-child fanout and separate true outages from intentionally parked sleeves",
        "command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"],
    },
    {
        "name": "sleeve_restart_storm_circuit_infrabot",
        "responsibility": "hold noisy children in quarantine instead of letting restart storms consume the computer",
        "command": [str(VENV_PY), "scripts/ops/process_watchdog.py", "--json"],
    },
    {
        "name": "sleeve_expansion_admission_infrabot",
        "responsibility": "allow new collection-only sleeves only when core, storage, and paper executor health are stable",
        "command": ["./scripts/ops/opsctl.sh", "sleeve-mechanics", "--json"],
    },
    {
        "name": "sleeve_computer_coexistence_infrabot",
        "responsibility": "keep sleeve launch pressure compatible with foreground Mac work and creative cotenant guards",
        "command": ["./scripts/ops/opsctl.sh", "runtime-throttle-control", "--json"],
    },
)

DEFAULT_SYMBOLS_CORE = (
    "SPY,QQQ,DIA,IWM,MDY,VOO,VTI,RSP,"
    "AAPL,MSFT,NVDA,AMD,AVGO,TSM,ASML,MU,ARM,SMH,SOXX,"
    "AMZN,GOOG,GOOGL,META,NFLX,DIS,WBD,ORCL,CRM,ADBE,NOW,PLTR,"
    "JPM,BAC,GS,MS,BLK,SCHW,AXP,V,MA,"
    "LLY,UNH,JNJ,ABBV,MRK,ABT,PFE,"
    "COST,WMT,HD,MCD,NKE,SBUX,"
    "CAT,DE,GE,BA,RTX,LMT,NOC,"
    "XOM,CVX,COP,EOG,SLB,MPC,VLO,"
    "BKNG,ABNB,MAR,HLT"
)
DEFAULT_SYMBOLS_VOLATILE = (
    "SOXL,SOXS,TQQQ,SQQQ,SPXL,SPXS,LABU,LABD,UVXY,VIXY,"
    "MSTR,SMCI,COIN,TSLA,AMD,NVDA,PLTR,ARM,MARA,RIOT,CLSK,HOOD,"
    "IBIT,FBTC,ETHA,ETHE"
)
DEFAULT_SYMBOLS_DEFENSIVE = (
    "TLT,GLD,XLV,XLU,XLP,MO,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,XLC,XLB,XLRE,"
    "XAR,KRE,XOP,IEF,SHY,TIP,TLH,JNK,AGG,BND,MUB,IGIB,USHY,FLOT,VGIT,VCIT,EMB,"
    "SCHD,VIG,DGRO,HDV,NOBL,VYM,DIVO,JEPI,JEPQ,SPLV,VTV,"
    "JNJ,PG,KO,PEP,MCD,ABBV,ABT,MRK,PFE,T,VZ,O,VICI,MAIN,"
    "ITA,LMT,NOC,RTX,GD,LHX,LDOS"
)
DEFAULT_SYMBOLS_COMMOD_FX_INTL = "DBC,USO,UNG,CORN,WEAT,SLV,CPER,URA,UUP,FXE,FXY,FXB,FXC,FXA,CYB,EUO,YCS,UDN,EFA,EEM,EWJ,FXI,EWZ,INDA,IXUS"
DEFAULT_DIVIDEND_SYMBOLS = "SCHD,VIG,DGRO,HDV,NOBL,VYM,DIVO,JEPI,JEPQ,SPYD,DIV,FDVV,SCHY,JNJ,PG,KO,PEP,MCD,MO,ABBV,ABT,MRK,PFE,T,VZ,O,VICI,MAIN,XOM,CVX,COP,KMI,MPC,PSX,VLO,EOG,SLB,MSFT,AAPL"
DEFAULT_BOND_SYMBOLS = "TLT,IEF,SHY,TIP,LQD,HYG,JNK,AGG,BND,TLH,MUB,IGIB,USHY,FLOT,VGIT,VCIT,EMB,BIL,SGOV"
DEFAULT_FX_SYMBOLS = "UUP,FXE,FXY,FXB,FXC,FXA,CYB,EUO,YCS,UDN,CEW,DBV"
DEFAULT_FX_CONTEXT_SYMBOLS = "SPY,QQQ,TLT,GLD,UUP,FXE,FXY,FXB,FXC,FXA,EFA,EEM,USO,DBC"
SPECIALIZED_SLEEVE_PROFILES = (
    "volatility",
    "pairs_correlation",
    "stat_arb_market_neutral",
    "earnings_event",
    "commodity_inflation",
    "international_macro",
    "market_making_liquidity",
    "short_bias_hedge",
    "single_name_options_event",
    "rates_credit_macro",
    "cash_rotation_tactical",
    "futures_index_intraday",
    "futures_rates_curve",
    "futures_commodity_macro",
    "crypto_futures_basis",
    "futures_event_reaction",
    "options_on_futures",
    "options_on_futures_aggressive",
    "compound_options",
    "swaptions",
    "structured_products",
    "synthetic_cdo",
    "cdo_squared",
    "cdo_cubed",
    "variance_volatility_swaps",
    "barrier_lookback_options",
    "second_third_order_greeks",
    "high_frequency_market_making",
    "tail_risk_parity",
    "black_swan_hedging",
    "sovereign_debt_macro",
    "gamma_scalping",
    "statistical_arbitrage",
    "vanna_volga_hedging",
    "order_flow_market_microstructure",
    "dispersion_trading",
    "cross_asset_basis_training",
    "volatility_arbitrage",
    "rainbow_options",
    "quant_pricing_models",
    "state_space_models",
    "tail_dependency_risk",
    "optimization_research",
    "nlp_sentiment_agents",
    "adaptive_architectures",
    "adversarial_ml_security",
    "low_latency_orchestration",
    "alternative_data_ingestion",
    "privacy_zkp_controls",
    "gpu_quant_acceleration",
    "qemc_path_volatility",
    "transport_topology_research",
    "neural_sde_kan_hedging",
    "order_flow_toxicity",
    "signature_hawkes_generators",
    "crowd_physics_games",
    "lit_order_book_transformers",
    "critic_hmm_pinsde",
    "causal_omni_symbolic",
    "rlbf_dms_equivariant",
    "arbitrage_execution_safety",
    "geometry_spillover_durability",
    "institutional_data_plumbing",
    "lobdif_crisis_microstructure",
    "macro_crisis_scenario_lab",
    "xva_counterparty_margin",
    "credit_derivatives_cdx_cds",
    "securitized_products_mbs_abs_clo",
    "repo_securities_lending",
    "market_data_tape_normalization",
    "provider_adapter_verification",
    "proof_quantum_formal_backends",
    "model_risk_validation",
    "transaction_cost_slippage_intelligence",
    "portfolio_construction",
    "event_intelligence",
    "feature_quality_data_confidence",
    "liquidity_regime",
    "system_governor_expansion",
    "collateral_margin_liquidity",
    "dealer_positioning_gamma_inventory",
    "etf_flow_creation_redemption",
    "signal_governance_integrity",
    "runtime_capacity_governance",
    "data_plane_backpressure_resilience",
    "halt_recovery_stability",
    "expansion_quality_governance",
    "neural_operator_surrogates",
    "uncertainty_robust_control",
    "causal_regime_discovery",
    "martingale_flow_pricing",
    "alpha_research_os",
    "research_meta_governance",
)


@dataclass
class JobSpec:
    name: str
    cmd: list[str]
    env: dict[str, str]
    breaker_group: str
    heartbeat_path: Path | None = None
    heartbeat_stale_seconds: int = 0
    heartbeat_startup_grace_seconds: int = 0


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _current_nice_value() -> int:
    try:
        return int(os.nice(0))
    except Exception:
        return 0


def _relative_nice_increment_for_target(target_nice: int, parent_nice: int | None = None) -> int:
    current = _current_nice_value() if parent_nice is None else int(parent_nice)
    target = max(int(target_nice), 0)
    if target <= current:
        return 0
    return min(target - current, 20)


def _nice_prefix_for_target(target_nice: int, parent_nice: int | None = None) -> list[str]:
    return ["nice", "-n", str(_relative_nice_increment_for_target(target_nice, parent_nice))]


def _paper_trade_lock_enabled() -> bool:
    lock_override = os.getenv("PAPER_TRADE_LOCK_PATH", "").strip()
    lock_path = Path(lock_override) if lock_override else PAPER_TRADE_LOCK_PATH
    return _env_flag("PAPER_TRADE_LOCK", "0") or lock_path.exists()


def _apply_paper_trade_lock(args: argparse.Namespace) -> bool:
    if not _paper_trade_lock_enabled():
        return False

    os.environ["PAPER_TRADE_LOCK"] = "1"
    os.environ["TOP_BOT_ENABLE_LIVE_EXECUTION"] = "0"
    os.environ["EXECUTION_LANE_LIVE_ENABLED"] = "0"
    os.environ["RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR"] = "0"
    if bool(getattr(args, "with_live_executor", False)):
        args.with_live_executor = False
        print("[PaperTradeLock] live executor disabled while paper trade lock is active.")
        _emit_incident_snapshot("paper_trade_lock_disabled_live_executor", "run_all_sleeves_startup")
    return True


def _global_trading_halt_enabled() -> bool:
    return _env_flag("GLOBAL_TRADING_HALT", "0") or HALT_FLAG_PATH.exists()


def _route_storage_or_fail() -> dict[str, Any] | None:
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from core.storage_router import describe_storage_routing, route_runtime_storage

        routing = route_runtime_storage(PROJECT_ROOT)
        print(describe_storage_routing(routing))
        return {
            "mode": str(getattr(routing, "mode", "") or ""),
            "active_root": str(getattr(routing, "active_root", "") or ""),
        }
    except Exception as exc:
        print(f"[StorageRoute] startup blocked err={exc}")
        return None


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _storage_disk_probe_path(storage_route: dict[str, Any] | None) -> Path:
    active_root = ""
    if isinstance(storage_route, dict):
        active_root = str(storage_route.get("active_root") or "").strip()
    if active_root:
        return Path(active_root)
    return PROJECT_ROOT


def _disk_gate_status(
    storage_route: dict[str, Any] | None,
    *,
    local_min_free_gb: float,
    storage_min_free_gb: float,
) -> dict[str, Any]:
    local_probe = PROJECT_ROOT
    storage_probe = _storage_disk_probe_path(storage_route)
    local_free_gb = _disk_free_gb(local_probe)
    storage_free_gb = _disk_free_gb(storage_probe)
    local_min = max(float(local_min_free_gb), 0.1)
    storage_min = max(float(storage_min_free_gb), 0.1)
    blocked_reasons: list[str] = []
    if local_free_gb < local_min:
        blocked_reasons.append("local_project_disk")
    if storage_free_gb < storage_min:
        blocked_reasons.append("runtime_storage_disk")
    return {
        "ok": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "local_probe": str(local_probe),
        "local_free_gb": local_free_gb,
        "local_min_free_gb": local_min,
        "storage_probe": str(storage_probe),
        "storage_free_gb": storage_free_gb,
        "storage_min_free_gb": storage_min,
    }


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_env_override(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _process_fanout_policy(path: Path = PROCESS_FANOUT_OVERRIDE_PATH) -> dict[str, Any]:
    values = _read_env_override(path)
    active = str(values.get("PROCESS_FANOUT_GUARD_ACTIVE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    return {
        "active": active,
        "reason": values.get("PROCESS_FANOUT_GUARD_REASON", ""),
        "specialized_enabled": values.get("RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES", "1") != "0",
        "aggressive_enabled": values.get("OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE", "1") != "0",
        "dividend_capture_enabled": values.get("RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE", "1") != "0",
    }


def _job_parked_by_fanout_policy(name: str, policy: dict[str, Any]) -> bool:
    if not bool(policy.get("active", False)):
        return False
    if name in SPECIALIZED_SLEEVE_PROFILES and not bool(policy.get("specialized_enabled", True)):
        return True
    if name == "aggressive_modes" and not bool(policy.get("aggressive_enabled", True)):
        return True
    if name == "dividend_capture" and not bool(policy.get("dividend_capture_enabled", True)):
        return True
    return False


def _apply_process_fanout_policy_to_args(args: argparse.Namespace, policy: dict[str, Any]) -> list[str]:
    if not bool(policy.get("active", False)):
        return []
    changes: list[str] = []
    if not bool(policy.get("specialized_enabled", True)) and bool(getattr(args, "with_specialized_sleeves", False)):
        args.with_specialized_sleeves = False
        changes.append("specialized_sleeves")
    if not bool(policy.get("aggressive_enabled", True)) and bool(getattr(args, "with_aggressive_modes", False)):
        args.with_aggressive_modes = False
        changes.append("aggressive_modes")
    if not bool(policy.get("dividend_capture_enabled", True)) and bool(getattr(args, "with_dividend_capture", False)):
        args.with_dividend_capture = False
        changes.append("dividend_capture")
    return changes


def _job_heartbeat_stale(
    spec: JobSpec,
    *,
    started_at: float,
    now_ts: float | None = None,
) -> tuple[bool, str]:
    heartbeat_path = spec.heartbeat_path
    stale_after = max(int(spec.heartbeat_stale_seconds or 0), 0)
    if heartbeat_path is None or stale_after <= 0:
        return False, ""

    now_epoch = float(now_ts if now_ts is not None else time.time())
    startup_grace = max(int(spec.heartbeat_startup_grace_seconds or 0), 0)
    elapsed = max(now_epoch - float(started_at or 0.0), 0.0)
    if elapsed < float(startup_grace):
        return False, "startup_grace"

    if not heartbeat_path.exists():
        return True, "heartbeat_missing"

    payload = _read_json(heartbeat_path)
    if bool(payload.get("stale", False)):
        return True, "payload_stale"

    try:
        file_age = max(now_epoch - heartbeat_path.stat().st_mtime, 0.0)
    except Exception:
        return True, "heartbeat_stat_failed"

    if file_age >= float(stale_after):
        return True, f"heartbeat_age={file_age:.1f}s"
    return False, ""


def _acquire_singleton_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = "unknown"
        fh.close()
        print(f"[AllSleevesLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={time.time():.0f} cmd={' '.join(sys.argv)}")
    fh.flush()
    print(f"[AllSleevesLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _stream(name: str, pipe) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(f"[{name}] {line}")
    pipe.close()


def _spawn(spec: JobSpec) -> subprocess.Popen:
    proc = subprocess.Popen(
        spec.cmd,
        cwd=str(PROJECT_ROOT),
        env=spec.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"Started {spec.name} pid={proc.pid}")
    t = threading.Thread(target=_stream, args=(spec.name, proc.stdout), daemon=True)
    t.start()
    return proc


def _launcher_job_class(name: str) -> str:
    if name in {"baseline_parallel", "dividend", "dividend_capture", "bond", "fx"}:
        return "core_collection"
    if name in {"paper_executor", "live_executor"}:
        return "execution_lane"
    if name == "aggressive_modes":
        return "aggressive_collection"
    if name in SPECIALIZED_SLEEVE_PROFILES:
        return "specialized_collection"
    return "other"


def _launcher_readiness_contract(
    *,
    jobs: list[dict[str, Any]],
    problem_jobs: list[dict[str, Any]],
    quarantined_jobs: dict[str, dict[str, object]],
    phase: str,
    overall_status: str,
    repair_active: bool,
    policy_parked_jobs: set[str],
    clean_exited_jobs: set[str],
) -> dict[str, Any]:
    class_counts: dict[str, dict[str, int]] = {}
    for row in jobs:
        name = str(row.get("name") or "")
        job_class = _launcher_job_class(name)
        row["job_class"] = job_class
        bucket = class_counts.setdefault(job_class, {"expected": 0, "running": 0, "stable_non_running": 0, "problem": 0})
        bucket["expected"] += 1
        if row.get("state") == "running":
            bucket["running"] += 1
        elif bool(row.get("policy_parked", False)) or bool(row.get("clean_exited", False)):
            bucket["stable_non_running"] += 1

    problem_names = {str(row.get("name") or "") for row in problem_jobs}
    for row in jobs:
        name = str(row.get("name") or "")
        if name in problem_names:
            class_counts.setdefault(_launcher_job_class(name), {"expected": 0, "running": 0, "stable_non_running": 0, "problem": 0})["problem"] += 1

    core_problem_jobs = [
        row
        for row in problem_jobs
        if _launcher_job_class(str(row.get("name") or "")) in {"core_collection", "execution_lane"}
    ]
    restart_pressure_jobs = [
        row
        for row in jobs
        if int(row.get("restart_count_last_hour", 0) or 0) >= 3
    ]
    score = 100.0
    score -= min(float(len(problem_jobs)) * 18.0, 54.0)
    score -= min(float(len(core_problem_jobs)) * 18.0, 36.0)
    score -= min(float(len(quarantined_jobs)) * 20.0, 40.0)
    score -= min(float(len(restart_pressure_jobs)) * 8.0, 24.0)
    if phase == "starting":
        score -= 6.0
    if overall_status == "blocked":
        score -= 20.0
    score = round(max(score, 0.0), 1)

    if phase == "starting":
        readiness_status = "starting"
    elif core_problem_jobs or overall_status == "blocked":
        readiness_status = "repair_core_first"
    elif problem_jobs or repair_active:
        readiness_status = "repair_optional_first"
    elif policy_parked_jobs or clean_exited_jobs:
        readiness_status = "stable_with_parked_lanes"
    else:
        readiness_status = "ready"

    can_expand = bool(
        phase == "running"
        and overall_status == "ready"
        and not problem_jobs
        and not quarantined_jobs
        and not restart_pressure_jobs
        and not policy_parked_jobs
        and not clean_exited_jobs
    )
    if can_expand:
        max_new_collect_only_sleeves = 10
    elif readiness_status == "stable_with_parked_lanes" and not core_problem_jobs:
        max_new_collect_only_sleeves = 3
    else:
        max_new_collect_only_sleeves = 0

    exact_needs = []
    for row in problem_jobs[:12]:
        name = str(row.get("name") or "")
        state = str(row.get("state") or "")
        reason = "quarantined" if bool(row.get("quarantined", False)) else state
        exact_needs.append(
            {
                "target": name,
                "job_class": _launcher_job_class(name),
                "blocker": reason,
                "exact_file": str(LAUNCHER_HEALTH_PATH),
                "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                "expected_impact": "repair or quarantine the specific sleeve lane without enabling live execution",
                "risk_level": "medium" if _launcher_job_class(name) in {"core_collection", "execution_lane"} else "low",
                "when_to_stop": "stop when this job is running, cleanly parked, or explicitly quarantined with no restart storm",
            }
        )
    if not exact_needs and not can_expand and phase == "starting":
        exact_needs.append(
            {
                "target": "all_sleeves",
                "job_class": "launcher_parent",
                "blocker": "startup_in_progress",
                "exact_file": str(LAUNCHER_HEALTH_PATH),
                "exact_command": ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--json"],
                "expected_impact": "wait for the parent to finish spawning children before declaring a launcher outage",
                "risk_level": "low",
                "when_to_stop": "stop when phase=running or repair_packet.status=needs_repair",
            }
        )

    return {
        "active": True,
        "mode": "sleeve_launcher_readiness_expansion_v2",
        "paper_only": True,
        "live_execution_allowed": False,
        "readiness_status": readiness_status,
        "readiness_score": score,
        "can_expand_collection_sleeves": can_expand or max_new_collect_only_sleeves > 0,
        "max_new_collect_only_sleeves": int(max_new_collect_only_sleeves),
        "class_counts": class_counts,
        "core_problem_count": len(core_problem_jobs),
        "problem_job_count": len(problem_jobs),
        "restart_pressure_job_count": len(restart_pressure_jobs),
        "policy_parked_job_count": len(policy_parked_jobs),
        "clean_exited_job_count": len(clean_exited_jobs),
        "quarantined_job_count": len(quarantined_jobs),
        "exact_needs": exact_needs,
        "auto_repair_sequence": [
            "refresh launcher health and watchdog intelligence",
            "clean orphaned child processes only when the parent is missing or hollow",
            "restart the all-sleeves parent in read-only paper-data mode when startup gates allow it",
            "quarantine repeat-crashing children before they create a restart storm",
            "admit new collection-only sleeves only after core and execution lanes stay stable",
        ],
        "expansion_rules": [
            "new sleeves are collection-only by default",
            "live execution remains disabled by this launcher contract",
            "core collection and paper executor stability come before optional sleeve widening",
            "policy-parked and clean-exited lanes are not treated as outages",
            "restart pressure reduces expansion slots before it becomes a storm",
        ],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-mechanics", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        ],
    }


def _launcher_health_payload(
    *,
    specs: dict[str, JobSpec],
    procs: dict[str, subprocess.Popen],
    proc_started_at: dict[str, float],
    restart_history: dict[str, list[float]],
    quarantined_jobs: dict[str, dict[str, object]],
    launcher_started_at: float,
    phase: str,
    note: str = "",
    policy_parked_jobs: set[str] | None = None,
    clean_exited_jobs: set[str] | None = None,
) -> dict[str, Any]:
    now = time.time()
    parked_jobs = set(policy_parked_jobs or set())
    clean_jobs = set(clean_exited_jobs or set())
    jobs: list[dict[str, Any]] = []
    running_count = 0
    exited_count = 0
    missing_count = 0

    for name, spec in specs.items():
        proc = procs.get(name)
        code: int | None = None
        pid = 0
        if proc is None:
            missing_count += 1
            state = "missing"
        else:
            pid = int(proc.pid)
            code = proc.poll()
            if code is None:
                running_count += 1
                state = "running"
            else:
                exited_count += 1
                state = "exited"

        started = float(proc_started_at.get(name, 0.0) or 0.0)
        jobs.append(
            {
                "name": name,
                "state": state,
                "pid": pid,
                "exit_code": code,
                "uptime_seconds": round(max(now - started, 0.0), 3) if started > 0 else 0.0,
                "restart_count_last_hour": len(restart_history.get(name, [])),
                "quarantined": name in quarantined_jobs,
                "policy_parked": name in parked_jobs,
                "clean_exited": name in clean_jobs,
                "breaker_group": spec.breaker_group,
            }
        )

    expected_count = len(specs)
    quarantined_count = len(quarantined_jobs)
    stable_non_running_count = sum(
        1
        for row in jobs
        if row.get("state") != "running"
        and (bool(row.get("policy_parked", False)) or bool(row.get("clean_exited", False)))
    )
    all_non_running_stable = (
        expected_count > 0
        and running_count + stable_non_running_count >= expected_count
        and missing_count == 0
        and quarantined_count == 0
    )
    if phase in {"stopped", "blocked"}:
        overall_status = phase
    elif expected_count <= 0:
        overall_status = "blocked"
    elif running_count >= expected_count and exited_count == 0 and missing_count == 0 and quarantined_count == 0:
        overall_status = "ready"
    elif running_count > 0 and all_non_running_stable:
        overall_status = "ready"
    elif running_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "blocked"

    if phase == "starting":
        problem_jobs = [
            row
            for row in jobs
            if not bool(row.get("policy_parked", False))
            and not bool(row.get("clean_exited", False))
            and (
                bool(row.get("quarantined", False))
            or (row.get("state") == "exited" and row.get("exit_code") not in {0, None})
            )
        ]
    else:
        problem_jobs = [
            row
            for row in jobs
            if not bool(row.get("policy_parked", False))
            and not bool(row.get("clean_exited", False))
            and (row.get("state") in {"missing", "exited"} or bool(row.get("quarantined", False)))
        ]
    if phase == "starting":
        repair_active = bool(problem_jobs or quarantined_count > 0)
    else:
        stable_non_running_jobs = bool(parked_jobs or clean_jobs)
        repair_active = bool(
            problem_jobs
            or quarantined_count > 0
            or (overall_status in {"degraded", "blocked"} and not stable_non_running_jobs)
        )
    repair_infrabots = [
        {
            **bot,
            "assigned": True,
            "active": repair_active,
        }
        for bot in SLEEVE_LAUNCHER_REPAIR_INFRABOTS
    ]
    repair_packet = {
        "status": "needs_repair" if repair_active and phase not in {"stopped"} else "clear",
        "problem_job_count": len(problem_jobs),
        "problem_jobs": [
            {
                "name": str(row.get("name") or ""),
                "state": str(row.get("state") or ""),
                "exit_code": row.get("exit_code"),
                "quarantined": bool(row.get("quarantined", False)),
            }
            for row in problem_jobs
        ],
        "infrabots": repair_infrabots,
        "recommended_commands": [
            [str(VENV_PY), "scripts/ops/process_watchdog.py", "--json"],
            ["./scripts/ops/opsctl.sh", "post-restart-settle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        ]
        if repair_active and phase not in {"stopped"}
        else [],
        "policy": "repair_read_only_sleeve_collection_without_enabling_live_execution",
    }
    launcher_readiness_contract = _launcher_readiness_contract(
        jobs=jobs,
        problem_jobs=problem_jobs,
        quarantined_jobs=quarantined_jobs,
        phase=phase,
        overall_status=overall_status,
        repair_active=repair_active,
        policy_parked_jobs=parked_jobs,
        clean_exited_jobs=clean_jobs,
    )
    repair_packet["launcher_readiness_status"] = launcher_readiness_contract["readiness_status"]
    repair_packet["launcher_readiness_score"] = launcher_readiness_contract["readiness_score"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "phase": phase,
        "note": note,
        "launcher_pid": os.getpid(),
        "launcher_uptime_seconds": round(max(now - launcher_started_at, 0.0), 3),
        "expected_job_count": expected_count,
        "running_job_count": running_count,
        "exited_job_count": exited_count,
        "missing_job_count": missing_count,
        "quarantined_job_count": quarantined_count,
        "policy_parked_job_count": len(parked_jobs),
        "policy_parked_jobs": sorted(parked_jobs),
        "clean_exited_job_count": len(clean_jobs),
        "clean_exited_jobs": sorted(clean_jobs),
        "quarantined_jobs": quarantined_jobs,
        "launcher_readiness_contract": launcher_readiness_contract,
        "repair_packet": repair_packet,
        "repair_infrabots": repair_infrabots,
        "jobs": jobs,
    }


def _write_launcher_health(payload: dict[str, Any], path: Path = LAUNCHER_HEALTH_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[LauncherHealth] warning failed err={exc}")


def _stop_processes(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    for proc in procs.values():
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


def _handle_shutdown_signal(signum, _frame) -> None:
    try:
        signal_name = signal.Signals(signum).name
    except Exception:
        signal_name = f"signal_{signum}"
    print(f"Received {signal_name}; stopping all sleeves...")
    raise KeyboardInterrupt()


def _install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_shutdown_signal)


def _within_restart_budget(restarts: list[float], max_restarts_per_hour: int) -> bool:
    now = time.time()
    one_hour_ago = now - 3600
    while restarts and restarts[0] < one_hour_ago:
        restarts.pop(0)
    return len(restarts) < max_restarts_per_hour


def _read_one_numbers(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _breaker_reasons(metrics: dict, args, *, runtime_seconds: float = 0.0) -> tuple[list[str], str]:
    reasons: list[str] = []
    dq = _safe_float(metrics.get("data_quality_score"), 0.0)
    blocked = _safe_float(metrics.get("combined_blocked_rate"), 0.0)
    data_quality_grace_seconds = max(float(getattr(args, "breaker_data_quality_grace_seconds", 0.0) or 0.0), 0.0)

    if dq < args.breaker_min_data_quality and runtime_seconds >= data_quality_grace_seconds:
        reasons.append(f"data_quality_low:{dq:.2f}")
    if blocked > args.breaker_max_blocked_rate:
        reasons.append(f"blocked_rate_high:{blocked:.4f}")

    broker_domain = "stocks" if args.broker == "schwab" else "crypto"
    pnl_key = "stocks_pnl_proxy" if broker_domain == "stocks" else "crypto_pnl_proxy"
    pnl_val = _safe_float(metrics.get(pnl_key), 0.0)
    if pnl_val < args.breaker_min_pnl_proxy:
        reasons.append(f"{pnl_key}_low:{pnl_val:.6f}")

    return reasons, broker_domain


def _emit_incident_snapshot(reason: str, detail: str = "") -> None:
    if not DEBUG_SNAPSHOT_SCRIPT.exists():
        return
    try:
        proc = subprocess.run([str(DEBUG_SNAPSHOT_SCRIPT)], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        msg = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        print(f"[IncidentSnapshot] reason={reason} detail={detail} rc={proc.returncode} note={msg[0] if msg else ''}")
    except Exception as exc:
        print(f"[IncidentSnapshot] failed reason={reason} err={exc}")


def _capture_full_run_config(args: argparse.Namespace) -> None:
    try:
        if CAPTURE_CONFIG_SCRIPT.exists() and VENV_PY.exists():
            subprocess.run([str(VENV_PY), str(CAPTURE_CONFIG_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)

        keys = [
            "MARKET_DATA_ONLY",
            "ALLOW_ORDER_EXECUTION",
            "DATA_BROKER",
            "SHADOW_SYMBOLS_CORE",
            "SHADOW_SYMBOLS_VOLATILE",
            "SHADOW_SYMBOLS_DEFENSIVE",
            "SHADOW_SYMBOLS_COMMOD_FX_INTL",
            "ASYNC_PIPELINE_WORKERS",
            "SHADOW_LOOP_INTERVAL",
            "ADAPTIVE_INTERVAL_MAX_SECONDS",
            "CANARY_MAX_WEIGHT",
            "GLOBAL_TRADING_HALT",
        ]
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "run_all_sleeves.py",
            "argv": sys.argv,
            "args": vars(args),
            "env": {k: os.getenv(k, "") for k in keys},
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()[:16]

        out_dir = PROJECT_ROOT / "governance" / "session_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"all_sleeves_config_{stamp}.json"
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        (out_dir / "all_sleeves_latest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[ConfigFreeze] hash={payload['config_hash']} file={out}")
    except Exception as exc:
        print(f"[ConfigFreeze] warning failed: {exc}")


def _run_preflight(args: argparse.Namespace) -> bool:
    if not PREFLIGHT_SCRIPT.exists():
        return True
    cmd = [
        str(VENV_PY),
        str(PREFLIGHT_SCRIPT),
        "--broker",
        args.broker,
        "--symbols-core",
        args.symbols_core,
        "--symbols-volatile",
        args.symbols_volatile,
        "--symbols-defensive",
        args.symbols_defensive,
    ]
    if args.simulate:
        cmd.append("--simulate")
    if not getattr(args, "strict_preflight_duplicates", False):
        cmd.append("--allow-running")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    if out:
        print(out)
    if proc.returncode != 0:
        _emit_incident_snapshot("preflight_failed", f"rc={proc.returncode}")
        return False
    return True


def main() -> int:
    _install_signal_handlers()

    parser = argparse.ArgumentParser(description="Run all Schwab sleeves together: baseline + dividend + dividend capture + bond + optional FX (+ optional aggressive modes).")
    parser.add_argument("--simulate", action="store_true", help="Run all sleeves in simulation mode.")
    parser.add_argument("--with-aggressive-modes", action="store_true", help="Also run intraday+swing aggressive modes.")
    parser.add_argument(
        "--with-dividend-capture",
        action="store_true",
        default=_env_flag("RUN_ALL_SLEEVES_WITH_DIVIDEND_CAPTURE", "1"),
        help="Also run the dedicated ex-dividend capture paper lane.",
    )
    parser.add_argument(
        "--with-fx",
        action="store_true",
        default=_env_flag("RUN_ALL_SLEEVES_WITH_FX", os.getenv("FX_SHADOW_ENABLED", "0")),
        help="Also run the paper-only FX proxy sleeve.",
    )
    parser.add_argument(
        "--with-paper-executor",
        action="store_true",
        default=_env_flag("RUN_ALL_SLEEVES_WITH_PAPER_EXECUTOR", "1"),
        help="Run the standalone paper execution lane consumer.",
    )
    parser.add_argument(
        "--with-live-executor",
        action="store_true",
        default=_env_flag("RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR", "0"),
        help="Run the standalone live execution lane consumer.",
    )
    parser.add_argument(
        "--with-specialized-sleeves",
        action="store_true",
        default=_env_flag("RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES", "1"),
        help="Run collect-only volatility, pairs, stat-arb, earnings, commodity, and international macro sleeves.",
    )
    parser.add_argument("--parallel-interval-seconds", type=int, default=int(os.getenv("SHADOW_LOOP_INTERVAL", "15")))
    parser.add_argument("--dividend-interval-seconds", type=int, default=int(os.getenv("DIVIDEND_SHADOW_INTERVAL", "60")))
    parser.add_argument("--dividend-capture-interval-seconds", type=int, default=int(os.getenv("DIVIDEND_CAPTURE_SHADOW_INTERVAL", os.getenv("DIVIDEND_SHADOW_INTERVAL", "60"))))
    parser.add_argument("--bond-interval-seconds", type=int, default=int(os.getenv("BOND_SHADOW_INTERVAL", "120")))
    parser.add_argument("--fx-interval-seconds", type=int, default=int(os.getenv("FX_SHADOW_INTERVAL", "45")))
    parser.add_argument("--specialized-interval-seconds", type=int, default=int(os.getenv("SPECIALIZED_SLEEVE_INTERVAL", "120")))
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SHADOW_LOOP_MAX_ITERS", "0")))
    parser.add_argument("--symbols-core", default=os.getenv("SHADOW_SYMBOLS_CORE", DEFAULT_SYMBOLS_CORE))
    parser.add_argument("--symbols-volatile", default=os.getenv("SHADOW_SYMBOLS_VOLATILE", DEFAULT_SYMBOLS_VOLATILE))
    parser.add_argument(
        "--symbols-defensive",
        default=(
            os.getenv("SHADOW_SYMBOLS_DEFENSIVE", DEFAULT_SYMBOLS_DEFENSIVE)
            + ","
            + os.getenv("SHADOW_SYMBOLS_COMMOD_FX_INTL", DEFAULT_SYMBOLS_COMMOD_FX_INTL)
        ).strip(","),
    )
    parser.add_argument("--dividend-symbols", default=os.getenv("DIVIDEND_SYMBOLS", DEFAULT_DIVIDEND_SYMBOLS))
    parser.add_argument("--bond-symbols", default=os.getenv("BOND_SYMBOLS", DEFAULT_BOND_SYMBOLS))
    parser.add_argument("--fx-symbols", default=os.getenv("FX_SYMBOLS", DEFAULT_FX_SYMBOLS))
    parser.add_argument("--fx-context-symbols", default=os.getenv("FX_CONTEXT_SYMBOLS", DEFAULT_FX_CONTEXT_SYMBOLS))
    parser.add_argument("--restart-delay-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_RESTART_DELAY", "3")))
    parser.add_argument("--max-restarts-per-hour", type=int, default=int(os.getenv("ALL_SLEEVES_MAX_RESTARTS_PER_HOUR", "40")))
    parser.add_argument(
        "--clean-exit-retry-seconds",
        type=int,
        default=int(os.getenv("ALL_SLEEVES_CLEAN_EXIT_RETRY_SECONDS", "1800")),
        help="Retry sleeves that exit cleanly after this many seconds, usually after a session gate reopens.",
    )
    parser.add_argument("--no-restart-on-exit", dest="restart_on_exit", action="store_false", default=True)
    parser.add_argument(
        "--strict-preflight-duplicates",
        action="store_true",
        default=os.getenv("RUN_ALL_SLEEVES_STRICT_PREFLIGHT_DUPLICATES", "0").strip() == "1",
        help="Fail preflight when a parallel launcher is already running.",
    )

    parser.add_argument("--nice-baseline", type=int, default=int(os.getenv("SLEEVE_NICE_BASELINE", "6")))
    parser.add_argument("--nice-dividend", type=int, default=int(os.getenv("SLEEVE_NICE_DIVIDEND", "10")))
    parser.add_argument("--nice-dividend-capture", type=int, default=int(os.getenv("SLEEVE_NICE_DIVIDEND_CAPTURE", os.getenv("SLEEVE_NICE_DIVIDEND", "10"))))
    parser.add_argument("--nice-bond", type=int, default=int(os.getenv("SLEEVE_NICE_BOND", "10")))
    parser.add_argument("--nice-fx", type=int, default=int(os.getenv("SLEEVE_NICE_FX", "9")))
    parser.add_argument("--nice-specialized", type=int, default=int(os.getenv("SLEEVE_NICE_SPECIALIZED", "12")))
    parser.add_argument("--nice-aggressive", type=int, default=int(os.getenv("SLEEVE_NICE_AGGRESSIVE", "5")))
    parser.add_argument("--workers-baseline", type=int, default=int(os.getenv("SLEEVE_WORKERS_BASELINE", os.getenv("ASYNC_PIPELINE_WORKERS", "4"))))
    parser.add_argument("--workers-dividend", type=int, default=int(os.getenv("SLEEVE_WORKERS_DIVIDEND", "2")))
    parser.add_argument("--workers-dividend-capture", type=int, default=int(os.getenv("SLEEVE_WORKERS_DIVIDEND_CAPTURE", os.getenv("SLEEVE_WORKERS_DIVIDEND", "2"))))
    parser.add_argument("--workers-bond", type=int, default=int(os.getenv("SLEEVE_WORKERS_BOND", "2")))
    parser.add_argument("--workers-fx", type=int, default=int(os.getenv("SLEEVE_WORKERS_FX", "2")))
    parser.add_argument("--workers-specialized", type=int, default=int(os.getenv("SLEEVE_WORKERS_SPECIALIZED", "1")))
    parser.add_argument("--workers-aggressive", type=int, default=int(os.getenv("SLEEVE_WORKERS_AGGRESSIVE", "3")))
    parser.add_argument(
        "--launcher-health-seconds",
        type=int,
        default=int(os.getenv("ALL_SLEEVES_HEALTH_WRITE_SECONDS", "15")),
        help="Write all-sleeves launcher health at this interval.",
    )

    parser.add_argument("--disable-circuit-breakers", action="store_true")
    parser.add_argument("--breaker-one-numbers-path", default=str(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"))
    parser.add_argument("--breaker-check-interval-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_CHECK_SECONDS", "60")))
    parser.add_argument("--breaker-consecutive-breaches", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_STREAK", "2")))
    parser.add_argument("--breaker-cooldown-seconds", type=int, default=int(os.getenv("ALL_SLEEVES_BREAKER_COOLDOWN", "300")))
    parser.add_argument(
        "--breaker-startup-grace-seconds",
        type=int,
        default=int(os.getenv("ALL_SLEEVES_BREAKER_STARTUP_GRACE_SECONDS", "180")),
        help="Skip all one-number breaker checks while the expanded feed stack is warming up.",
    )
    parser.add_argument(
        "--breaker-data-quality-grace-seconds",
        type=int,
        default=int(os.getenv("ALL_SLEEVES_BREAKER_DATA_QUALITY_GRACE_SECONDS", "900")),
        help="Ignore low data-quality score until fresh live observations have had time to arrive.",
    )
    parser.add_argument("--breaker-min-data-quality", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MIN_DQ", "75")))
    parser.add_argument("--breaker-max-blocked-rate", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MAX_BLOCKED", "0.35")))
    parser.add_argument("--breaker-min-pnl-proxy", type=float, default=float(os.getenv("ALL_SLEEVES_BREAKER_MIN_PNL", "-0.020")))
    parser.add_argument(
        "--hard-min-free-gb",
        type=float,
        default=float(os.getenv("ALL_SLEEVES_HARD_MIN_FREE_GB", "15")),
        help="Hard startup block if active runtime storage is below this GB threshold.",
    )
    parser.add_argument(
        "--local-hard-min-free-gb",
        type=float,
        default=float(os.getenv("ALL_SLEEVES_LOCAL_HARD_MIN_FREE_GB", "2")),
        help="Hard startup block if the local project volume is below this GB threshold.",
    )

    args = parser.parse_args()
    startup_fanout_policy = _process_fanout_policy()
    fanout_policy_changes = _apply_process_fanout_policy_to_args(args, startup_fanout_policy)
    if fanout_policy_changes:
        print(
            "[ProcessFanoutPolicy] startup_parking "
            f"reason={startup_fanout_policy.get('reason') or 'process_fanout_guard_active'} "
            f"disabled={','.join(fanout_policy_changes)}"
        )
    paper_trade_lock_active = _apply_paper_trade_lock(args)

    storage_route = _route_storage_or_fail()
    if not storage_route:
        return 6

    if _global_trading_halt_enabled():
        print("GLOBAL_TRADING_HALT=1 set; refusing to start all sleeves.")
        _emit_incident_snapshot("global_halt_refusal", "startup")
        return 3

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2

    disk_gate = _disk_gate_status(
        storage_route,
        local_min_free_gb=float(args.local_hard_min_free_gb),
        storage_min_free_gb=float(args.hard_min_free_gb),
    )
    if not bool(disk_gate.get("ok", False)):
        print(
            "[HardDiskGate] blocked "
            f"reasons={','.join(disk_gate.get('blocked_reasons') or [])} "
            f"local_free_gb={float(disk_gate.get('local_free_gb') or 0.0):.2f} "
            f"local_min_required_gb={float(disk_gate.get('local_min_free_gb') or 0.0):.2f} "
            f"storage_probe={disk_gate.get('storage_probe')} "
            f"storage_free_gb={float(disk_gate.get('storage_free_gb') or 0.0):.2f} "
            f"storage_min_required_gb={float(disk_gate.get('storage_min_free_gb') or 0.0):.2f}"
        )
        _emit_incident_snapshot(
            "hard_disk_gate_blocked",
            (
                f"reasons={','.join(disk_gate.get('blocked_reasons') or [])};"
                f"local_free_gb={float(disk_gate.get('local_free_gb') or 0.0):.2f};"
                f"storage_free_gb={float(disk_gate.get('storage_free_gb') or 0.0):.2f}"
            ),
        )
        return 5

    lock_path = Path(os.getenv("ALL_SLEEVES_LOCK_PATH", str(PROJECT_ROOT / "governance" / "all_sleeves.lock")))
    lock_handle = _acquire_singleton_lock(lock_path)
    if lock_handle is None:
        _emit_incident_snapshot("all_sleeves_lock_busy", str(lock_path))
        return 1

    if not _run_preflight(args):
        print("[Preflight] startup blocked.")
        return 4

    _capture_full_run_config(args)

    base_env = os.environ.copy()
    base_env["MARKET_DATA_ONLY"] = "1"
    base_env["ALLOW_ORDER_EXECUTION"] = "0"
    if paper_trade_lock_active:
        base_env["PAPER_TRADE_LOCK"] = "1"
        base_env["TOP_BOT_ENABLE_LIVE_EXECUTION"] = "0"
        base_env["EXECUTION_LANE_LIVE_ENABLED"] = "0"
        base_env["RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR"] = "0"
    base_env["EXECUTION_LANE_ENABLED"] = os.getenv("EXECUTION_LANE_ENABLED", "1")
    base_env["MASTER_EXECUTION_LANE_ENABLED"] = os.getenv("MASTER_EXECUTION_LANE_ENABLED", "1")
    base_env["INLINE_PAPER_EXECUTION_ENABLED"] = os.getenv("INLINE_PAPER_EXECUTION_ENABLED", "0")

    specs: dict[str, JobSpec] = {}

    parent_nice = _current_nice_value()
    print(
        "[SleevePriority] "
        f"parent_nice={parent_nice} "
        f"baseline_target={args.nice_baseline} "
        f"specialized_target={args.nice_specialized} "
        f"aggressive_target={args.nice_aggressive}"
    )

    parallel_cmd = [
        *_nice_prefix_for_target(args.nice_baseline, parent_nice),
        str(VENV_PY), str(PARALLEL_SHADOWS),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.parallel_interval_seconds, 5)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        parallel_cmd.append("--simulate")
    if args.symbols_core:
        parallel_cmd.extend(["--symbols-core", args.symbols_core])
    if args.symbols_volatile:
        parallel_cmd.extend(["--symbols-volatile", args.symbols_volatile])
    if args.symbols_defensive:
        parallel_cmd.extend(["--symbols-defensive", args.symbols_defensive])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_baseline, 1))
    specs["baseline_parallel"] = JobSpec("baseline_parallel", parallel_cmd, env, breaker_group="core")

    dividend_cmd = [
        *_nice_prefix_for_target(args.nice_dividend, parent_nice),
        str(VENV_PY), str(DIVIDEND_SHADOW),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.dividend_interval_seconds, 15)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        dividend_cmd.append("--simulate")
    if args.dividend_symbols:
        dividend_cmd.extend(["--symbols", args.dividend_symbols])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_dividend, 1))
    specs["dividend"] = JobSpec("dividend", dividend_cmd, env, breaker_group="core")

    if args.with_dividend_capture:
        dividend_capture_cmd = [
            *_nice_prefix_for_target(args.nice_dividend_capture, parent_nice),
            str(VENV_PY), str(DIVIDEND_CAPTURE_SHADOW),
            "--broker", args.broker,
            "--interval-seconds", str(max(args.dividend_capture_interval_seconds, 15)),
            "--max-iterations", str(args.max_iterations),
        ]
        if args.simulate:
            dividend_capture_cmd.append("--simulate")
        if args.dividend_symbols:
            dividend_capture_cmd.extend(["--symbols", args.dividend_symbols])
        env = dict(base_env)
        env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_dividend_capture, 1))
        specs["dividend_capture"] = JobSpec("dividend_capture", dividend_capture_cmd, env, breaker_group="core")

    bond_cmd = [
        *_nice_prefix_for_target(args.nice_bond, parent_nice),
        str(VENV_PY), str(BOND_SHADOW),
        "--broker", args.broker,
        "--interval-seconds", str(max(args.bond_interval_seconds, 15)),
        "--max-iterations", str(args.max_iterations),
    ]
    if args.simulate:
        bond_cmd.append("--simulate")
    if args.bond_symbols:
        bond_cmd.extend(["--symbols", args.bond_symbols])
    env = dict(base_env)
    env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_bond, 1))
    specs["bond"] = JobSpec("bond", bond_cmd, env, breaker_group="core")

    if args.with_fx:
        fx_cmd = [
            *_nice_prefix_for_target(args.nice_fx, parent_nice),
            str(VENV_PY), str(FX_SHADOW),
            "--broker", "schwab",
            "--interval-seconds", str(max(args.fx_interval_seconds, 15)),
            "--max-iterations", str(args.max_iterations),
        ]
        if args.simulate:
            fx_cmd.append("--simulate")
        if args.fx_symbols:
            fx_cmd.extend(["--symbols", args.fx_symbols])
        if args.fx_context_symbols:
            fx_cmd.extend(["--context-symbols", args.fx_context_symbols])
        env = dict(base_env)
        env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_fx, 1))
        env["FX_DIRECT_EXECUTION_ENABLED"] = "0"
        env["SCHWAB_FOREX_API_VERIFIED"] = os.getenv("SCHWAB_FOREX_API_VERIFIED", "0")
        specs["fx"] = JobSpec("fx", fx_cmd, env, breaker_group="core")

    if args.with_specialized_sleeves:
        for profile in SPECIALIZED_SLEEVE_PROFILES:
            cmd = [
                *_nice_prefix_for_target(args.nice_specialized, parent_nice),
                str(VENV_PY), str(SPECIALIZED_SLEEVE_SHADOW),
                "--broker", args.broker,
                "--profile", profile,
                "--interval-seconds", str(max(args.specialized_interval_seconds, 30)),
                "--max-iterations", str(args.max_iterations),
            ]
            if args.simulate:
                cmd.append("--simulate")
            env = dict(base_env)
            env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_specialized, 1))
            env["AUTO_RETRAIN_ON_GOVERNANCE"] = "0"
            env["SLEEVE_LIFECYCLE_STATE"] = "data_collection_only"
            env["TRAINING_EXCLUDED_UNTIL_READY"] = "1"
            specs[profile] = JobSpec(profile, cmd, env, breaker_group="core")

    if args.with_aggressive_modes:
        aggressive_cmd = [
            *_nice_prefix_for_target(args.nice_aggressive, parent_nice),
            str(VENV_PY), str(AGGRESSIVE_MODES),
            "--broker", args.broker,
            "--max-iterations", str(args.max_iterations),
        ]
        if args.simulate:
            aggressive_cmd.append("--simulate")
        env = dict(base_env)
        env["ASYNC_PIPELINE_WORKERS"] = str(max(args.workers_aggressive, 1))
        specs["aggressive_modes"] = JobSpec("aggressive_modes", aggressive_cmd, env, breaker_group="core")

    if args.with_paper_executor:
        paper_exec_cmd = [
            *_nice_prefix_for_target(args.nice_baseline, parent_nice),
            str(VENV_PY), str(EXECUTION_LANE),
            "--mode", "paper",
        ]
        env = dict(base_env)
        env["MARKET_DATA_ONLY"] = "0"
        env["ALLOW_ORDER_EXECUTION"] = "1"
        paper_heartbeat_stale_seconds = max(int(os.getenv("OPS_WATCHDOG_PAPER_EXECUTOR_HEARTBEAT_STALE_SECONDS", "240") or 240), 60)
        specs["paper_executor"] = JobSpec(
            "paper_executor",
            paper_exec_cmd,
            env,
            breaker_group="core",
            heartbeat_path=PROJECT_ROOT / "governance" / "health" / "execution_lane_paper_latest.json",
            heartbeat_stale_seconds=paper_heartbeat_stale_seconds,
            heartbeat_startup_grace_seconds=paper_heartbeat_stale_seconds,
        )

    if args.with_live_executor:
        live_exec_cmd = [
            *_nice_prefix_for_target(args.nice_baseline, parent_nice),
            str(VENV_PY), str(EXECUTION_LANE),
            "--mode", "live",
        ]
        env = dict(base_env)
        env["MARKET_DATA_ONLY"] = "0"
        env["ALLOW_ORDER_EXECUTION"] = "1"
        live_heartbeat_stale_seconds = max(int(os.getenv("OPS_WATCHDOG_EXECUTION_HEARTBEAT_STALE_SECONDS", "240") or 240), 60)
        specs["live_executor"] = JobSpec(
            "live_executor",
            live_exec_cmd,
            env,
            breaker_group="core",
            heartbeat_path=PROJECT_ROOT / "governance" / "health" / "execution_lane_live_latest.json",
            heartbeat_stale_seconds=live_heartbeat_stale_seconds,
            heartbeat_startup_grace_seconds=live_heartbeat_stale_seconds,
        )

    procs: dict[str, subprocess.Popen] = {}
    proc_started_at: dict[str, float] = {}
    restart_history: dict[str, list[float]] = {name: [] for name in specs}
    quarantined_jobs: dict[str, dict[str, object]] = {}
    clean_exited_at: dict[str, float] = {}
    breaker_streaks: dict[str, int] = {"core": 0}
    group_disabled_until: dict[str, float] = {"core": 0.0}
    parked_jobs_reported: set[str] = set()
    last_breaker_check_ts = 0.0
    breaker_path = Path(args.breaker_one_numbers_path)
    launcher_started_at = time.time()
    last_health_write_ts = 0.0

    _write_launcher_health(
        _launcher_health_payload(
            specs=specs,
            procs=procs,
            proc_started_at=proc_started_at,
            restart_history=restart_history,
            quarantined_jobs=quarantined_jobs,
            launcher_started_at=launcher_started_at,
            phase="starting",
            note="specs_prepared",
        )
    )

    try:
        for idx, (name, spec) in enumerate(specs.items(), start=1):
            procs[name] = _spawn(spec)
            proc_started_at[name] = time.time()
            if idx == 1 or idx == len(specs) or idx % 10 == 0:
                _write_launcher_health(
                    _launcher_health_payload(
                        specs=specs,
                        procs=procs,
                        proc_started_at=proc_started_at,
                        restart_history=restart_history,
                        quarantined_jobs=quarantined_jobs,
                        launcher_started_at=launcher_started_at,
                        phase="starting",
                        note=f"spawned_{idx}_of_{len(specs)}",
                    )
                )
            time.sleep(0.8)

        print("All sleeves live:", ", ".join(specs.keys()))
        _write_launcher_health(
            _launcher_health_payload(
                specs=specs,
                procs=procs,
                proc_started_at=proc_started_at,
                restart_history=restart_history,
                quarantined_jobs=quarantined_jobs,
                launcher_started_at=launcher_started_at,
                phase="running",
                note="initial_spawn_complete",
            )
        )
        while True:
            fanout_policy = _process_fanout_policy()
            policy_parked_jobs = {
                name
                for name in specs
                if _job_parked_by_fanout_policy(name, fanout_policy)
            }
            if not policy_parked_jobs:
                parked_jobs_reported.clear()

            if _global_trading_halt_enabled():
                print("GLOBAL_TRADING_HALT=1 detected; stopping all sleeves.")
                _stop_processes(procs)
                _emit_incident_snapshot("global_halt_detected", "runtime")
                _write_launcher_health(
                    _launcher_health_payload(
                        specs=specs,
                        procs=procs,
                        proc_started_at=proc_started_at,
                        restart_history=restart_history,
                        quarantined_jobs=quarantined_jobs,
                        launcher_started_at=launcher_started_at,
                        phase="stopped",
                        note="global_halt_detected",
                    )
                )
                return 0

            now = time.time()
            if (now - last_health_write_ts) >= max(int(args.launcher_health_seconds), 5):
                last_health_write_ts = now
                _write_launcher_health(
                    _launcher_health_payload(
                        specs=specs,
                        procs=procs,
                        proc_started_at=proc_started_at,
                        restart_history=restart_history,
                        quarantined_jobs=quarantined_jobs,
                        launcher_started_at=launcher_started_at,
                        phase="running",
                        policy_parked_jobs=policy_parked_jobs,
                        clean_exited_jobs=set(clean_exited_at),
                    )
                )

            if not args.disable_circuit_breakers and (now - last_breaker_check_ts) >= max(args.breaker_check_interval_seconds, 15):
                last_breaker_check_ts = now
                runtime_seconds = max(now - launcher_started_at, 0.0)
                if runtime_seconds < max(args.breaker_startup_grace_seconds, 0):
                    print(
                        "[CircuitBreaker] startup_grace "
                        f"remaining_s={int(max(args.breaker_startup_grace_seconds - runtime_seconds, 0))}"
                    )
                else:
                    metrics = _read_one_numbers(breaker_path)
                    reasons, _domain = _breaker_reasons(metrics, args, runtime_seconds=runtime_seconds)
                    if reasons:
                        breaker_streaks["core"] = breaker_streaks.get("core", 0) + 1
                        print(
                            f"[CircuitBreaker] breach_streak={breaker_streaks['core']}/{max(args.breaker_consecutive_breaches,1)} "
                            f"reasons={'|'.join(reasons)}"
                        )
                    else:
                        breaker_streaks["core"] = 0

                    if breaker_streaks["core"] >= max(args.breaker_consecutive_breaches, 1):
                        group_disabled_until["core"] = now + max(args.breaker_cooldown_seconds, 30)
                        breaker_streaks["core"] = 0
                        print(
                            f"[CircuitBreaker] TRIPPED group=core cooldown_s={max(args.breaker_cooldown_seconds,30)} "
                            f"reasons={'|'.join(reasons)}"
                        )
                        _emit_incident_snapshot("circuit_breaker_tripped", "|".join(reasons))
                        for name, proc in list(procs.items()):
                            if specs[name].breaker_group != "core":
                                continue
                            if proc.poll() is None:
                                proc.terminate()

            for name, proc in list(procs.items()):
                if name in quarantined_jobs:
                    continue
                code = proc.poll()
                if code is None:
                    if name in policy_parked_jobs:
                        if name not in parked_jobs_reported:
                            print(f"[{name}] parking reason=process_fanout_guard_active")
                            parked_jobs_reported.add(name)
                        proc.terminate()
                        _write_launcher_health(
                            _launcher_health_payload(
                                specs=specs,
                                procs=procs,
                                proc_started_at=proc_started_at,
                                restart_history=restart_history,
                                quarantined_jobs=quarantined_jobs,
                                launcher_started_at=launcher_started_at,
                                phase="running",
                                note=f"policy_parked_{name}",
                                policy_parked_jobs=policy_parked_jobs,
                                clean_exited_jobs=set(clean_exited_at),
                            )
                        )
                        continue
                    stale, reason = _job_heartbeat_stale(specs[name], started_at=proc_started_at.get(name, 0.0))
                    if stale:
                        print(f"[{name}] heartbeat_stale reason={reason}; recycling child")
                        _emit_incident_snapshot("execution_lane_heartbeat_stale", f"{name}:{reason}")
                        proc.terminate()
                        _write_launcher_health(
                            _launcher_health_payload(
                                specs=specs,
                                procs=procs,
                                proc_started_at=proc_started_at,
                                restart_history=restart_history,
                                quarantined_jobs=quarantined_jobs,
                                launcher_started_at=launcher_started_at,
                                phase="running",
                                note=f"repair_recycle_{name}:{reason}",
                                policy_parked_jobs=policy_parked_jobs,
                                clean_exited_jobs=set(clean_exited_at),
                            )
                        )
                    continue

                if name in clean_exited_at:
                    retry_after = clean_exited_at[name] + max(int(args.clean_exit_retry_seconds), 60)
                    if time.time() < retry_after:
                        continue
                    clean_exited_at.pop(name, None)
                    if not _within_restart_budget(restart_history[name], args.max_restarts_per_hour):
                        _emit_incident_snapshot("restart_budget_exceeded", f"{name}:{args.max_restarts_per_hour}")
                        quarantined_jobs[name] = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "reason": "restart_budget_exceeded_after_clean_exit",
                            "last_exit_code": int(code),
                            "max_restarts_per_hour": int(args.max_restarts_per_hour),
                        }
                        procs.pop(name, None)
                        print(
                            f"[{name}] quarantined reason=restart_budget_exceeded_after_clean_exit "
                            f"budget={args.max_restarts_per_hour}/hour parent=continuing"
                        )
                        _write_launcher_health(
                            _launcher_health_payload(
                                specs=specs,
                                procs=procs,
                                proc_started_at=proc_started_at,
                                restart_history=restart_history,
                                quarantined_jobs=quarantined_jobs,
                                launcher_started_at=launcher_started_at,
                                phase="running",
                                note=f"repair_quarantine_{name}",
                                policy_parked_jobs=policy_parked_jobs,
                                clean_exited_jobs=set(clean_exited_at),
                            )
                        )
                        continue
                    restart_history[name].append(time.time())
                    procs[name] = _spawn(specs[name])
                    proc_started_at[name] = time.time()
                    print(f"[{name}] clean_exit_retry restart_count_last_hour={len(restart_history[name])}")
                    _write_launcher_health(
                        _launcher_health_payload(
                            specs=specs,
                            procs=procs,
                            proc_started_at=proc_started_at,
                            restart_history=restart_history,
                            quarantined_jobs=quarantined_jobs,
                            launcher_started_at=launcher_started_at,
                            phase="running",
                            note=f"clean_exit_retry_{name}",
                            policy_parked_jobs=policy_parked_jobs,
                            clean_exited_jobs=set(clean_exited_at),
                        )
                    )
                    continue

                print(f"[{name}] exited code={code}")
                if name in policy_parked_jobs:
                    if name not in parked_jobs_reported:
                        print(f"[{name}] restart_parked reason=process_fanout_guard_active")
                        parked_jobs_reported.add(name)
                    _write_launcher_health(
                        _launcher_health_payload(
                            specs=specs,
                            procs=procs,
                            proc_started_at=proc_started_at,
                            restart_history=restart_history,
                            quarantined_jobs=quarantined_jobs,
                            launcher_started_at=launcher_started_at,
                            phase="running",
                            note=f"policy_parked_{name}",
                            policy_parked_jobs=policy_parked_jobs,
                            clean_exited_jobs=set(clean_exited_at),
                        )
                    )
                    continue
                if code == 0:
                    clean_exited_at[name] = time.time()
                    print(
                        f"[{name}] clean_exit_parked retry_seconds={max(int(args.clean_exit_retry_seconds), 60)}"
                    )
                    _write_launcher_health(
                        _launcher_health_payload(
                            specs=specs,
                            procs=procs,
                            proc_started_at=proc_started_at,
                            restart_history=restart_history,
                            quarantined_jobs=quarantined_jobs,
                            launcher_started_at=launcher_started_at,
                            phase="running",
                            note=f"clean_exit_parked_{name}",
                            policy_parked_jobs=policy_parked_jobs,
                            clean_exited_jobs=set(clean_exited_at),
                        )
                    )
                    continue
                if not args.restart_on_exit:
                    _stop_processes(procs)
                    _emit_incident_snapshot("sleeve_exit_no_restart", f"{name}:{code}")
                    print("Stopped because one sleeve exited and restart mode is disabled.")
                    return 1

                grp = specs[name].breaker_group
                cooldown_until = group_disabled_until.get(grp, 0.0)
                if cooldown_until > time.time():
                    remaining = int(max(cooldown_until - time.time(), 0))
                    print(f"[{name}] restart_paused circuit_breaker_cooldown_remaining_s={remaining}")
                    continue

                if not _within_restart_budget(restart_history[name], args.max_restarts_per_hour):
                    _emit_incident_snapshot("restart_budget_exceeded", f"{name}:{args.max_restarts_per_hour}")
                    quarantined_jobs[name] = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": "restart_budget_exceeded",
                        "last_exit_code": int(code),
                        "max_restarts_per_hour": int(args.max_restarts_per_hour),
                    }
                    procs.pop(name, None)
                    print(
                        f"[{name}] quarantined reason=restart_budget_exceeded "
                        f"budget={args.max_restarts_per_hour}/hour parent=continuing"
                    )
                    _write_launcher_health(
                        _launcher_health_payload(
                            specs=specs,
                            procs=procs,
                            proc_started_at=proc_started_at,
                            restart_history=restart_history,
                            quarantined_jobs=quarantined_jobs,
                            launcher_started_at=launcher_started_at,
                            phase="running",
                            note=f"repair_quarantine_{name}",
                            policy_parked_jobs=policy_parked_jobs,
                            clean_exited_jobs=set(clean_exited_at),
                        )
                    )
                    continue

                time.sleep(max(args.restart_delay_seconds, 1))
                restart_history[name].append(time.time())
                procs[name] = _spawn(specs[name])
                proc_started_at[name] = time.time()
                print(f"[{name}] restart_count_last_hour={len(restart_history[name])}")
                _write_launcher_health(
                    _launcher_health_payload(
                        specs=specs,
                        procs=procs,
                        proc_started_at=proc_started_at,
                        restart_history=restart_history,
                        quarantined_jobs=quarantined_jobs,
                        launcher_started_at=launcher_started_at,
                        phase="running",
                        note=f"repair_restarted_{name}",
                        policy_parked_jobs=policy_parked_jobs,
                        clean_exited_jobs=set(clean_exited_at),
                    )
                )

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping all sleeves...")
        _stop_processes(procs)
        _write_launcher_health(
            _launcher_health_payload(
                specs=specs,
                procs=procs,
                proc_started_at=proc_started_at,
                restart_history=restart_history,
                quarantined_jobs=quarantined_jobs,
                launcher_started_at=launcher_started_at,
                phase="stopped",
                note="keyboard_interrupt",
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
