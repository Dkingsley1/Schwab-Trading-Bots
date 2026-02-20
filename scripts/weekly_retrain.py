import argparse
import fcntl
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "master_bot_registry.json")
VENV_PY = os.path.join(PROJECT_ROOT, ".venv312", "bin", "python")
MASTER_RUNNER = os.path.join(PROJECT_ROOT, "scripts", "run_master_bot.py")
TRADE_DATASET_BUILDER = os.path.join(PROJECT_ROOT, "scripts", "build_trade_learning_dataset.py")
TRADE_BEHAVIOR_TRAINER = os.path.join(PROJECT_ROOT, "scripts", "train_trade_behavior_bot.py")
PRUNE_UNDERPERFORMERS = os.path.join(PROJECT_ROOT, "scripts", "prune_underperformers.py")
PRUNE_REDUNDANT = os.path.join(PROJECT_ROOT, "scripts", "prune_redundant_bots.py")


_MLX_LOCK_HANDLE = None


def _acquire_mlx_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
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
        print(f"[MLXLock] busy lock_path={lock_path} owner={owner or 'unknown'}")
        return None

    fh.seek(0)
    fh.truncate(0)
    fh.write(json.dumps({
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cmd": sys.argv,
    }, ensure_ascii=True))
    fh.flush()
    print(f"[MLXLock] acquired lock_path={lock_path} pid={os.getpid()}")
    return fh


def _normalized_bot_id_from_script(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".py"):
        name = name[:-3]
    return name.lower()


def _market_open_now_et(start_hour: int, end_hour: int) -> bool:
    if ZoneInfo is None:
        return False
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    wd = now_et.weekday()
    if wd >= 5:
        return False
    h = now_et.hour
    return start_hour <= h < end_hour


def _monthly_stamp_path() -> str:
    return os.path.join(PROJECT_ROOT, "governance", "monthly_prune_stamp.json")


def _monthly_prune_due() -> bool:
    stamp = _monthly_stamp_path()
    now = datetime.now(timezone.utc)
    if not os.path.exists(stamp):
        return True
    try:
        with open(stamp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        y = int(obj.get("year", 0))
        m = int(obj.get("month", 0))
        return (y, m) != (now.year, now.month)
    except Exception:
        return True


def _write_monthly_prune_stamp() -> None:
    now = datetime.now(timezone.utc)
    path = _monthly_stamp_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"year": now.year, "month": now.month, "timestamp_utc": now.isoformat()}, f, ensure_ascii=True)


def _load_deleted_bot_ids(registry_path: str) -> set[str]:
    if not os.path.exists(registry_path):
        return set()
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return set()

    out: set[str] = set()
    for row in reg.get("sub_bots", []):
        if bool(row.get("deleted_from_rotation", False)):
            bot_id = str(row.get("bot_id", "")).strip().lower()
            if bot_id:
                out.add(bot_id)
    return out


def build_targets(include_deleted: bool = False) -> list[str]:
    deleted_ids = set()
    if not include_deleted:
        deleted_ids = _load_deleted_bot_ids(REGISTRY_PATH)

    targets: list[str] = []

    v3 = os.path.join(CORE_DIR, "brain_refinery_V3.py")
    if os.path.exists(v3):
        if include_deleted or _normalized_bot_id_from_script(v3) not in deleted_ids:
            targets.append(v3)

    versioned = sorted(glob.glob(os.path.join(CORE_DIR, "brain_refinery_v*.py")))
    for script in versioned:
        if include_deleted or _normalized_bot_id_from_script(script) not in deleted_ids:
            targets.append(script)

    return targets


def _parse_size_to_gb(value: str) -> float:
    try:
        s = value.strip().upper()
        if s.endswith("G"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1024.0
        if s.endswith("K"):
            return float(s[:-1]) / (1024.0 * 1024.0)
        return float(s)
    except Exception:
        return 0.0


def _memory_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}

    try:
        proc = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
        out = proc.stdout or ""
        for raw in out.splitlines():
            line = raw.strip()
            lower = line.lower()
            if "free percentage" in lower:
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["free_pct"] = float(rhs)
            elif "available percentage" in lower:
                rhs = line.split(":", 1)[-1].strip().replace("%", "")
                snapshot["available_pct"] = float(rhs)
    except Exception:
        pass

    try:
        proc = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "").strip()
        if "used =" in out:
            used_part = out.split("used =", 1)[1].strip().split()[0]
            snapshot["swap_used_gb"] = _parse_size_to_gb(used_part)
    except Exception:
        pass

    return snapshot


def _memory_ready(min_free_pct: float, max_swap_gb: float) -> tuple[bool, str, dict[str, float]]:
    snap = _memory_snapshot()

    free_pct = snap.get("free_pct")
    if free_pct is not None and free_pct < min_free_pct:
        return False, f"free_pct={free_pct:.1f} < min_free_pct={min_free_pct:.1f}", snap

    swap_gb = snap.get("swap_used_gb")
    if swap_gb is not None and swap_gb > max_swap_gb:
        return False, f"swap_used_gb={swap_gb:.2f} > max_swap_gb={max_swap_gb:.2f}", snap

    return True, "ok", snap


def _thermal_snapshot() -> dict[str, float]:
    snap: dict[str, float] = {}
    try:
        proc = subprocess.run(["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for raw in out.splitlines():
            line = raw.strip()
            if "CPU_Speed_Limit" in line and "=" in line:
                snap["cpu_speed_limit"] = float(line.split("=", 1)[1].strip())
            if "Scheduler_Limit" in line and "=" in line:
                snap["scheduler_limit"] = float(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return snap


def _thermal_ready(min_cpu_speed_limit: float, min_scheduler_limit: float) -> tuple[bool, str, dict[str, float]]:
    snap = _thermal_snapshot()
    csl = snap.get("cpu_speed_limit")
    if csl is not None and csl < min_cpu_speed_limit:
        return False, f"cpu_speed_limit={csl:.0f} < min_cpu_speed_limit={min_cpu_speed_limit:.0f}", snap
    sl = snap.get("scheduler_limit")
    if sl is not None and sl < min_scheduler_limit:
        return False, f"scheduler_limit={sl:.0f} < min_scheduler_limit={min_scheduler_limit:.0f}", snap
    return True, "ok", snap


def _wait_for_thermal_gate(
    *,
    enabled: bool,
    min_cpu_speed_limit: float,
    min_scheduler_limit: float,
    poll_seconds: int,
    max_wait_seconds: int,
    label: str,
    dry_run: bool,
) -> bool:
    if dry_run or not enabled:
        return True

    start = time.time()
    while True:
        ok, reason, snap = _thermal_ready(
            min_cpu_speed_limit=min_cpu_speed_limit,
            min_scheduler_limit=min_scheduler_limit,
        )
        if ok:
            return True

        waited = int(time.time() - start)
        if max_wait_seconds > 0 and waited >= max_wait_seconds:
            print(f"[ThermalGate] skip label={label} waited={waited}s reason={reason}")
            return False

        print(f"[ThermalGate] wait label={label} waited={waited}s reason={reason} snapshot={snap}")
        time.sleep(max(poll_seconds, 1))


def _wait_for_memory_gate(
    *,
    enabled: bool,
    min_free_pct: float,
    max_swap_gb: float,
    poll_seconds: int,
    max_wait_seconds: int,
    label: str,
    dry_run: bool,
) -> bool:
    if dry_run or not enabled:
        return True

    start = time.time()
    while True:
        ok, reason, snap = _memory_ready(min_free_pct=min_free_pct, max_swap_gb=max_swap_gb)
        if ok:
            return True

        waited = int(time.time() - start)
        if max_wait_seconds > 0 and waited >= max_wait_seconds:
            print(f"[MemoryGate] skip label={label} waited={waited}s reason={reason}")
            return False

        print(
            f"[MemoryGate] wait label={label} waited={waited}s reason={reason} "
            f"snapshot={snap}"
        )
        time.sleep(max(poll_seconds, 1))


def _build_child_env(thread_cap: int) -> dict[str, str]:
    env = os.environ.copy()
    cap = str(max(int(thread_cap), 1))

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(key, cap)

    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _apply_nice(nice_value: int) -> None:
    if nice_value <= 0:
        return
    try:
        os.nice(nice_value)
        print(f"Applied process nice={nice_value} for retrain wrapper and child jobs")
    except Exception as exc:
        print(f"WARN: could not apply nice={nice_value}: {exc}")


def run_cmd(cmd: list[str], dry_run: bool, env: dict[str, str], extra_nice: int = 0) -> int:
    full_cmd = cmd
    if extra_nice > 0:
        full_cmd = ["/usr/bin/nice", "-n", str(extra_nice)] + cmd
    print("$ " + " ".join(full_cmd))
    if dry_run:
        return 0
    proc = subprocess.run(full_cmd, cwd=PROJECT_ROOT, env=env)
    return proc.returncode



def _registry_snapshot(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    s = obj.get("summary", {}) if isinstance(obj, dict) else {}
    top = s.get("top_active", []) if isinstance(s, dict) else []
    top_quality = 0.0
    if top and isinstance(top, list):
        first = top[0] if isinstance(top[0], dict) else {}
        top_quality = float(first.get("quality_score", first.get("test_accuracy", 0.0)) or 0.0)

    return {
        "active_bots": float(s.get("active_bots", 0) or 0),
        "deleted_from_rotation": float(s.get("deleted_from_rotation", 0) or 0),
        "top_quality": top_quality,
    }


def _should_rollback_registry(prev: dict[str, float], curr: dict[str, float]) -> tuple[bool, str]:
    if not prev or not curr:
        return False, "snapshot_missing"

    prev_active = prev.get("active_bots", 0.0)
    curr_active = curr.get("active_bots", 0.0)
    prev_deleted = prev.get("deleted_from_rotation", 0.0)
    curr_deleted = curr.get("deleted_from_rotation", 0.0)
    curr_top_quality = curr.get("top_quality", 0.0)

    min_active = float(os.getenv("ROLLBACK_MIN_ACTIVE_BOTS", "12"))
    max_active_drop_pct = float(os.getenv("ROLLBACK_MAX_ACTIVE_DROP_PCT", "0.55"))
    max_deleted_jump = float(os.getenv("ROLLBACK_MAX_DELETED_JUMP", "20"))
    min_top_quality = float(os.getenv("ROLLBACK_MIN_TOP_QUALITY", "0.28"))

    if curr_active < min_active:
        return True, f"active_bots_below_floor curr={curr_active:.0f} min={min_active:.0f}"

    if prev_active > 0:
        drop_pct = (prev_active - curr_active) / prev_active
        if drop_pct > max_active_drop_pct:
            return True, f"active_drop_pct={drop_pct:.2f} > max_active_drop_pct={max_active_drop_pct:.2f}"

    if (curr_deleted - prev_deleted) > max_deleted_jump:
        return True, f"deleted_jump={curr_deleted - prev_deleted:.0f} > max_deleted_jump={max_deleted_jump:.0f}"

    if curr_top_quality < min_top_quality:
        return True, f"top_quality={curr_top_quality:.3f} < min_top_quality={min_top_quality:.3f}"

    return False, "healthy"




def _load_active_bot_map(registry_path: str) -> dict[str, bool]:
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}

    out: dict[str, bool] = {}
    for row in reg.get("sub_bots", []):
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        out[bot_id] = bool(row.get("active", False))
    return out


def _latest_model_age_hours(bot_id: str) -> float | None:
    model_glob = os.path.join(PROJECT_ROOT, "models", f"{bot_id}_*.npz")
    paths = sorted(glob.glob(model_glob))
    if not paths:
        return None
    latest = paths[-1]
    try:
        age_sec = max(time.time() - os.path.getmtime(latest), 0.0)
        return age_sec / 3600.0
    except Exception:
        return None


def _filter_targets_for_efficiency(
    targets: list[str],
    *,
    active_only: bool,
    max_targets: int,
    min_model_age_hours: float,
) -> tuple[list[str], dict[str, int]]:
    active_map = _load_active_bot_map(REGISTRY_PATH)

    rows: list[tuple[str, str, bool, float]] = []
    for t in targets:
        bot_id = _normalized_bot_id_from_script(t)
        is_active = bool(active_map.get(bot_id, False))
        age_h = _latest_model_age_hours(bot_id)
        if age_h is None:
            age_h = 1e9  # prioritize bots without prior model artifact
        rows.append((t, bot_id, is_active, age_h))

    pre = len(rows)
    if active_only:
        rows = [r for r in rows if r[2]]

    if min_model_age_hours > 0:
        rows = [r for r in rows if r[3] >= min_model_age_hours]

    # Prioritize active first, then stalest models first.
    rows.sort(key=lambda r: (0 if r[2] else 1, -float(r[3]), r[1]))

    if max_targets > 0:
        rows = rows[:max_targets]

    filtered = [r[0] for r in rows]
    stats = {
        "pre": pre,
        "post": len(filtered),
        "active_selected": sum(1 for r in rows if r[2]),
    }
    return filtered, stats



def _load_accuracy_map(registry_path: str) -> dict[str, float]:
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for row in reg.get("sub_bots", []):
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        try:
            out[bot_id] = float(row.get("test_accuracy", 0.0) or 0.0)
        except Exception:
            out[bot_id] = 0.0
    return out


def _apply_retrain_curriculum(targets: list[str], registry_path: str) -> list[str]:
    if os.getenv("RETRAIN_CURRICULUM_ENABLED", "1").strip() != "1":
        return targets

    acc = _load_accuracy_map(registry_path)

    def rank(path: str) -> tuple[int, float, str]:
        bot_id = _normalized_bot_id_from_script(path)
        a = acc.get(bot_id, 0.0)
        # Train stronger anchors first, then weak/missing models.
        band = 0
        if a >= 0.58:
            band = 0
        elif a >= 0.50:
            band = 1
        else:
            band = 2
        return (band, -a, bot_id)

    return sorted(targets, key=rank)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all brain_refinery training scripts and refresh master bot registry.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining scripts when one fails.")
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Also retrain bots marked deleted_from_rotation in registry.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        default=os.getenv("RETRAIN_ACTIVE_ONLY", "1").strip() == "1",
        help="Retrain only currently active bots for faster runs (default on).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=int(os.getenv("RETRAIN_MAX_TARGETS", "30")),
        help="Maximum number of bot scripts to retrain per run (0 = no cap).",
    )
    parser.add_argument(
        "--min-model-age-hours",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_MODEL_AGE_HOURS", "0")),
        help="Skip bots retrained more recently than this many hours.",
    )
    parser.add_argument(
        "--thread-cap",
        type=int,
        default=int(os.getenv("RETRAIN_THREAD_CAP", "2")),
        help="Cap BLAS/OpenMP threads per training process (default: RETRAIN_THREAD_CAP or 2).",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=int(os.getenv("RETRAIN_NICE", "10")),
        help="Niceness to apply to retrain process (default: RETRAIN_NICE or 10).",
    )
    parser.add_argument(
        "--memory-guard",
        action="store_true",
        default=os.getenv("RETRAIN_MEMORY_GUARD", "1").strip() == "1",
        help="Enable memory gate before each target run.",
    )
    parser.add_argument(
        "--min-free-pct",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_FREE_PCT", "22")),
        help="Minimum free memory percentage required before launching next model.",
    )
    parser.add_argument(
        "--max-swap-gb",
        type=float,
        default=float(os.getenv("RETRAIN_MAX_SWAP_GB", "1.0")),
        help="Maximum allowed swap usage (GB) before launching next model.",
    )
    parser.add_argument(
        "--memory-poll-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_MEMORY_POLL_SECONDS", "20")),
        help="How often to re-check memory gate while waiting.",
    )
    parser.add_argument(
        "--memory-max-wait-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_MEMORY_MAX_WAIT_SECONDS", "1800")),
        help="Max wait per target before skipping due to memory pressure.",
    )
    parser.add_argument(
        "--between-target-sleep-seconds",
        type=int,
        default=int(os.getenv("RETRAIN_BETWEEN_TARGET_SLEEP_SECONDS", "4")),
        help="Cooldown sleep between targets to smooth memory pressure.",
    )
    parser.add_argument(
        "--thermal-guard",
        action="store_true",
        default=os.getenv("RETRAIN_THERMAL_GUARD", "1").strip() == "1",
        help="Enable thermal gate checks before each target run.",
    )
    parser.add_argument(
        "--thermal-min-cpu-speed-limit",
        type=float,
        default=float(os.getenv("RETRAIN_THERMAL_MIN_CPU_SPEED_LIMIT", "75")),
        help="Minimum pmset CPU_Speed_Limit required to launch next model.",
    )
    parser.add_argument(
        "--thermal-min-scheduler-limit",
        type=float,
        default=float(os.getenv("RETRAIN_THERMAL_MIN_SCHEDULER_LIMIT", "75")),
        help="Minimum pmset Scheduler_Limit required to launch next model.",
    )
    parser.add_argument(
        "--ops-extra-nice",
        type=int,
        default=int(os.getenv("RETRAIN_OPS_EXTRA_NICE", "6")),
        help="Extra nice offset for ops tasks (master registry update + behavior jobs).",
    )
    parser.add_argument(
        "--after-hours-only",
        action="store_true",
        default=os.getenv("RETRAIN_AFTER_HOURS_ONLY", "1").strip() == "1",
        help="Skip retrain during market hours (ET) unless explicitly disabled.",
    )
    parser.add_argument(
        "--session-start-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_START_HOUR", "8")),
    )
    parser.add_argument(
        "--session-end-hour",
        type=int,
        default=int(os.getenv("MARKET_SESSION_END_HOUR", "20")),
    )
    parser.add_argument(
        "--monthly-prune",
        action="store_true",
        default=os.getenv("MONTHLY_PRUNE_ENABLED", "1").strip() == "1",
        help="Run monthly underperformer/redundancy prune once per month.",
    )
    args = parser.parse_args()

    lock_path = os.getenv("MLX_RETRAIN_LOCK_PATH", os.path.join(PROJECT_ROOT, "governance", "mlx_retrain.lock"))
    global _MLX_LOCK_HANDLE
    _MLX_LOCK_HANDLE = _acquire_mlx_lock(lock_path)
    if _MLX_LOCK_HANDLE is None:
        print("Another MLX retrain is already active. Skipping this retrain run.")
        return 0

    if args.after_hours_only and _market_open_now_et(args.session_start_hour, args.session_end_hour):
        print("Retrain skipped: market session is open (after-hours-only enabled).")
        return 0

    if not os.path.exists(VENV_PY):
        print(f"ERROR: venv python not found at {VENV_PY}")
        return 2

    deleted_ids = _load_deleted_bot_ids(REGISTRY_PATH)
    targets = build_targets(include_deleted=args.include_deleted)
    if not targets:
        print("ERROR: no brain_refinery targets found")
        return 2

    base_targets = list(targets)
    min_age = max(float(args.min_model_age_hours), 0.0)
    targets, target_stats = _filter_targets_for_efficiency(
        targets,
        active_only=args.active_only,
        max_targets=max(int(args.max_targets), 0),
        min_model_age_hours=min_age,
    )
    if not targets and min_age > 0:
        print("WARN: age filter selected zero targets; retrying with min_model_age_hours=0")
        targets, target_stats = _filter_targets_for_efficiency(
            base_targets,
            active_only=args.active_only,
            max_targets=max(int(args.max_targets), 0),
            min_model_age_hours=0.0,
        )
    if not targets:
        print("WARN: efficiency filter selected zero targets; falling back to full target set")
        targets = base_targets
        target_stats = {"pre": len(base_targets), "post": len(base_targets), "active_selected": 0}

    targets = _apply_retrain_curriculum(targets, REGISTRY_PATH)

    child_env = _build_child_env(args.thread_cap)
    _apply_nice(args.nice)

    print(
        "Resource limits: "
        f"thread_cap={args.thread_cap} "
        f"OMP={child_env.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={child_env.get('OPENBLAS_NUM_THREADS')} "
        f"VECLIB={child_env.get('VECLIB_MAXIMUM_THREADS')}"
    )
    print(
        "Memory gate: "
        f"enabled={args.memory_guard} "
        f"min_free_pct={args.min_free_pct:.1f} "
        f"max_swap_gb={args.max_swap_gb:.2f} "
        f"poll={args.memory_poll_seconds}s "
        f"max_wait={args.memory_max_wait_seconds}s "
        f"cooldown={args.between_target_sleep_seconds}s"
    )
    print(
        "Thermal gate: "
        f"enabled={args.thermal_guard} "
        f"min_cpu_speed_limit={args.thermal_min_cpu_speed_limit:.0f} "
        f"min_scheduler_limit={args.thermal_min_scheduler_limit:.0f}"
    )

    if not args.include_deleted and deleted_ids:
        print(f"Skipping deleted bots from rotation: {len(deleted_ids)}")

    print(
        "Efficiency filter: "
        f"active_only={args.active_only} "
        f"max_targets={args.max_targets} "
        f"min_model_age_hours={args.min_model_age_hours:.1f} "
        f"selected={target_stats.get('post', 0)}/{target_stats.get('pre', 0)}"
    )

    started = datetime.now(timezone.utc).isoformat()
    print(f"Weekly retrain start (UTC): {started}")
    print(f"Targets: {len(targets)}")
    print("Efficiency tip: keep streaming/video/browser load low during retrain windows.")

    failures: list[str] = []
    skipped_by_memory: list[str] = []
    for target in targets:
        target_name = os.path.basename(target)
        allowed = _wait_for_memory_gate(
            enabled=args.memory_guard,
            min_free_pct=args.min_free_pct,
            max_swap_gb=args.max_swap_gb,
            poll_seconds=args.memory_poll_seconds,
            max_wait_seconds=args.memory_max_wait_seconds,
            label=target_name,
            dry_run=args.dry_run,
        )
        if not allowed:
            skipped_by_memory.append(target)
            continue

        thermal_ok = _wait_for_thermal_gate(
            enabled=args.thermal_guard,
            min_cpu_speed_limit=args.thermal_min_cpu_speed_limit,
            min_scheduler_limit=args.thermal_min_scheduler_limit,
            poll_seconds=args.memory_poll_seconds,
            max_wait_seconds=args.memory_max_wait_seconds,
            label=target_name,
            dry_run=args.dry_run,
        )
        if not thermal_ok:
            skipped_by_memory.append(target)
            continue

        rc = run_cmd([VENV_PY, target], args.dry_run, child_env)
        if rc != 0:
            failures.append(target)
            print(f"FAIL: {target} (exit={rc})")
            if not args.continue_on_error:
                break

        if not args.dry_run:
            gc.collect()
            time.sleep(max(args.between_target_sleep_seconds, 0))

    if failures and not args.continue_on_error:
        print("Stopped early due to failure.")
        return 1

    prev_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
    registry_backup_path = os.path.join(PROJECT_ROOT, "governance", "registry_backup_before_retrain.json")
    try:
        if os.path.exists(REGISTRY_PATH):
            os.makedirs(os.path.dirname(registry_backup_path), exist_ok=True)
            shutil.copy2(REGISTRY_PATH, registry_backup_path)
    except Exception as exc:
        print(f"WARN: could not backup registry before update: {exc}")

    if _wait_for_memory_gate(
        enabled=args.memory_guard,
        min_free_pct=args.min_free_pct,
        max_swap_gb=args.max_swap_gb,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label="run_master_bot.py",
        dry_run=args.dry_run,
    ) and _wait_for_thermal_gate(
        enabled=args.thermal_guard,
        min_cpu_speed_limit=args.thermal_min_cpu_speed_limit,
        min_scheduler_limit=args.thermal_min_scheduler_limit,
        poll_seconds=args.memory_poll_seconds,
        max_wait_seconds=args.memory_max_wait_seconds,
        label="run_master_bot.py",
        dry_run=args.dry_run,
    ):
        rc = run_cmd([sys.executable, MASTER_RUNNER], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
        if rc != 0:
            print(f"FAIL: master bot update (exit={rc})")
            return 1

        if not args.dry_run:
            curr_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
            rollback_bad, rollback_reason = _should_rollback_registry(prev_registry_snapshot, curr_registry_snapshot)
            if rollback_bad and os.path.exists(registry_backup_path):
                shutil.copy2(registry_backup_path, REGISTRY_PATH)
                print(f"[Rollback] restored previous master registry reason={rollback_reason}")
            else:
                print(f"[Rollback] registry check status={rollback_reason}")
    else:
        print("WARN: skipped master registry update due to memory gate timeout")

    ended = datetime.now(timezone.utc).isoformat()
    print(f"Weekly retrain end (UTC): {ended}")

    if skipped_by_memory:
        print(f"Skipped by memory gate: {len(skipped_by_memory)}")
        for s in skipped_by_memory:
            print(f" - {s}")

    if failures:
        print(f"Completed with {len(failures)} failures.")
        for f in failures:
            print(f" - {f}")
        return 1

    enable_trade_behavior_retrain = os.getenv("ENABLE_TRADE_BEHAVIOR_RETRAIN", "1").strip() == "1"
    trade_behavior_strict = os.getenv("TRADE_BEHAVIOR_STRICT", "0").strip() == "1"

    if enable_trade_behavior_retrain:
        print("Running trade history behavior learning step...")
        if os.path.exists(TRADE_DATASET_BUILDER):
            rc = run_cmd([VENV_PY, TRADE_DATASET_BUILDER], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if rc != 0 and trade_behavior_strict:
                print("FAIL: trade dataset build")
                return 1
        else:
            print(f"WARN: trade dataset builder missing: {TRADE_DATASET_BUILDER}")

        if os.path.exists(TRADE_BEHAVIOR_TRAINER):
            rc = run_cmd([VENV_PY, TRADE_BEHAVIOR_TRAINER], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if rc != 0 and trade_behavior_strict:
                print("FAIL: trade behavior trainer")
                return 1
        else:
            print(f"WARN: trade behavior trainer missing: {TRADE_BEHAVIOR_TRAINER}")

    if args.monthly_prune and (not args.dry_run) and _monthly_prune_due():
        print("Running monthly prune pass...")
        if os.path.exists(PRUNE_UNDERPERFORMERS):
            _ = run_cmd([VENV_PY, PRUNE_UNDERPERFORMERS, "--min-streak", os.getenv("MONTHLY_PRUNE_MIN_STREAK", "3")], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
        if os.path.exists(PRUNE_REDUNDANT):
            _ = run_cmd([VENV_PY, PRUNE_REDUNDANT], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
        _write_monthly_prune_stamp()

    if skipped_by_memory:
        print("Completed with memory-gate skips.")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
