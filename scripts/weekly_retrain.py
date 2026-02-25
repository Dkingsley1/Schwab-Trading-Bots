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
ARCHIVE_OLD_MODELS = os.path.join(PROJECT_ROOT, "scripts", "archive_old_models.py")
CANARY_DIAGNOSTICS = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "canary_diagnostics_latest.json")
RETIRE_PERSISTENT_LOSERS = os.path.join(PROJECT_ROOT, "scripts", "retire_persistent_losers.py")
PROMOTION_READINESS_PATH = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "promotion_readiness_latest.json")
PROMOTION_BOTTLENECK_PATH = os.path.join(PROJECT_ROOT, "governance", "walk_forward", "promotion_bottleneck_latest.json")
WALK_FORWARD_VALIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "walk_forward_validate.py")
WALK_FORWARD_PROMOTION_GATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "walk_forward_promotion_gate.py")
PROMOTION_READINESS_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "promotion_readiness_summary.py")
PROMOTION_BOTTLENECK_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "promotion_bottleneck_focus.py")
NEW_BOT_GRADUATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "new_bot_graduation_gate.py")
LEAK_OVERFIT_GUARD_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "leak_overfit_guard.py")
MODEL_LIFECYCLE_HYGIENE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "model_lifecycle_hygiene.py")
WEEKLY_GATE_BLOCKER_REPORT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "weekly_gate_blocker_report.py")


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


SEGMENT_KEYWORDS = {
    "trend": ["trend", "breakout", "donchian"],
    "mean_revert": ["mean_revert", "vwap", "bollinger", "keltner"],
    "shock": ["flash", "shock", "event", "crash", "anomaly"],
    "liquidity": ["liquidity", "spread", "order_flow", "microstructure"],
}


def _segment_bot_id(bot_id: str) -> str:
    b = (bot_id or "").lower()
    for seg, keys in SEGMENT_KEYWORDS.items():
        if any(k in b for k in keys):
            return seg
    return "other"


def _apply_regime_focus(targets: list[str], regime_focus: str) -> list[str]:
    focus = {x.strip().lower() for x in str(regime_focus or "").split(",") if x.strip()}
    if not focus:
        return targets
    return [t for t in targets if _segment_bot_id(_normalized_bot_id_from_script(t)) in focus]


def _apply_regime_balanced_order(targets: list[str]) -> list[str]:
    if not targets:
        return targets
    buckets: dict[str, list[str]] = {}
    for t in targets:
        seg = _segment_bot_id(_normalized_bot_id_from_script(t))
        buckets.setdefault(seg, []).append(t)

    for k in buckets:
        buckets[k] = sorted(buckets[k], key=lambda x: _normalized_bot_id_from_script(x))

    ordered: list[str] = []
    seg_order = ["trend", "mean_revert", "shock", "liquidity", "other"]
    while True:
        moved = False
        for seg in seg_order:
            rows = buckets.get(seg, [])
            if rows:
                ordered.append(rows.pop(0))
                moved = True
        if not moved:
            break
    return ordered


def _load_json_file(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _check_data_quality_floor(
    *,
    coverage_file: str,
    divergence_file: str,
    min_coverage_ratio: float,
    max_divergence_spread: float,
) -> tuple[bool, str, dict]:
    coverage = _load_json_file(coverage_file)
    divergence = _load_json_file(divergence_file)

    coverage_ratio = float(coverage.get("coverage_ratio", 0.0) or 0.0)
    worst_spread = float(divergence.get("worst_relative_spread", 0.0) or 0.0)

    if coverage and (coverage_ratio < float(min_coverage_ratio)):
        return False, f"snapshot_coverage_ratio={coverage_ratio:.4f} < min_coverage_ratio={float(min_coverage_ratio):.4f}", {
            "coverage_ratio": coverage_ratio,
            "min_coverage_ratio": float(min_coverage_ratio),
            "worst_relative_spread": worst_spread,
            "max_divergence_spread": float(max_divergence_spread),
        }

    if divergence and (worst_spread > float(max_divergence_spread)):
        return False, f"worst_relative_spread={worst_spread:.4f} > max_divergence_spread={float(max_divergence_spread):.4f}", {
            "coverage_ratio": coverage_ratio,
            "min_coverage_ratio": float(min_coverage_ratio),
            "worst_relative_spread": worst_spread,
            "max_divergence_spread": float(max_divergence_spread),
        }

    return True, "ok", {
        "coverage_ratio": coverage_ratio,
        "min_coverage_ratio": float(min_coverage_ratio),
        "worst_relative_spread": worst_spread,
        "max_divergence_spread": float(max_divergence_spread),
    }


def _apply_canary_priority(targets: list[str], diagnostics_file: str, top_n: int) -> tuple[list[str], int]:
    if not targets or top_n <= 0:
        return targets, 0
    diag = _load_json_file(diagnostics_file)
    rows = diag.get("top_failing_bots") if isinstance(diag.get("top_failing_bots"), list) else []
    ids = []
    for row in rows[:max(int(top_n), 0)]:
        bot_id = str((row or {}).get("bot_id", "")).strip().lower()
        if bot_id:
            ids.append(bot_id)
    if not ids:
        return targets, 0

    wanted = set(ids)
    front = [t for t in targets if _normalized_bot_id_from_script(t) in wanted]
    rest = [t for t in targets if _normalized_bot_id_from_script(t) not in wanted]
    return front + rest, len(front)


def _registry_accuracy_map(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for row in obj.get("sub_bots", []) if isinstance(obj, dict) else []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        try:
            out[bot_id] = float(row.get("test_accuracy", 0.0) or 0.0)
        except Exception:
            out[bot_id] = 0.0
    return out


def _write_retrain_scorecard(
    *,
    started_utc: str,
    ended_utc: str,
    target_count: int,
    failures: list[str],
    skipped_by_memory: list[str],
    target_outcomes: list[dict],
    prev_registry_snapshot: dict[str, float],
    curr_registry_snapshot: dict[str, float],
    prev_acc: dict[str, float],
    curr_acc: dict[str, float],
    master_update_status: str,
    data_quality_summary: dict,
    canary_priority_selected: int,
    distill_selected: int,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "exports", "sql_reports")
    os.makedirs(out_dir, exist_ok=True)

    improved = 0
    degraded = 0
    unchanged = 0
    for bot_id, old_acc in prev_acc.items():
        if bot_id not in curr_acc:
            continue
        new_acc = curr_acc.get(bot_id, old_acc)
        if new_acc > old_acc + 1e-9:
            improved += 1
        elif new_acc < old_acc - 1e-9:
            degraded += 1
        else:
            unchanged += 1

    status_counts: dict[str, int] = {}
    for row in target_outcomes:
        s = str((row or {}).get("status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "target_count": int(target_count),
        "status_counts": status_counts,
        "failure_count": len(failures),
        "skipped_by_memory_count": len(skipped_by_memory),
        "master_update_status": master_update_status,
        "canary_priority_selected": int(canary_priority_selected),
        "distillation_priority_selected": int(distill_selected),
        "data_quality": data_quality_summary,
        "registry_before": prev_registry_snapshot,
        "registry_after": curr_registry_snapshot,
        "accuracy_delta": {
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
        },
        "failures": failures,
        "skipped_by_memory": skipped_by_memory,
    }

    json_path = os.path.join(out_dir, f"retrain_scorecard_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    latest_json = os.path.join(PROJECT_ROOT, "governance", "health", "retrain_scorecard_latest.json")
    os.makedirs(os.path.dirname(latest_json), exist_ok=True)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    md_path = os.path.join(out_dir, f"retrain_scorecard_{ts}.md")
    lines = [
        f"# Retrain Scorecard ({payload['timestamp_utc']})",
        f"- Window: {started_utc} -> {ended_utc}",
        f"- Targets: {target_count}",
        f"- Master update: {master_update_status}",
        f"- Failures: {len(failures)}",
        f"- Skipped by memory/thermal: {len(skipped_by_memory)}",
        f"- Accuracy delta: improved={improved} degraded={degraded} unchanged={unchanged}",
        f"- Canary-priority selected: {int(canary_priority_selected)}",
        f"- Distillation-priority selected: {int(distill_selected)}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path


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


def _weekly_archive_stamp_path() -> str:
    return os.path.join(PROJECT_ROOT, "governance", "weekly_model_archive_stamp.json")


def _weekly_archive_due() -> bool:
    now = datetime.now(timezone.utc)
    yw = now.isocalendar()
    year = int(yw[0])
    week = int(yw[1])

    path = _weekly_archive_stamp_path()
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        y = int(obj.get("year", 0))
        w = int(obj.get("week", 0))
        return (y, w) != (year, week)
    except Exception:
        return True


def _write_weekly_archive_stamp() -> None:
    now = datetime.now(timezone.utc)
    yw = now.isocalendar()
    path = _weekly_archive_stamp_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"year": int(yw[0]), "week": int(yw[1]), "timestamp_utc": now.isoformat()}, f, ensure_ascii=True)


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



def _load_walk_forward_runs(path: str) -> dict[str, int]:
    obj = _load_json_file(path)
    bots = obj.get("bots") if isinstance(obj.get("bots"), dict) else {}
    out: dict[str, int] = {}
    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        key = str(bot_id).strip().lower()
        if not key:
            continue
        out[key] = int(row.get("runs", 0) or 0)
    return out


def _select_new_bot_targets(targets: list[str], runs_map: dict[str, int], max_runs: int) -> list[str]:
    out: list[str] = []
    for t in targets:
        bid = _normalized_bot_id_from_script(t)
        runs = int(runs_map.get(bid, 0) or 0)
        if runs <= max(int(max_runs), 0):
            out.append(t)
    return out


def _derive_regime_focus_from_readiness(path: str, top_n: int = 2) -> str:
    obj = _load_json_file(path)
    rows = obj.get("failed_by_segment") if isinstance(obj.get("failed_by_segment"), dict) else {}
    if not rows:
        return ""
    ranked = sorted(rows.items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))
    picks = [k for k, _ in ranked if str(k).strip().lower() in {"trend", "mean_revert", "shock", "liquidity", "other"}]
    return ",".join(picks[: max(int(top_n), 1)])


def _effective_int(base: int, floor_value: int) -> int:
    return int(max(int(base), int(floor_value)))


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


def _load_distillation_plan(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _distillation_assignment_map(plan: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in plan.get("assignments", []) if isinstance(plan, dict) else []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("student_bot_id", "")).strip().lower()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _prioritize_targets_for_distillation(targets: list[str], assign_map: dict[str, dict]) -> tuple[list[str], int]:
    if not targets or not assign_map:
        return targets, 0

    student_targets: list[str] = []
    other_targets: list[str] = []
    for t in targets:
        bot_id = _normalized_bot_id_from_script(t)
        if bot_id in assign_map:
            student_targets.append(t)
        else:
            other_targets.append(t)

    return student_targets + other_targets, len(student_targets)


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
        "--adaptive-swap-gate",
        action="store_true",
        default=os.getenv("RETRAIN_ADAPTIVE_SWAP_GATE", "1").strip() == "1",
        help="Auto-relax swap gate when swap is persistently above threshold.",
    )
    parser.add_argument(
        "--adaptive-swap-step-gb",
        type=float,
        default=float(os.getenv("RETRAIN_ADAPTIVE_SWAP_STEP_GB", "0.4")),
        help="Step increase for adaptive swap gate.",
    )
    parser.add_argument(
        "--adaptive-swap-max-gb",
        type=float,
        default=float(os.getenv("RETRAIN_ADAPTIVE_SWAP_MAX_GB", "3.5")),
        help="Upper cap for adaptive swap gate relaxation.",
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
    parser.add_argument(
        "--weekly-model-archive",
        action="store_true",
        default=os.getenv("WEEKLY_MODEL_ARCHIVE_ENABLED", "1").strip() == "1",
        help="Archive old model artifacts once per ISO week.",
    )
    parser.add_argument(
        "--archive-keep-per-bot",
        type=int,
        default=int(os.getenv("MODEL_ARCHIVE_KEEP_PER_BOT", "8")),
    )
    parser.add_argument(
        "--archive-min-age-hours",
        type=float,
        default=float(os.getenv("MODEL_ARCHIVE_MIN_AGE_HOURS", "24")),
    )
    parser.add_argument(
        "--distillation-priority",
        action="store_true",
        default=os.getenv("RETRAIN_DISTILLATION_PRIORITY", "1").strip() == "1",
        help="Prioritize student bots from distillation plan in retrain order.",
    )
    parser.add_argument(
        "--distillation-plan",
        default=os.getenv("DISTILLATION_PLAN_PATH", os.path.join(PROJECT_ROOT, "governance", "distillation", "teacher_student_plan_latest.json")),
        help="Path to teacher-student distillation plan JSON.",
    )
    parser.add_argument(
        "--distillation-student-extra-pass",
        type=int,
        default=int(os.getenv("RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS", "0")),
        help="Optional extra retrain passes for prioritized student bots (count).",
    )
    parser.add_argument(
        "--require-data-quality-floor",
        action="store_true",
        default=os.getenv("RETRAIN_REQUIRE_DATA_QUALITY_FLOOR", "1").strip() == "1",
        help="Block retrain start when snapshot coverage/divergence quality floor is not met.",
    )
    parser.add_argument(
        "--min-snapshot-coverage-ratio",
        type=float,
        default=float(os.getenv("RETRAIN_MIN_SNAPSHOT_COVERAGE_RATIO", "0.75")),
    )
    parser.add_argument(
        "--max-data-divergence-spread",
        type=float,
        default=float(os.getenv("RETRAIN_MAX_DATA_DIVERGENCE_SPREAD", "0.04")),
    )
    parser.add_argument(
        "--snapshot-coverage-file",
        default=os.getenv("SNAPSHOT_COVERAGE_FILE", os.path.join(PROJECT_ROOT, "governance", "health", "snapshot_coverage_latest.json")),
    )
    parser.add_argument(
        "--data-divergence-file",
        default=os.getenv("DATA_DIVERGENCE_FILE", os.path.join(PROJECT_ROOT, "governance", "health", "data_source_divergence_latest.json")),
    )
    parser.add_argument(
        "--regime-balance",
        action="store_true",
        default=os.getenv("RETRAIN_REGIME_BALANCE", "1").strip() == "1",
        help="Distribute retrain targets across regime buckets instead of clustered order.",
    )
    parser.add_argument(
        "--regime-focus",
        default=os.getenv("RETRAIN_REGIME_FOCUS", ""),
        help="Optional comma-separated regime focus list (trend,mean_revert,shock,liquidity,other).",
    )
    parser.add_argument(
        "--canary-priority-file",
        default=os.getenv("RETRAIN_CANARY_PRIORITY_FILE", CANARY_DIAGNOSTICS),
    )
    parser.add_argument(
        "--canary-priority-top-n",
        type=int,
        default=int(os.getenv("RETRAIN_CANARY_PRIORITY_TOP_N", "10")),
        help="Prioritize top recurring canary-failing bots at the front of retrain queue.",
    )
    parser.add_argument(
        "--retire-persistent-losers",
        action="store_true",
        default=os.getenv("RETRAIN_RETIRE_PERSISTENT_LOSERS", "1").strip() == "1",
        help="Run persistent-loser retirement automation after retrain summary.",
    )
    parser.add_argument(
        "--retire-apply",
        action="store_true",
        default=os.getenv("RETRAIN_RETIRE_APPLY", "0").strip() == "1",
        help="Apply retirement changes to registry (otherwise report-only).",
    )
    parser.add_argument(
        "--retire-lookback-days",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_LOOKBACK_DAYS", "14")),
    )
    parser.add_argument(
        "--retire-min-fail-days",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MIN_FAIL_DAYS", "7")),
    )
    parser.add_argument(
        "--retire-min-no-improvement-streak",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MIN_NO_IMPROVEMENT_STREAK", "3")),
    )
    parser.add_argument(
        "--retire-max-per-run",
        type=int,
        default=int(os.getenv("RETRAIN_RETIRE_MAX_PER_RUN", "4")),
    )
    parser.add_argument(
        "--new-bot-boost",
        action="store_true",
        default=os.getenv("RETRAIN_NEW_BOT_BOOST", "1").strip() == "1",
        help="Accelerate learning for newer bots with stronger teacher pressure and extra passes.",
    )
    parser.add_argument(
        "--new-bot-max-runs",
        type=int,
        default=int(os.getenv("RETRAIN_NEW_BOT_MAX_RUNS", "24")),
        help="Bots at or below this walk-forward run count are treated as newer bots.",
    )
    parser.add_argument(
        "--new-bot-extra-pass",
        type=int,
        default=int(os.getenv("RETRAIN_NEW_BOT_EXTRA_PASS", "2")),
        help="Extra retrain passes to apply to newer bots.",
    )
    parser.add_argument(
        "--new-bot-distillation-weight",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_DISTILLATION_WEIGHT", "0.45")),
        help="Minimum teacher blend weight for newer bots when distillation metadata exists.",
    )
    parser.add_argument(
        "--new-bot-feature-freshness-max-age-seconds",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_FEATURE_FRESHNESS_MAX_AGE_SECONDS", "12")),
        help="Tighter feature freshness age budget for newer bots.",
    )
    parser.add_argument(
        "--new-bot-neutral-hold-min",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_NEUTRAL_HOLD_MIN", "0.68")),
    )
    parser.add_argument(
        "--new-bot-neutral-hold-margin-min",
        type=float,
        default=float(os.getenv("RETRAIN_NEW_BOT_NEUTRAL_HOLD_MARGIN_MIN", "0.08")),
    )
    parser.add_argument(
        "--new-bot-regime-auto-focus",
        action="store_true",
        default=os.getenv("RETRAIN_NEW_BOT_REGIME_AUTO_FOCUS", "1").strip() == "1",
        help="Auto-focus retrain queue on worst failing regime segments when boost mode is enabled.",
    )
    parser.add_argument(
        "--walk-forward-file",
        default=os.getenv("RETRAIN_WALK_FORWARD_FILE", os.path.join(PROJECT_ROOT, "governance", "walk_forward", "walk_forward_latest.json")),
    )
    parser.add_argument(
        "--promotion-bottleneck-priority",
        action="store_true",
        default=os.getenv("RETRAIN_PROMOTION_BOTTLENECK_PRIORITY", "1").strip() == "1",
        help="Use promotion bottleneck profile to bias regime focus and priority queue.",
    )
    parser.add_argument(
        "--promotion-bottleneck-file",
        default=os.getenv("RETRAIN_PROMOTION_BOTTLENECK_FILE", PROMOTION_BOTTLENECK_PATH),
    )
    parser.add_argument(
        "--refresh-promotion-artifacts",
        action="store_true",
        default=os.getenv("RETRAIN_REFRESH_PROMOTION_ARTIFACTS", "1").strip() == "1",
        help="Refresh walk-forward, promotion gate, graduation gate, and leak/overfit artifacts before master update.",
    )
    parser.add_argument(
        "--weekly-gate-blocker-report",
        action="store_true",
        default=os.getenv("RETRAIN_WEEKLY_GATE_BLOCKER_REPORT", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-hygiene",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_HYGIENE", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-apply-prune",
        action="store_true",
        default=os.getenv("RETRAIN_LIFECYCLE_APPLY_PRUNE", "1").strip() == "1",
    )
    parser.add_argument(
        "--lifecycle-keep-backups",
        type=int,
        default=int(os.getenv("RETRAIN_LIFECYCLE_KEEP_BACKUPS", "25")),
    )
    parser.add_argument(
        "--lifecycle-min-free-gb",
        type=float,
        default=float(os.getenv("RETRAIN_LIFECYCLE_MIN_FREE_GB", "10")),
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

    data_quality_summary: dict = {}
    if args.require_data_quality_floor:
        dq_ok, dq_reason, dq_detail = _check_data_quality_floor(
            coverage_file=str(args.snapshot_coverage_file),
            divergence_file=str(args.data_divergence_file),
            min_coverage_ratio=float(args.min_snapshot_coverage_ratio),
            max_divergence_spread=float(args.max_data_divergence_spread),
        )
        data_quality_summary = {"ok": dq_ok, "reason": dq_reason, **dq_detail}
        print(
            "Data quality floor: "
            f"ok={dq_ok} "
            f"coverage={dq_detail.get('coverage_ratio', 0.0):.4f}/{dq_detail.get('min_coverage_ratio', 0.0):.4f} "
            f"divergence={dq_detail.get('worst_relative_spread', 0.0):.4f}/{dq_detail.get('max_divergence_spread', 0.0):.4f}"
        )
        if not dq_ok:
            print(f"Retrain blocked by data quality floor: {dq_reason}")
            return 1

    if args.promotion_bottleneck_priority and os.path.exists(PROMOTION_BOTTLENECK_SCRIPT):
        _ = run_cmd([VENV_PY, PROMOTION_BOTTLENECK_SCRIPT, "--json"], args.dry_run, os.environ.copy(), extra_nice=max(args.ops_extra_nice, 0))

    effective_canary_priority_top_n = int(args.canary_priority_top_n)
    effective_distillation_extra_pass = int(args.distillation_student_extra_pass)
    effective_regime_focus = str(args.regime_focus or "")

    if args.new_bot_boost:
        effective_canary_priority_top_n = _effective_int(effective_canary_priority_top_n, 30)
        effective_distillation_extra_pass = _effective_int(effective_distillation_extra_pass, int(args.new_bot_extra_pass))
        args.distillation_priority = True
        args.regime_balance = True
        if args.new_bot_regime_auto_focus and not effective_regime_focus:
            auto_focus = _derive_regime_focus_from_readiness(PROMOTION_READINESS_PATH, top_n=2)
            if auto_focus:
                effective_regime_focus = auto_focus

    bottleneck_profile = _load_json_file(str(args.promotion_bottleneck_file)) if args.promotion_bottleneck_priority else {}
    if bottleneck_profile:
        rec = bottleneck_profile.get("recommended_retrain_profile") if isinstance(bottleneck_profile.get("recommended_retrain_profile"), dict) else {}
        if (not effective_regime_focus) and str(rec.get("RETRAIN_REGIME_FOCUS", "")).strip():
            effective_regime_focus = str(rec.get("RETRAIN_REGIME_FOCUS", "")).strip()
        try:
            effective_canary_priority_top_n = max(
                effective_canary_priority_top_n,
                int(rec.get("RETRAIN_CANARY_PRIORITY_TOP_N", 0) or 0),
            )
        except Exception:
            pass
        try:
            rec_targets = int(rec.get("RETRAIN_MAX_TARGETS", 0) or 0)
            if rec_targets > 0:
                args.max_targets = min(int(args.max_targets), rec_targets) if int(args.max_targets) > 0 else rec_targets
        except Exception:
            pass

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

    if effective_regime_focus:
        focused = _apply_regime_focus(targets, str(effective_regime_focus))
        if focused:
            targets = focused
        else:
            print(f"WARN: regime_focus produced zero targets, keeping original list: {effective_regime_focus}")

    if args.regime_balance:
        targets = _apply_regime_balanced_order(targets)

    targets, canary_priority_selected = _apply_canary_priority(
        targets,
        diagnostics_file=str(args.canary_priority_file),
        top_n=int(effective_canary_priority_top_n),
    )

    wf_runs = _load_walk_forward_runs(str(args.walk_forward_file))
    new_bot_targets = _select_new_bot_targets(targets, wf_runs, int(args.new_bot_max_runs)) if args.new_bot_boost else []
    new_bot_ids = {_normalized_bot_id_from_script(x) for x in new_bot_targets} if args.new_bot_boost else set()

    distill_plan = _load_distillation_plan(args.distillation_plan) if args.distillation_priority else {}
    distill_assign_map = _distillation_assignment_map(distill_plan)
    distill_selected = 0
    if args.distillation_priority and distill_assign_map:
        targets, distill_selected = _prioritize_targets_for_distillation(targets, distill_assign_map)

    if args.distillation_priority and distill_assign_map and int(effective_distillation_extra_pass) > 0:
        student_targets = [t for t in targets if _normalized_bot_id_from_script(t) in distill_assign_map]
        extra_n = min(max(int(effective_distillation_extra_pass), 0), len(student_targets))
        if extra_n > 0:
            targets = targets + student_targets[:extra_n]

    if args.new_bot_boost and new_bot_targets and int(args.new_bot_extra_pass) > 0:
        extra_new_n = min(max(int(args.new_bot_extra_pass), 0), len(new_bot_targets))
        if extra_new_n > 0:
            targets = targets + new_bot_targets[:extra_new_n]

    child_env = _build_child_env(args.thread_cap)
    child_env["DISTILLATION_ENABLED"] = "1" if args.distillation_priority else "0"
    child_env["DISTILLATION_PLAN_PATH"] = str(args.distillation_plan)
    child_env["REQUIRE_CANARY_PROMOTION_GATE"] = "1"
    if args.new_bot_boost:
        child_env["TRADE_BEHAVIOR_STRICT_NEUTRAL_GATE"] = "1"
        child_env["TRADE_BEHAVIOR_HOLD_NEUTRAL_MIN"] = f"{float(args.new_bot_neutral_hold_min):.4f}"
        child_env["TRADE_BEHAVIOR_HOLD_MARGIN_MIN"] = f"{float(args.new_bot_neutral_hold_margin_min):.4f}"
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
        f"adaptive={args.adaptive_swap_gate} "
        f"adaptive_step_gb={args.adaptive_swap_step_gb:.2f} "
        f"adaptive_cap_gb={args.adaptive_swap_max_gb:.2f} "
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
    print(
        "Queue strategy: "
        f"regime_balance={args.regime_balance} "
        f"regime_focus={effective_regime_focus or 'all'} "
        f"canary_priority_selected={canary_priority_selected} "
        f"new_bot_boost={args.new_bot_boost} "
        f"bottleneck_profile_used={bool(bottleneck_profile)}"
    )

    started = datetime.now(timezone.utc).isoformat()
    print(f"Weekly retrain start (UTC): {started}")
    print(f"Targets: {len(targets)}")
    if args.new_bot_boost:
        print(
            "New-bot boost: "
            f"new_targets={len(new_bot_targets)} "
            f"extra_pass={args.new_bot_extra_pass} "
            f"teacher_weight_floor={float(args.new_bot_distillation_weight):.2f} "
            f"feature_freshness_max_age_s={float(args.new_bot_feature_freshness_max_age_seconds):.1f}"
        )
    print("Efficiency tip: keep streaming/video/browser load low during retrain windows.")

    failures: list[str] = []
    skipped_by_memory: list[str] = []
    target_outcomes: list[dict] = []
    dynamic_max_swap_gb = float(args.max_swap_gb)
    for target in targets:
        target_name = os.path.basename(target)
        allowed = _wait_for_memory_gate(
            enabled=args.memory_guard,
            min_free_pct=args.min_free_pct,
            max_swap_gb=dynamic_max_swap_gb,
            poll_seconds=args.memory_poll_seconds,
            max_wait_seconds=args.memory_max_wait_seconds,
            label=target_name,
            dry_run=args.dry_run,
        )
        if (not allowed) and args.adaptive_swap_gate and (not args.dry_run):
            ok_now, reason_now, snap_now = _memory_ready(min_free_pct=args.min_free_pct, max_swap_gb=dynamic_max_swap_gb)
            swap_now = float(snap_now.get("swap_used_gb", 0.0) or 0.0)
            free_now = float(snap_now.get("free_pct", 0.0) or 0.0)
            if (not ok_now) and ("swap" in reason_now) and (swap_now > dynamic_max_swap_gb) and (free_now >= args.min_free_pct):
                next_swap = min(float(args.adaptive_swap_max_gb), max(dynamic_max_swap_gb + float(args.adaptive_swap_step_gb), swap_now + 0.10))
                if next_swap > dynamic_max_swap_gb:
                    print(
                        f"[AdaptiveSwapGate] raise label={target_name} "
                        f"from={dynamic_max_swap_gb:.2f} to={next_swap:.2f} "
                        f"reason={reason_now}"
                    )
                    dynamic_max_swap_gb = next_swap
                    allowed = _wait_for_memory_gate(
                        enabled=args.memory_guard,
                        min_free_pct=args.min_free_pct,
                        max_swap_gb=dynamic_max_swap_gb,
                        poll_seconds=args.memory_poll_seconds,
                        max_wait_seconds=max(int(args.memory_max_wait_seconds / 2), 120),
                        label=target_name,
                        dry_run=args.dry_run,
                    )
        if not allowed:
            skipped_by_memory.append(target)
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "skipped_memory"})
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
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "skipped_thermal"})
            continue

        target_env = dict(child_env)
        bot_id = _normalized_bot_id_from_script(target)
        is_new_bot = bot_id in new_bot_ids if args.new_bot_boost else False
        if is_new_bot:
            target_env["FEATURE_FRESHNESS_GUARD_ENABLED"] = "1"
            target_env["FEATURE_FRESHNESS_MAX_AGE_SECONDS"] = f"{float(args.new_bot_feature_freshness_max_age_seconds):.4f}"
            target_env["RETRAIN_NEW_BOT_MODE"] = "1"
        dist_row = distill_assign_map.get(bot_id, {}) if args.distillation_priority else {}
        if dist_row:
            teacher_ids = [str((t or {}).get("bot_id", "")).strip() for t in (dist_row.get("teachers", []) or []) if str((t or {}).get("bot_id", "")).strip()]
            target_env["DISTILLATION_STUDENT"] = "1"
            target_env["DISTILLATION_TEACHERS"] = ",".join(teacher_ids)
            base_tw = float(dist_row.get("teacher_blend_weight", 0.30) or 0.30)
            if is_new_bot:
                base_tw = max(base_tw, float(args.new_bot_distillation_weight))
            target_env["DISTILLATION_TEACHER_WEIGHT"] = str(base_tw)
        else:
            target_env["DISTILLATION_STUDENT"] = "0"

        rc = run_cmd([VENV_PY, target], args.dry_run, target_env)
        if rc != 0:
            failures.append(target)
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "failed", "rc": rc})
            print(f"FAIL: {target} (exit={rc})")
            if not args.continue_on_error:
                break
        else:
            target_outcomes.append({"bot_id": _normalized_bot_id_from_script(target), "target": target, "status": "trained"})

        if not args.dry_run:
            gc.collect()
            time.sleep(max(args.between_target_sleep_seconds, 0))

    if failures and not args.continue_on_error:
        print("Stopped early due to failure.")

    prev_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
    prev_acc_map = _registry_accuracy_map(REGISTRY_PATH)
    curr_registry_snapshot = dict(prev_registry_snapshot)
    curr_acc_map = dict(prev_acc_map)
    master_update_status = "skipped"
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
        max_swap_gb=dynamic_max_swap_gb,
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
        precheck_failures: list[str] = []
        if args.refresh_promotion_artifacts:
            artifact_steps = [
                (WALK_FORWARD_VALIDATE_SCRIPT, False),
                (WALK_FORWARD_PROMOTION_GATE_SCRIPT, True),
                (PROMOTION_READINESS_SCRIPT, False),
                (PROMOTION_BOTTLENECK_SCRIPT, False),
                (NEW_BOT_GRADUATION_SCRIPT, True),
                (LEAK_OVERFIT_GUARD_SCRIPT, True),
            ]
            for script_path, required_ok in artifact_steps:
                if not os.path.exists(script_path):
                    if required_ok:
                        precheck_failures.append(f"missing:{os.path.basename(script_path)}")
                    continue
                cmd = [VENV_PY, script_path]
                if script_path in {PROMOTION_READINESS_SCRIPT, PROMOTION_BOTTLENECK_SCRIPT, NEW_BOT_GRADUATION_SCRIPT, LEAK_OVERFIT_GUARD_SCRIPT}:
                    cmd.append("--json")
                rc_art = run_cmd(cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
                if rc_art != 0 and required_ok:
                    precheck_failures.append(f"{os.path.basename(script_path)}:exit_{rc_art}")

        if precheck_failures:
            master_update_status = "precheck_failed"
            print("FAIL: promotion prechecks failed")
            for item in precheck_failures:
                print(f" - {item}")
        else:
            rc = run_cmd([sys.executable, MASTER_RUNNER], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))
            if rc != 0:
                master_update_status = f"failed_exit_{rc}"
                print(f"FAIL: master bot update (exit={rc})")
            else:
                master_update_status = "updated"

            if not args.dry_run and rc == 0:
                curr_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
                curr_acc_map = _registry_accuracy_map(REGISTRY_PATH)
                rollback_bad, rollback_reason = _should_rollback_registry(prev_registry_snapshot, curr_registry_snapshot)
                if rollback_bad and os.path.exists(registry_backup_path):
                    shutil.copy2(registry_backup_path, REGISTRY_PATH)
                    curr_registry_snapshot = _registry_snapshot(REGISTRY_PATH)
                    curr_acc_map = _registry_accuracy_map(REGISTRY_PATH)
                    master_update_status = f"rolled_back:{rollback_reason}"
                    print(f"[Rollback] restored previous master registry reason={rollback_reason}")
                else:
                    print(f"[Rollback] registry check status={rollback_reason}")
    else:
        master_update_status = "skipped_memory_or_thermal_gate"
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

    if args.weekly_model_archive and (not args.dry_run) and _weekly_archive_due():
        print("Running weekly model archive pass...")
        if os.path.exists(ARCHIVE_OLD_MODELS):
            _ = run_cmd(
                [
                    VENV_PY,
                    ARCHIVE_OLD_MODELS,
                    "--keep-per-bot",
                    str(max(args.archive_keep_per_bot, 1)),
                    "--min-age-hours",
                    str(max(args.archive_min_age_hours, 0.0)),
                ],
                args.dry_run,
                child_env,
                extra_nice=max(args.ops_extra_nice, 0),
            )
            _write_weekly_archive_stamp()
        else:
            print(f"WARN: archive script missing: {ARCHIVE_OLD_MODELS}")

    if args.retire_persistent_losers and (not args.dry_run) and os.path.exists(RETIRE_PERSISTENT_LOSERS):
        print("Running persistent-loser retirement scan...")
        retire_cmd = [
            VENV_PY,
            RETIRE_PERSISTENT_LOSERS,
            "--lookback-days",
            str(max(args.retire_lookback_days, 1)),
            "--min-fail-days",
            str(max(args.retire_min_fail_days, 1)),
            "--min-no-improvement-streak",
            str(max(args.retire_min_no_improvement_streak, 1)),
            "--max-retire-per-run",
            str(max(args.retire_max_per_run, 0)),
            "--json",
        ]
        if args.retire_apply:
            retire_cmd.append("--apply")
        _ = run_cmd(retire_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if args.weekly_gate_blocker_report and os.path.exists(WEEKLY_GATE_BLOCKER_REPORT_SCRIPT):
        _ = run_cmd([VENV_PY, WEEKLY_GATE_BLOCKER_REPORT_SCRIPT, "--json"], args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    scorecard_path = _write_retrain_scorecard(
        started_utc=started,
        ended_utc=ended,
        target_count=len(targets),
        failures=failures,
        skipped_by_memory=skipped_by_memory,
        target_outcomes=target_outcomes,
        prev_registry_snapshot=prev_registry_snapshot,
        curr_registry_snapshot=curr_registry_snapshot,
        prev_acc=prev_acc_map,
        curr_acc=curr_acc_map,
        master_update_status=master_update_status,
        data_quality_summary=data_quality_summary,
        canary_priority_selected=canary_priority_selected,
        distill_selected=distill_selected,
    )
    print(f"Retrain scorecard written: {scorecard_path}")

    if args.lifecycle_hygiene and os.path.exists(MODEL_LIFECYCLE_HYGIENE_SCRIPT):
        lifecycle_cmd = [
            VENV_PY,
            MODEL_LIFECYCLE_HYGIENE_SCRIPT,
            "--keep-backups",
            str(max(int(args.lifecycle_keep_backups), 1)),
            "--min-free-gb",
            str(max(float(args.lifecycle_min_free_gb), 0.0)),
            "--json",
        ]
        if args.lifecycle_apply_prune:
            lifecycle_cmd.append("--apply-prune")
        if str(master_update_status).startswith("updated") or str(master_update_status).startswith("rolled_back"):
            lifecycle_cmd.append("--update-last-known-good")
        _ = run_cmd(lifecycle_cmd, args.dry_run, child_env, extra_nice=max(args.ops_extra_nice, 0))

    if skipped_by_memory:
        print("Completed with memory-gate skips.")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
