import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone


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


def _memory_snapshot() -> dict:
    snap = {}
    try:
        proc = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
        out = proc.stdout or ""
        for raw in out.splitlines():
            line = raw.strip()
            low = line.lower()
            if "free percentage" in low:
                snap["free_pct"] = float(line.split(":", 1)[-1].strip().replace("%", ""))
            elif "available percentage" in low:
                snap["available_pct"] = float(line.split(":", 1)[-1].strip().replace("%", ""))
    except Exception:
        pass

    try:
        proc = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, check=False)
        out = (proc.stdout or "").strip()
        if "used =" in out:
            used_part = out.split("used =", 1)[1].strip().split()[0]
            snap["swap_used_gb"] = _parse_size_to_gb(used_part)
    except Exception:
        pass
    return snap


def _thermal_snapshot() -> dict:
    snap = {}
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


def _lock_state(lock_path: str) -> str:
    if not os.path.exists(lock_path):
        return "free"
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return "busy"
        try:
            obj = json.loads(raw)
            return f"busy(pid={obj.get('pid')})"
        except Exception:
            return "busy"
    except Exception:
        return "busy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live dashboard for memory/thermal/retrain gate state.")
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lock_path = os.getenv("MLX_RETRAIN_LOCK_PATH", os.path.join(project_root, "governance", "mlx_retrain.lock"))

    min_free = float(os.getenv("RETRAIN_MIN_FREE_PCT", "22"))
    max_swap = float(os.getenv("RETRAIN_MAX_SWAP_GB", "1.0"))
    min_cpu = float(os.getenv("RETRAIN_THERMAL_MIN_CPU_SPEED_LIMIT", "75"))
    min_sched = float(os.getenv("RETRAIN_THERMAL_MIN_SCHEDULER_LIMIT", "75"))

    while True:
        now = datetime.now(timezone.utc).isoformat()
        mem = _memory_snapshot()
        therm = _thermal_snapshot()
        free_pct = float(mem.get("free_pct", mem.get("available_pct", 0.0)) or 0.0)
        swap_gb = float(mem.get("swap_used_gb", 0.0) or 0.0)
        cpu_lim = float(therm.get("cpu_speed_limit", 100.0) or 100.0)
        sched_lim = float(therm.get("scheduler_limit", 100.0) or 100.0)

        gate_ok = (free_pct >= min_free) and (swap_gb <= max_swap) and (cpu_lim >= min_cpu) and (sched_lim >= min_sched)
        state = "OPEN" if gate_ok else "BLOCKED"

        print(
            f"{now} | gate={state} | free={free_pct:.1f}%/{min_free:.1f}% "
            f"swap={swap_gb:.2f}GB/{max_swap:.2f}GB "
            f"cpu_limit={cpu_lim:.0f}/{min_cpu:.0f} sched_limit={sched_lim:.0f}/{min_sched:.0f} "
            f"mlx_lock={_lock_state(lock_path)}"
        )

        if args.once:
            return 0
        time.sleep(max(args.interval_seconds, 1))


if __name__ == "__main__":
    raise SystemExit(main())
