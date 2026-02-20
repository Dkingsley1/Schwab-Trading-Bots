import argparse
import glob
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Freshness:
    status: str
    age_seconds: float


def _safe_parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _seconds_ago(ts: Optional[datetime]) -> float:
    if ts is None:
        return 1e12
    return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)


def _freshness(age_seconds: float, fresh_s: int, stale_s: int) -> Freshness:
    if age_seconds <= fresh_s:
        return Freshness("FRESH", age_seconds)
    if age_seconds <= stale_s:
        return Freshness("WARM", age_seconds)
    return Freshness("STALE", age_seconds)


def _tail_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = bytearray()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            b = f.read(1)
            if b == b"\n" and buf:
                break
            if b != b"\n":
                buf.extend(b)
        line = bytes(reversed(buf)).decode("utf-8", errors="replace").strip()

    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def _latest_file(pattern: str) -> Optional[Path]:
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _shadow_loop_pids() -> list[int]:
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout or ""
    except Exception:
        return []

    pids: list[int] = []
    for line in out.splitlines():
        if "run_shadow_training_loop.py" not in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except Exception:
            continue
    return sorted(set(pids))


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += int(fp.stat().st_size)
            except OSError:
                continue
    return total


def _human_bytes(n: int) -> str:
    size = float(max(n, 0))
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f}{units[i]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick health dashboard for shadow trading bot runtime.")
    parser.add_argument("--fresh-seconds", type=int, default=180)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    dec_path = _latest_file(str(PROJECT_ROOT / "decision_explanations" / "shadow" / "decision_explanations_*.jsonl"))
    gov_path = _latest_file(str(PROJECT_ROOT / "governance" / "shadow" / "master_control_*.jsonl"))
    retrain_path = _latest_file(str(PROJECT_ROOT / "governance" / "shadow" / "auto_retrain_events_*.jsonl"))

    dec_last = _tail_last_jsonl(dec_path) if dec_path else None
    gov_last = _tail_last_jsonl(gov_path) if gov_path else None
    retrain_last = _tail_last_jsonl(retrain_path) if retrain_path else None

    dec_ts = _safe_parse_ts((dec_last or {}).get("timestamp_utc"))
    gov_ts = _safe_parse_ts((gov_last or {}).get("timestamp_utc"))
    retrain_ts = _safe_parse_ts((retrain_last or {}).get("timestamp_utc"))

    dec_age = _seconds_ago(dec_ts)
    gov_age = _seconds_ago(gov_ts)
    retrain_age = _seconds_ago(retrain_ts)

    dec_state = _freshness(dec_age, args.fresh_seconds, args.stale_seconds)
    gov_state = _freshness(gov_age, args.fresh_seconds, args.stale_seconds)

    latest_decision_csv = PROJECT_ROOT / "exports" / "csv" / "latest_decision_explanations.csv"
    latest_master_csv = PROJECT_ROOT / "exports" / "csv" / "latest_master_control.csv"

    now = datetime.now(timezone.utc)
    dec_csv_age = (now - datetime.fromtimestamp(latest_decision_csv.stat().st_mtime, tz=timezone.utc)).total_seconds() if latest_decision_csv.exists() else 1e12
    mas_csv_age = (now - datetime.fromtimestamp(latest_master_csv.stat().st_mtime, tz=timezone.utc)).total_seconds() if latest_master_csv.exists() else 1e12
    dec_csv_state = _freshness(dec_csv_age, 600, 3600)
    mas_csv_state = _freshness(mas_csv_age, 600, 3600)

    loop_pids = _shadow_loop_pids()
    loop_live = len(loop_pids) > 0

    retrain_event = (retrain_last or {}).get("event", "none")
    retrain_reason = (retrain_last or {}).get("reason", "")

    storage_targets = {
        ".venv312": PROJECT_ROOT / ".venv312",
        "decision_explanations": PROJECT_ROOT / "decision_explanations",
        "decisions": PROJECT_ROOT / "decisions",
        "exports": PROJECT_ROOT / "exports",
        "governance": PROJECT_ROOT / "governance",
        "models": PROJECT_ROOT / "models",
        "logs": PROJECT_ROOT / "logs",
    }
    storage_sizes = {k: _dir_size_bytes(v) for k, v in storage_targets.items()}
    project_total = _dir_size_bytes(PROJECT_ROOT)

    overall = "OK"
    if not loop_live or dec_state.status == "STALE" or gov_state.status == "STALE":
        overall = "DEGRADED"

    payload = {
        "overall": overall,
        "loop_live": loop_live,
        "loop_pids": loop_pids,
        "decision_log": {
            "status": dec_state.status,
            "age_seconds": round(dec_state.age_seconds, 1),
            "path": str(dec_path) if dec_path else None,
            "last_symbol": (dec_last or {}).get("symbol"),
            "last_strategy": (dec_last or {}).get("strategy"),
            "last_status": (dec_last or {}).get("status"),
        },
        "governance_log": {
            "status": gov_state.status,
            "age_seconds": round(gov_state.age_seconds, 1),
            "path": str(gov_path) if gov_path else None,
            "last_symbol": (gov_last or {}).get("symbol"),
            "last_master_action": (gov_last or {}).get("master_action"),
            "last_options_master_action": ((gov_last or {}).get("options_master") or {}).get("action"),
        },
        "retrain": {
            "event": retrain_event,
            "reason": retrain_reason,
            "age_seconds": round(retrain_age, 1),
            "path": str(retrain_path) if retrain_path else None,
        },
        "finder_csv": {
            "decision_csv": {"status": dec_csv_state.status, "age_seconds": round(dec_csv_state.age_seconds, 1), "path": str(latest_decision_csv)},
            "master_csv": {"status": mas_csv_state.status, "age_seconds": round(mas_csv_state.age_seconds, 1), "path": str(latest_master_csv)},
        },
        "storage": {
            "project_total_bytes": project_total,
            "project_total_human": _human_bytes(project_total),
            "by_path": {
                k: {
                    "bytes": v,
                    "human": _human_bytes(v),
                    "path": str(storage_targets[k]),
                }
                for k, v in storage_sizes.items()
            },
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    print(
        "HEALTH {overall} | loop={loop} | decisions={dstat}({dage:.0f}s) | governance={gstat}({gage:.0f}s) | "
        "retrain={revent}({rage:.0f}s) | csv=sub:{c1}({a1:.0f}s),master:{c2}({a2:.0f}s)".format(
            overall=overall,
            loop="LIVE" if loop_live else "DOWN",
            dstat=dec_state.status,
            dage=dec_state.age_seconds,
            gstat=gov_state.status,
            gage=gov_state.age_seconds,
            revent=retrain_event,
            rage=retrain_age,
            c1=dec_csv_state.status,
            a1=dec_csv_state.age_seconds,
            c2=mas_csv_state.status,
            a2=mas_csv_state.age_seconds,
        )
    )

    print(f"Loop PIDs: {loop_pids if loop_pids else 'none'}")
    print(f"Last decision: symbol={(dec_last or {}).get('symbol')} strategy={(dec_last or {}).get('strategy')} status={(dec_last or {}).get('status')}")
    print(f"Last governance: symbol={(gov_last or {}).get('symbol')} master_action={(gov_last or {}).get('master_action')}")
    print(f"Last retrain event: event={retrain_event} reason={retrain_reason or 'n/a'}")
    print(
        "Storage: total={total} | venv={venv} | decision_explanations={de} | decisions={ds} | exports={ex} | governance={gov}".format(
            total=_human_bytes(project_total),
            venv=_human_bytes(storage_sizes.get('.venv312', 0)),
            de=_human_bytes(storage_sizes.get('decision_explanations', 0)),
            ds=_human_bytes(storage_sizes.get('decisions', 0)),
            ex=_human_bytes(storage_sizes.get('exports', 0)),
            gov=_human_bytes(storage_sizes.get('governance', 0)),
        )
    )


if __name__ == "__main__":
    main()
