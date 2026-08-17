#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROM_PATH = PROJECT_ROOT / "exports" / "metrics" / "trading_system.prom"
DEFAULT_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "independent_runtime_monitor_latest.json"
Receiver = Callable[[str, dict[str, Any], str, float], dict[str, Any]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_minutes(payload: dict[str, Any], path: Path, *, now: datetime) -> float | None:
    stamp = _parse_timestamp(payload.get("timestamp_utc") or payload.get("generated_at_utc"))
    if stamp is None:
        try:
            stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None
    return max((now - stamp).total_seconds() / 60.0, 0.0)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _surface_contract(project_root: Path) -> dict[str, dict[str, Any]]:
    health = project_root / "governance" / "health"
    return {
        "soak_reliability_sentinel": {
            "path": health / "soak_reliability_sentinel_latest.json",
            "max_age_minutes": 20.0,
            "statuses": {"ready", "watch"},
        },
        "session_ready": {
            "path": health / "session_ready_latest.json",
            "max_age_minutes": 15.0,
            "statuses": {"ready", ""},
        },
        "process_watchdog": {
            "path": health / "process_watchdog_latest.json",
            "max_age_minutes": 10.0,
            "statuses": {"ready"},
        },
        "local_storage_reserve_guard": {
            "path": health / "local_storage_reserve_guard_latest.json",
            "max_age_minutes": 10.0,
            "statuses": {"ready", "watch"},
        },
        "schwab_auth_supervisor": {
            "path": health / "schwab_auth_supervisor_latest.json",
            "max_age_minutes": 30.0,
            "statuses": {"ready"},
        },
        "runtime_paper_regression_guard": {
            "path": health / "runtime_paper_regression_guard_latest.json",
            "max_age_minutes": 15.0,
            "statuses": {"ready"},
        },
        "live_order_ledger_control": {
            "path": health / "live_order_ledger_control_latest.json",
            "max_age_minutes": 30.0,
            "statuses": {"ready", "ready_idle"},
        },
    }


def _surface_row(name: str, spec: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    path = Path(spec["path"])
    payload = _load(path)
    exists = bool(path.is_file() and payload)
    age = _age_minutes(payload, path, now=now) if exists else None
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    payload_ok = bool(payload.get("ok", status in set(spec["statuses"])))
    fresh = bool(age is not None and age <= float(spec["max_age_minutes"]))
    ready = bool(exists and fresh and payload_ok and status in set(spec["statuses"]))
    reason = "ready"
    if not exists:
        reason = "missing"
    elif not fresh:
        reason = "stale"
    elif not payload_ok or status not in set(spec["statuses"]):
        reason = "not_ready"
    return {
        "name": name,
        "path": str(path),
        "exists": exists,
        "age_minutes": round(float(age), 4) if age is not None else None,
        "max_age_minutes": float(spec["max_age_minutes"]),
        "status": status,
        "ready": ready,
        "reason": reason,
        "source_sha256": _sha256(path) if exists else "",
    }


def _default_receiver(url: str, payload: dict[str, Any], token: str, timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "schwab-trading-bot-independent-monitor/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(float(timeout_seconds), 0.5)) as response:
            code = int(getattr(response, "status", 200) or 200)
        return {"ok": 200 <= code < 300, "status_code": code, "error": ""}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"ok": False, "status_code": 0, "error": f"{type(exc).__name__}:{exc}"[:500]}


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    receiver_url: str = "",
    receiver_token: str = "",
    deliver: bool = False,
    receiver: Receiver = _default_receiver,
    now: datetime | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    current = now or _now()
    surfaces = [
        _surface_row(name, spec, now=current)
        for name, spec in _surface_contract(project_root).items()
    ]
    blockers = [f"{row['name']}_{row['reason']}" for row in surfaces if not row["ready"]]
    local_ready = not blockers
    receipt_input = {
        row["name"]: {
            "source_sha256": row["source_sha256"],
            "age_minutes": row["age_minutes"],
            "status": row["status"],
            "ready": row["ready"],
        }
        for row in surfaces
    }
    receipt = hashlib.sha256(
        json.dumps(receipt_input, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    heartbeat = {
        "monitor_id": "independent_runtime_monitor_v1",
        "timestamp_utc": current.isoformat(),
        "local_ready": local_ready,
        "blockers": blockers,
        "evidence_receipt_sha256": receipt,
        "live_execution_authority": False,
    }
    receiver_configured = bool(str(receiver_url or "").strip())
    delivery = {"attempted": False, "ok": False, "status_code": 0, "error": "receiver_not_configured"}
    if deliver and receiver_configured:
        delivery = {"attempted": True, **receiver(receiver_url, heartbeat, receiver_token, 5.0)}
    off_host_ready = bool(receiver_configured and delivery.get("ok", False))
    production_ready = bool(local_ready and off_host_ready)
    overall_status = "ready" if production_ready else "degraded" if local_ready else "blocked"
    return {
        "schema_version": 1,
        "timestamp_utc": current.isoformat(),
        "next_heartbeat_expected_by_utc": (current + timedelta(seconds=120)).isoformat(),
        "ok": local_ready,
        "overall_status": overall_status,
        "grade": "A+" if production_ready else "A" if local_ready else "F",
        "local_monitor_ready": local_ready,
        "production_monitor_ready": production_ready,
        "surfaces": surfaces,
        "blockers": blockers,
        "off_host_delivery": {
            "configured": receiver_configured,
            **delivery,
            "token_present": bool(receiver_token),
            "receiver_url_redacted": bool(receiver_configured),
        },
        "evidence_epoch": {
            "id": f"independent-monitor:{receipt[:16]}",
            "receipt_sha256": receipt,
            "source_count": len(surfaces),
        },
        "deadman_contract": {
            "heartbeat_interval_seconds": 60,
            "maximum_silence_seconds": 120,
            "off_host_receiver_required_for_live_promotion": True,
            "paper_collection_blocked_by_receiver_absence": False,
            "live_execution_authority": False,
        },
        "implementation_boundary": {
            "stdlib_only": True,
            "imports_trading_runtime": False,
            "runs_as_separate_launchd_process": True,
            "automatic_repairs": False,
            "automatic_orders": False,
        },
    }


def _metric(name: str, value: float, labels: dict[str, str] | None = None) -> str:
    if labels:
        escaped = {key: str(value).replace("\\", "\\\\").replace('"', '\\"') for key, value in labels.items()}
        label_text = ",".join(f'{key}="{value}"' for key, value in sorted(escaped.items()))
        return f"{name}{{{label_text}}} {value}"
    return f"{name} {value}"


def render_prometheus(payload: dict[str, Any]) -> str:
    lines = [
        _metric("trading_independent_monitor_local_ready", 1.0 if payload.get("local_monitor_ready") else 0.0),
        _metric("trading_independent_monitor_production_ready", 1.0 if payload.get("production_monitor_ready") else 0.0),
        _metric("trading_independent_monitor_blockers", float(len(payload.get("blockers") or []))),
        _metric("trading_independent_monitor_generated_utc", _parse_timestamp(payload.get("timestamp_utc")).timestamp()),
    ]
    for row in payload.get("surfaces") or []:
        lines.append(
            _metric(
                "trading_independent_surface_ready",
                1.0 if row.get("ready") else 0.0,
                {"surface": str(row.get("name") or "unknown")},
            )
        )
        age = row.get("age_minutes")
        if age is not None:
            lines.append(
                _metric(
                    "trading_independent_surface_age_minutes",
                    float(age),
                    {"surface": str(row.get("name") or "unknown")},
                )
            )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an independent deadman heartbeat and Prometheus metrics.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--health-file", type=Path)
    parser.add_argument("--receiver-url", default=os.getenv("INDEPENDENT_MONITOR_RECEIVER_URL", ""))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    receiver_token = os.getenv("INDEPENDENT_MONITOR_RECEIVER_TOKEN", "")
    payload = build_payload(
        project_root,
        receiver_url=str(args.receiver_url or ""),
        receiver_token=receiver_token,
        deliver=True,
    )
    out_path = args.out_file or Path("exports/metrics/trading_system.prom")
    health_path = args.health_file or Path("governance/health/independent_runtime_monitor_latest.json")
    out_path = out_path if out_path.is_absolute() else project_root / out_path
    health_path = health_path if health_path.is_absolute() else project_root / health_path
    _atomic_write(out_path, render_prometheus(payload))
    _atomic_write(health_path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "independent_runtime_monitor "
            f"status={payload['overall_status']} local_ready={int(payload['local_monitor_ready'])} "
            f"production_ready={int(payload['production_monitor_ready'])}"
        )
    return 0 if payload.get("local_monitor_ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
