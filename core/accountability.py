import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_schema_version() -> int:
    try:
        return max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1)
    except Exception:
        return 2


def current_correlation() -> Dict[str, str]:
    return {
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
    }


def _ensure_message_contract(out: Dict[str, Any]) -> None:
    msg_id = str(out.get("message_id") or "").strip()
    if not msg_id:
        msg_id = str(uuid.uuid4())
        out["message_id"] = msg_id

    parent = str(out.get("parent_message_id") or "").strip()
    if not parent:
        parent = str(out.get("parent_decision_id") or "").strip()
    if (not parent) and isinstance(out.get("metadata"), dict):
        parent = str(out["metadata"].get("parent_message_id") or out["metadata"].get("parent_decision_id") or "").strip()
    if parent and ("parent_message_id" not in out):
        out["parent_message_id"] = parent


def enrich_log_row(
    row: Dict[str, Any],
    *,
    include_correlation: bool = True,
    include_schema: bool = True,
) -> Dict[str, Any]:
    out = dict(row or {})
    if include_schema and ("log_schema_version" not in out):
        out["log_schema_version"] = log_schema_version()

    if include_correlation:
        corr = current_correlation()
        if corr.get("run_id") and ("run_id" not in out):
            out["run_id"] = corr["run_id"]
        if corr.get("iter_id") and ("iter_id" not in out):
            out["iter_id"] = corr["iter_id"]

    _ensure_message_contract(out)
    return out


CHANNEL_SCHEMA_REQUIRED: Dict[str, tuple[str, ...]] = {
    "runtime": ("timestamp_utc", "event", "message_id"),
    "gate": ("timestamp_utc", "symbol", "gate", "message_id"),
    "ingress": ("timestamp_utc", "symbol", "status", "message_id"),
    "api": ("timestamp_utc", "symbol", "endpoint", "status", "message_id"),
    "loop_state": ("timestamp_utc", "state", "iter", "message_id"),
    "decision": ("timestamp_utc", "symbol", "action", "message_id"),
    "risk": ("timestamp_utc", "message_id"),
    "execution_guard": ("timestamp_utc", "event", "status", "message_id"),
    "softguard": ("timestamp_utc", "event", "status", "message_id"),
    "auth": ("timestamp_utc", "event", "status", "message_id"),
}

HOT_QUEUE_CHANNELS = {
    "runtime",
    "gate",
    "ingress",
    "api",
    "loop_state",
    "decision",
    "risk",
}


def _schema_errors(payload: Dict[str, Any], *, schema: str) -> List[str]:
    req = CHANNEL_SCHEMA_REQUIRED.get(str(schema or "").strip(), ())
    if not req:
        return []
    errors: List[str] = []
    for key in req:
        val = payload.get(key)
        if val is None:
            errors.append(f"missing:{key}")
            continue
        if isinstance(val, str) and (not val.strip()):
            errors.append(f"missing:{key}")
    return errors


def _write_lines(path: str, lines: Sequence[str]) -> bool:
    if not lines:
        return True
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(lines))
        return True
    except Exception:
        return False


def safe_append_jsonl_batch(
    path: str,
    rows: Iterable[Dict[str, Any]],
    *,
    project_root: str = "",
    source: str = "",
) -> int:
    payloads = [enrich_log_row(dict(r or {})) for r in rows]
    if not payloads:
        return 0

    lines = [json.dumps(p, ensure_ascii=True) + "\n" for p in payloads]
    if _write_lines(path, lines):
        return len(payloads)

    _emit_write_failure_event(
        project_root=project_root,
        source=source or "jsonl_writer_batch",
        target_path=path,
        error=RuntimeError("batch_append_failed"),
    )
    return 0


def _default_queue_db(project_root: str) -> str:
    return str(Path(project_root) / "data" / "bot_channel_queue.sqlite3") if project_root else ""


def _queue_publish(
    *,
    project_root: str,
    channel: str,
    source_path: str,
    payloads: Sequence[Dict[str, Any]],
    queue_db_path: str = "",
) -> None:
    if not project_root:
        return
    if str(channel or "") not in HOT_QUEUE_CHANNELS:
        return

    try:
        from core.channel_queue import ChannelQueue, default_queue_db_path, queue_enabled

        if not queue_enabled():
            return

        db_path = queue_db_path or default_queue_db_path(project_root)
        q = ChannelQueue(db_path)

        require_consumer = os.getenv("BOT_CHANNEL_QUEUE_REQUIRE_RECENT_CONSUMER", "1").strip().lower() in {"1", "true", "yes", "on"}
        consumer_max_age_seconds = max(int(os.getenv("BOT_CHANNEL_QUEUE_CONSUMER_MAX_AGE_SECONDS", "86400")), 60)
        if require_consumer and not q.has_recent_consumer(channel=channel, max_age_seconds=consumer_max_age_seconds):
            return

        q.enqueue_batch(
            channel=channel,
            payloads=list(payloads),
            source_path=source_path,
        )
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source="channel_queue.enqueue",
            target_path=queue_db_path or _default_queue_db(project_root),
            error=exc,
        )


def _schema_violation_log(
    *,
    project_root: str,
    source: str,
    channel: str,
    target_path: str,
    payload: Dict[str, Any],
    errors: Sequence[str],
) -> None:
    if not project_root:
        return
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(project_root, "governance", "events", f"channel_schema_violations_{day}.jsonl")
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": "channel_schema_violation",
        "source": str(source or "unknown"),
        "channel": str(channel or ""),
        "target_path": str(target_path or ""),
        "errors": list(errors),
        "payload": dict(payload),
        "log_schema_version": log_schema_version(),
    }
    corr = current_correlation()
    if corr.get("run_id"):
        row["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        row["iter_id"] = corr["iter_id"]
    safe_append_jsonl(out_path, row, project_root=project_root, source="channel_schema_violation")


def _schema_strict_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.getenv("CHANNEL_SCHEMA_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}


def safe_append_channel_batch(
    path: str,
    rows: Iterable[Dict[str, Any]],
    *,
    project_root: str = "",
    source: str = "",
    channel: str = "",
    schema: str = "",
    mirror_paths: Optional[Sequence[str]] = None,
    strict_schema: Optional[bool] = None,
    queue_db_path: str = "",
) -> int:
    raw_payloads = [dict(r or {}) for r in rows]
    if not raw_payloads:
        return 0

    ch = str(channel or "").strip()
    sch = str(schema or ch).strip()
    strict = _schema_strict_enabled(strict_schema)

    valid_payloads: List[Dict[str, Any]] = []
    for raw in raw_payloads:
        payload = enrich_log_row(raw)
        if ch and ("channel" not in payload):
            payload["channel"] = ch
        errors = _schema_errors(payload, schema=sch)
        if errors:
            _schema_violation_log(
                project_root=project_root,
                source=source,
                channel=ch or sch,
                target_path=path,
                payload=payload,
                errors=errors,
            )
            if strict:
                continue
            payload["schema_errors"] = list(errors)
            payload["schema_valid"] = False
        else:
            payload["schema_valid"] = True
        valid_payloads.append(payload)

    if not valid_payloads:
        return 0

    lines = [json.dumps(p, ensure_ascii=True) + "\n" for p in valid_payloads]
    if not _write_lines(path, lines):
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "channel_writer",
            target_path=path,
            error=RuntimeError("channel_batch_append_failed"),
        )
        return 0

    mirrors = [str(p) for p in (mirror_paths or []) if str(p or "").strip()]
    for mirror in mirrors:
        if os.path.abspath(mirror) == os.path.abspath(path):
            continue
        if not _write_lines(mirror, lines):
            _emit_write_failure_event(
                project_root=project_root,
                source=source or "channel_writer_mirror",
                target_path=mirror,
                error=RuntimeError("channel_mirror_append_failed"),
            )

    if ch:
        _queue_publish(
            project_root=project_root,
            channel=ch,
            source_path=path,
            payloads=valid_payloads,
            queue_db_path=queue_db_path,
        )

    return len(valid_payloads)


def safe_append_channel_event(
    path: str,
    row: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    channel: str = "",
    schema: str = "",
    mirror_paths: Optional[Sequence[str]] = None,
    strict_schema: Optional[bool] = None,
    queue_db_path: str = "",
) -> bool:
    wrote = safe_append_channel_batch(
        path,
        [row],
        project_root=project_root,
        source=source,
        channel=channel,
        schema=schema,
        mirror_paths=mirror_paths,
        strict_schema=strict_schema,
        queue_db_path=queue_db_path,
    )
    return wrote > 0


def safe_append_jsonl(
    path: str,
    row: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
) -> bool:
    wrote = safe_append_jsonl_batch(path, [row], project_root=project_root, source=source)
    return wrote > 0


def safe_write_json(
    path: str,
    payload: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    indent: int = 2,
) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=indent)
        return True
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "json_writer",
            target_path=path,
            error=exc,
        )
        return False


def safe_write_json_atomic(
    path: str,
    payload: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    indent: int = 2,
    marker: bool = True,
) -> bool:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=indent), encoding="utf-8")
        tmp.replace(target)

        if marker:
            marker_path = target.with_suffix(target.suffix + ".ok")
            marker_payload = {
                "timestamp_utc": now_utc_iso(),
                "source": str(source or "json_writer_atomic"),
                "payload_sha256": sha256_json_obj(payload),
                "target": str(target),
            }
            marker_path.write_text(json.dumps(marker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "json_writer_atomic",
            target_path=path,
            error=exc,
        )
        return False


def _emit_write_failure_event(
    *,
    project_root: str,
    source: str,
    target_path: str,
    error: Exception,
) -> None:
    if not project_root:
        print(f"[WriteFail] source={source} target={target_path} err={error}")
        return

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    fail_path = os.path.join(project_root, "governance", "events", f"write_failures_{day}.jsonl")
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": "write_failure",
        "source": source,
        "target_path": target_path,
        "error": str(error),
        "error_type": type(error).__name__,
        "log_schema_version": log_schema_version(),
    }
    corr = current_correlation()
    if corr.get("run_id"):
        row["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        row["iter_id"] = corr["iter_id"]

    try:
        os.makedirs(os.path.dirname(fail_path), exist_ok=True)
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception as inner_exc:
        print(
            f"[WriteFail] source={source} target={target_path} err={error} "
            f"failure_event_write_err={inner_exc}"
        )


def git_commit(project_root: str) -> str:
    if not project_root:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return str(proc.stdout or "").strip()
        return ""
    except Exception:
        return ""


def sha256_file(path: str) -> str:
    if not path or (not os.path.exists(path)):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def sha256_json_obj(obj: Any) -> str:
    try:
        encoded = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return ""


def _bot_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _bot_field_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    changed: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(before.keys()) | set(after.keys()))
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        if b != a:
            changed[key] = {"before": b, "after": a}
    return changed


def compute_registry_mutation(
    *,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    before_map = _bot_map(before)
    after_map = _bot_map(after)

    bot_diffs = []
    for bot_id in sorted(set(before_map.keys()) | set(after_map.keys())):
        b = before_map.get(bot_id)
        a = after_map.get(bot_id)
        if b is None and a is not None:
            bot_diffs.append({"bot_id": bot_id, "change_type": "added", "after": a})
            continue
        if a is None and b is not None:
            bot_diffs.append({"bot_id": bot_id, "change_type": "removed", "before": b})
            continue
        if b is None or a is None:
            continue
        fields = _bot_field_diff(b, a)
        if fields:
            bot_diffs.append(
                {
                    "bot_id": bot_id,
                    "change_type": "updated",
                    "changed_fields": fields,
                }
            )

    return {
        "bots_total_before": int(len(before_map)),
        "bots_total_after": int(len(after_map)),
        "bot_diff_count": int(len(bot_diffs)),
        "bot_diffs": bot_diffs,
        "registry_sha256_before": sha256_json_obj(before),
        "registry_sha256_after": sha256_json_obj(after),
    }


def write_registry_mutation_journal(
    *,
    project_root: str,
    actor: str,
    reason: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    mutation = compute_registry_mutation(before=before, after=after)
    payload: Dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "actor": actor,
        "reason": reason,
        "mutation": mutation,
        "log_schema_version": log_schema_version(),
    }
    if extra:
        payload["extra"] = extra

    corr = current_correlation()
    if corr.get("run_id"):
        payload["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        payload["iter_id"] = corr["iter_id"]

    audit_dir = os.path.join(project_root, "governance", "audits")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    journal_path = os.path.join(audit_dir, f"registry_mutation_journal_{day}.jsonl")
    latest_path = os.path.join(audit_dir, "registry_mutation_latest.json")

    safe_append_jsonl(
        journal_path,
        payload,
        project_root=project_root,
        source="registry_mutation_journal",
    )
    safe_write_json(
        latest_path,
        payload,
        project_root=project_root,
        source="registry_mutation_latest",
    )
    return journal_path
