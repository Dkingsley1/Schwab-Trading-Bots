from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "timestamp":
        return _datetime(value) is not None
    return False


def run_checkpoint(
    *,
    checkpoint_id: str,
    rows: Sequence[Mapping[str, Any]],
    expectations: Mapping[str, Any],
    observed_at_utc: str,
) -> dict[str, Any]:
    now = _datetime(observed_at_utc)
    if now is None:
        raise ValueError("observed_at_utc must be an ISO-8601 timestamp")
    issues: list[dict[str, Any]] = []
    required = [str(value) for value in expectations.get("required_columns") or []]
    expected_types = dict(expectations.get("types") or {})
    ranges = dict(expectations.get("ranges") or {})
    freshness = dict(expectations.get("freshness") or {})

    for index, raw in enumerate(rows):
        row = dict(raw)
        for column in required:
            if column not in row or row[column] is None:
                issues.append(
                    {"row": index, "column": column, "reason": "required_value_missing"}
                )
        for column, expected in expected_types.items():
            if (
                column in row
                and row[column] is not None
                and not _type_ok(row[column], str(expected))
            ):
                issues.append(
                    {"row": index, "column": column, "reason": "type_mismatch"}
                )
        for column, bounds_raw in ranges.items():
            if column not in row or not isinstance(row[column], (int, float)):
                continue
            bounds = dict(bounds_raw)
            if "min" in bounds and float(row[column]) < float(bounds["min"]):
                issues.append(
                    {"row": index, "column": column, "reason": "below_minimum"}
                )
            if "max" in bounds and float(row[column]) > float(bounds["max"]):
                issues.append(
                    {"row": index, "column": column, "reason": "above_maximum"}
                )
        if freshness:
            column = str(freshness.get("column") or "")
            parsed = _datetime(row.get(column))
            max_age = float(freshness.get("max_age_seconds") or 0.0)
            if parsed is None:
                issues.append(
                    {
                        "row": index,
                        "column": column,
                        "reason": "freshness_timestamp_invalid",
                    }
                )
            elif (now - parsed).total_seconds() < 0:
                issues.append(
                    {"row": index, "column": column, "reason": "future_timestamp"}
                )
            elif (now - parsed).total_seconds() > max_age:
                issues.append({"row": index, "column": column, "reason": "stale_row"})

    for column in expectations.get("unique_columns") or []:
        seen: dict[Any, int] = {}
        for index, raw in enumerate(rows):
            value = raw.get(column)
            if value in seen:
                issues.append(
                    {
                        "row": index,
                        "column": column,
                        "reason": "duplicate_value",
                        "first_row": seen[value],
                    }
                )
            else:
                seen[value] = index

    for column in expectations.get("monotonic_columns") or []:
        previous: Any = None
        for index, raw in enumerate(rows):
            value = raw.get(column)
            comparable = _datetime(value) or value
            if (
                previous is not None
                and comparable is not None
                and comparable < previous
            ):
                issues.append(
                    {"row": index, "column": column, "reason": "monotonicity_violation"}
                )
            if comparable is not None:
                previous = comparable

    invalid_rows = sorted({int(issue["row"]) for issue in issues})
    failure_action = str(expectations.get("failure_action") or "quarantine")
    status = (
        "passed"
        if not issues
        else ("blocked" if failure_action == "block" else "quarantined")
    )
    return {
        "checkpoint_id": checkpoint_id,
        "ok": not issues,
        "status": status,
        "row_count": len(rows),
        "valid_row_count": len(rows) - len(invalid_rows),
        "invalid_row_count": len(invalid_rows),
        "quarantine_row_indices": invalid_rows,
        "issues": issues,
        "declarative_engine": "great_expectations_inspired_stdlib",
        "great_expectations_runtime_installed": False,
        "execution_authority": False,
    }
