from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ValidityIssue:
    path: str
    line: int
    column: int
    rule_id: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _constant_number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _constant_number(node.operand)
        return -value if value is not None else None
    return None


def _constant_text(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip().lower()
    return ""


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


class _ValidityVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.issues: list[ValidityIssue] = []

    def _add(self, node: ast.AST, rule_id: str, message: str) -> None:
        self.issues.append(
            ValidityIssue(
                path=self.path,
                line=int(getattr(node, "lineno", 0) or 0),
                column=int(getattr(node, "col_offset", 0) or 0),
                rule_id=rule_id,
                message=message,
            )
        )

    def visit_Call(self, node: ast.Call) -> Any:
        name = _call_name(node)
        first = node.args[0] if node.args else None
        keyword_values = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
        }
        if name in {"shift", "pct_change", "diff"}:
            periods = _constant_number(keyword_values.get("periods") or first)
            if periods is not None and periods < 0:
                self._add(
                    node,
                    "future_period_access",
                    f"{name} with negative periods reads future observations",
                )
        if name == "rolling":
            center = keyword_values.get("center")
            if isinstance(center, ast.Constant) and center.value is True:
                self._add(
                    node,
                    "centered_rolling_window",
                    "centered rolling windows include future observations",
                )
        if name in {"bfill", "backfill"}:
            self._add(
                node,
                "backward_fill",
                "backward fill propagates future values into earlier rows",
            )
        if name == "fillna" and _constant_text(keyword_values.get("method")) in {
            "bfill",
            "backfill",
        }:
            self._add(
                node,
                "backward_fill",
                "fillna(method='bfill') propagates future values into earlier rows",
            )
        if name == "merge_asof" and _constant_text(keyword_values.get("direction")) in {
            "forward",
            "nearest",
        }:
            self._add(
                node,
                "forward_asof_join",
                "forward or nearest as-of joins are not point-in-time safe by default",
            )
        self.generic_visit(node)


def scan_source_text(source: str, *, path: str = "<memory>") -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [
            ValidityIssue(
                path=path,
                line=int(exc.lineno or 0),
                column=int(exc.offset or 0),
                rule_id="syntax_error",
                message=str(exc.msg or "invalid Python source"),
            ).to_dict()
        ]
    visitor = _ValidityVisitor(path)
    visitor.visit(tree)
    return [issue.to_dict() for issue in visitor.issues]


def scan_paths(paths: Iterable[str | Path]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    scanned: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.suffix != ".py":
            continue
        scanned.append(str(path))
        issues.extend(
            scan_source_text(path.read_text(encoding="utf-8"), path=str(path))
        )
    return {
        "ok": not issues,
        "scanned_file_count": len(scanned),
        "issue_count": len(issues),
        "scanned_files": scanned,
        "issues": issues,
    }


def _equal(left: Any, right: Any, tolerance: float) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def future_suffix_invariance(
    compute: Callable[[Sequence[float]], Sequence[Any]],
    values: Sequence[float],
    *,
    cut_points: Sequence[int] | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    source = [float(value) for value in values]
    baseline = list(compute(source))
    cuts = list(
        cut_points or (max(len(source) // 3, 1), max((2 * len(source)) // 3, 1))
    )
    errors: list[str] = []
    if len(baseline) != len(source):
        errors.append("output_length_mismatch")
    for cut in cuts:
        if cut <= 0 or cut >= len(source):
            continue
        perturbed = list(source)
        for index in range(cut, len(perturbed)):
            perturbed[index] = (index + 1) * 1000003.0
        candidate = list(compute(perturbed))
        if len(candidate) != len(baseline):
            errors.append(f"output_length_changed:cut={cut}")
            continue
        changed = [
            index
            for index in range(min(cut, len(baseline)))
            if not _equal(baseline[index], candidate[index], tolerance)
        ]
        if changed:
            errors.append(f"future_suffix_changed_prefix:cut={cut}:first={changed[0]}")
    return {
        "ok": not errors,
        "sample_count": len(source),
        "cut_points": cuts,
        "errors": errors,
    }


def recursive_warmup_stability(
    compute: Callable[[Sequence[float]], Sequence[Any]],
    values: Sequence[float],
    *,
    startup_lengths: Sequence[int] | None = None,
    comparison_points: int = 1,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    source = [float(value) for value in values]
    lengths = sorted(
        {
            int(length)
            for length in (startup_lengths or (max(len(source) // 2, 2), len(source)))
            if 1 < int(length) <= len(source)
        }
    )
    errors: list[str] = []
    outputs: dict[int, list[Any]] = {}
    for length in lengths:
        result = list(compute(source[-length:]))
        if len(result) != length:
            errors.append(f"output_length_mismatch:length={length}")
        outputs[length] = result
    if lengths:
        reference = outputs[lengths[-1]]
        points = max(min(int(comparison_points), len(reference)), 1)
        for length in lengths[:-1]:
            candidate = outputs[length]
            overlap = min(points, len(candidate), len(reference))
            if overlap <= 0:
                errors.append(f"no_comparable_output:length={length}")
                continue
            for offset in range(1, overlap + 1):
                if not _equal(candidate[-offset], reference[-offset], tolerance):
                    errors.append(
                        f"recursive_instability:length={length}:offset={offset}"
                    )
                    break
    return {
        "ok": not errors,
        "sample_count": len(source),
        "startup_lengths": lengths,
        "comparison_points": int(comparison_points),
        "errors": errors,
    }


def validity_contract_receipt(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(contract), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_validity_contract() -> dict[str, Any]:
    contract = {
        "point_in_time_features_required": True,
        "future_suffix_invariance_required": True,
        "recursive_warmup_stability_required": True,
        "high_confidence_static_lookahead_scan_required": True,
        "late_data_quarantine_required": True,
        "failure_behavior": "block_candidate_promotion_and_live_execution",
        "may_change_action_or_quantity": False,
    }
    return {**contract, "receipt_sha256": validity_contract_receipt(contract)}
