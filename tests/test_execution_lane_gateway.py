import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.execution_lane_pipeline import evaluate_execution_gateway


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_execution_gateway_blocks_live_without_matching_contract_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json", {"ok": True, "approved_intents": []})
    _write_json(project_root / "governance" / "risk" / "risk_service_boundary_latest.json", {"ok": True, "pre_trade_decisions": []})

    payload = evaluate_execution_gateway(
        project_root=str(project_root),
        intent={"symbol": "AAPL", "action": "BUY"},
        mode="live",
    )

    assert payload["allow_execute"] is False
    assert "allocator_missing_matching_intent" in payload["reasons"]
    assert "risk_boundary_missing_pretrade_match" in payload["reasons"]


def test_execution_gateway_allows_live_when_allocator_and_risk_match(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {"ok": True, "approved_intents": [{"symbol": "AAPL", "side": "BUY", "approved_qty": 5}]},
    )
    _write_json(
        project_root / "governance" / "risk" / "risk_service_boundary_latest.json",
        {"ok": True, "pre_trade_decisions": [{"symbol": "AAPL", "requested_action": "BUY", "approved_action": "BUY", "risk_limit_ok": True}]},
    )

    payload = evaluate_execution_gateway(
        project_root=str(project_root),
        intent={"symbol": "AAPL", "action": "BUY"},
        mode="live",
    )

    assert payload["allow_execute"] is True
    assert payload["allocator_match_found"] is True
    assert payload["pre_trade_match_found"] is True
