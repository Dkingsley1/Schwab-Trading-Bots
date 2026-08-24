from pathlib import Path

from core.formal_safety_invariants import bounded_verify_order_safety, inspect_tla_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_bounded_order_model_and_tla_spec_expose_formal_proof_debt() -> None:
    bounded = bounded_verify_order_safety()
    spec = inspect_tla_spec(PROJECT_ROOT / "formal" / "OrderSafety.tla")

    assert bounded["ok"] is True
    assert bounded["state_count"] > 0
    assert spec["ok"] is True
    assert spec["tlc_executed"] is False
    assert spec["formal_proof_ready"] is False
    assert "tlc_model_check_receipt" in spec["external_evidence_debt"]
