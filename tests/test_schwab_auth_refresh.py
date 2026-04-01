import json
import tempfile
from pathlib import Path

from scripts.ops import schwab_auth_refresh as sar


def test_token_status_reads_nested_epoch_expiry() -> None:
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        token_path.write_text(
            json.dumps(
                {
                    "creation_timestamp": 1773069288,
                    "token": {
                        "refresh_token": "refresh-token",
                        "access_token": "access-token",
                        "expires_at": 4102444800,
                    },
                }
            ),
            encoding="utf-8",
        )

        status = sar._token_status(token_path)

    assert status["exists"] is True
    assert status["expires_at"] == "4102444800"
    assert float(status["expires_in_seconds"]) > 0.0


def test_token_needs_refresh_uses_min_expiry_floor() -> None:
    status = {
        "exists": True,
        "size_bytes": 808,
        "expires_in_seconds": 450.0,
    }

    needs_refresh, reason = sar._token_needs_refresh(status, min_expires_seconds=300.0)
    assert needs_refresh is False
    assert reason == "token_ready"

    needs_refresh, reason = sar._token_needs_refresh(status, min_expires_seconds=600.0)
    assert needs_refresh is True
    assert reason.startswith("token_expiring_soon:")
