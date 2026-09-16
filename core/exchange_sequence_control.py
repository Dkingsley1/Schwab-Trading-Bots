from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SequencePolicy:
    first_sequence_minimum: int = 0
    reset_sequence: int = 1


class ExchangeSequenceGuard:
    """ITCH/OUCH-inspired channel sequencing without protocol or venue authority."""

    def __init__(self, policy: SequencePolicy | None = None) -> None:
        self.policy = policy or SequencePolicy()
        self._channels: dict[str, dict[str, Any]] = {}

    def ingest(
        self,
        *,
        channel: str,
        session_id: str,
        sequence_number: int,
        payload: Mapping[str, Any],
        session_reset: bool = False,
    ) -> dict[str, Any]:
        if (
            not channel
            or not session_id
            or sequence_number < self.policy.first_sequence_minimum
        ):
            return self._result(
                False, "invalid_envelope", channel, session_id, sequence_number
            )

        digest = _digest(payload)
        state = self._channels.get(channel)
        if state is None:
            self._channels[channel] = self._state(session_id, sequence_number, digest)
            return self._result(
                True, "session_initialized", channel, session_id, sequence_number
            )

        if session_id != state["session_id"]:
            valid_reset = (
                session_reset and sequence_number == self.policy.reset_sequence
            )
            if not valid_reset:
                return self._result(
                    False,
                    "session_change_requires_explicit_reset",
                    channel,
                    session_id,
                    sequence_number,
                )
            self._channels[channel] = self._state(session_id, sequence_number, digest)
            return self._result(
                True, "session_reset", channel, session_id, sequence_number
            )

        last = int(state["last_sequence"])
        if sequence_number == last:
            disposition = (
                "duplicate_idempotent"
                if digest == state["last_digest"]
                else "duplicate_conflict"
            )
            return self._result(
                False, disposition, channel, session_id, sequence_number
            )
        if sequence_number < last:
            return self._result(
                False, "stale_out_of_order", channel, session_id, sequence_number
            )
        if sequence_number > last + 1:
            return {
                **self._result(
                    False, "sequence_gap", channel, session_id, sequence_number
                ),
                "recovery_required": True,
                "missing_range": [last + 1, sequence_number - 1],
            }

        state.update(last_sequence=sequence_number, last_digest=digest)
        return self._result(True, "contiguous", channel, session_id, sequence_number)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": {
                "first_sequence_minimum": self.policy.first_sequence_minimum,
                "reset_sequence": self.policy.reset_sequence,
            },
            "channels": {
                key: dict(value) for key, value in sorted(self._channels.items())
            },
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "ExchangeSequenceGuard":
        policy = dict(snapshot.get("policy") or {})
        guard = cls(SequencePolicy(**policy))
        guard._channels = {
            str(key): dict(value)
            for key, value in dict(snapshot.get("channels") or {}).items()
        }
        return guard

    @staticmethod
    def _state(session_id: str, sequence_number: int, digest: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "last_sequence": sequence_number,
            "last_digest": digest,
        }

    @staticmethod
    def _result(
        accepted: bool,
        disposition: str,
        channel: str,
        session_id: str,
        sequence_number: int,
    ) -> dict[str, Any]:
        return {
            "accepted": accepted,
            "disposition": disposition,
            "channel": channel,
            "session_id": session_id,
            "sequence_number": sequence_number,
            "recovery_required": disposition in {"sequence_gap", "duplicate_conflict"},
            "execution_authority": False,
        }
