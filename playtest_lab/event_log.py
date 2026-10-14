from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Event:
    seq: int
    event_type: str
    payload: Dict[str, Any]
    truth_hash: str
    previous_hash: str
    event_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLog:
    """Append-only hash-chained event log.

    The log does not prove the game is correct, but it makes mid-run mutation
    and replay mismatches observable instead of relying on a start/end hash only.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1].event_hash if self._events else "GENESIS"

    def append(self, event_type: str, payload: Dict[str, Any], truth_hash: str) -> Event:
        seq = len(self._events)
        previous_hash = self.head_hash
        material = {
            "seq": seq,
            "event_type": event_type,
            "payload": payload,
            "truth_hash": truth_hash,
            "previous_hash": previous_hash,
        }
        event_hash = hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()
        event = Event(
            seq=seq,
            event_type=event_type,
            payload=json.loads(_canonical(payload)),
            truth_hash=truth_hash,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
        self._events.append(event)
        return event

    def verify(self) -> bool:
        previous = "GENESIS"
        for expected_seq, event in enumerate(self._events):
            if event.seq != expected_seq or event.previous_hash != previous:
                return False
            material = {
                "seq": event.seq,
                "event_type": event.event_type,
                "payload": event.payload,
                "truth_hash": event.truth_hash,
                "previous_hash": event.previous_hash,
            }
            expected_hash = hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()
            if expected_hash != event.event_hash:
                return False
            previous = event.event_hash
        return True


def decision_plan_from_events(events: Iterable[Event]) -> Dict[str, str]:
    plan: Dict[str, str] = {}
    for event in events:
        if event.event_type == "decision":
            plan[str(event.payload["decision_id"])] = str(event.payload["action"])
    return plan
