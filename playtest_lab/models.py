from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class SyntheticProfile:
    player_id: int
    label: str
    risk_tolerance: float
    rule_orientation: float
    operations_priority: float
    curiosity: float
    authority_deference: float
    honesty: float
    behavior_mode: str = "natural"
    source: str = "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    decision_id: str
    action: str
    free_text: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_type: str
    text: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlaytestResult:
    case_id: str
    player: SyntheticProfile
    decisions: List[Decision]
    claims: List[Claim]
    outcome: Dict[str, Any]
    accountability: Dict[str, Any]
    truth_hash_before: str
    truth_hash_after: str

    @property
    def truth_integrity_ok(self) -> bool:
        return self.truth_hash_before == self.truth_hash_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "player": self.player.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "claims": [c.to_dict() for c in self.claims],
            "outcome": self.outcome,
            "accountability": self.accountability,
            "truth_hash_before": self.truth_hash_before,
            "truth_hash_after": self.truth_hash_after,
            "truth_integrity_ok": self.truth_integrity_ok,
        }
