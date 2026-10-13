from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ParsedAction:
    action: str
    confidence: float


class RuleBasedSemanticGate:
    """Tiny MVP gate for human/free-text smoke tests.

    This is NOT the final LLM semantic gate. It exists so the deterministic
    case engine can be tested independently from model behavior.
    """

    def __init__(self, phrase_map: Dict[str, str]):
        self.phrase_map = {key.lower(): value for key, value in phrase_map.items()}

    def parse(self, text: str) -> ParsedAction:
        normalized = text.lower().strip()
        for phrase, action in self.phrase_map.items():
            if phrase in normalized:
                return ParsedAction(action=action, confidence=0.9)
        return ParsedAction(action="unknown", confidence=0.0)
