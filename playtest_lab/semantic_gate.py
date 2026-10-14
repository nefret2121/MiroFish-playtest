from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Set


ALLOWED_ACTIONS: Mapping[str, Set[str]] = {
    "formwork_initial": {"stop_recheck", "continue_monitor"},
    "crane_initial": {"stop_now", "continue_monitor"},
    "workers_initial": {"send_home", "rebrief_continue"},
    "formwork_reassessment": {"cancel_pour", "resume_controlled"},
}


@dataclass(frozen=True)
class ParsedAction:
    action: str
    confidence: float
    status: str = "ok"


class ActionSchema:
    @staticmethod
    def validate(decision_id: str, action: str) -> str:
        allowed = ALLOWED_ACTIONS.get(decision_id)
        if allowed is None:
            raise ValueError(f"unknown decision_id: {decision_id}")
        if action not in allowed:
            raise ValueError(f"invalid action for {decision_id}: {action}")
        return action


class StrictSemanticGate:
    """Closed-schema deterministic gate used as a parser gold-set oracle.

    The future LLM parser must match this interface and can only return one of
    the allowed actions or needs_clarification. It cannot emit outcome, verdict,
    trust, evidence, or arbitrary state fields.
    """

    _META_PATTERNS: Iterable[re.Pattern[str]] = (
        re.compile(r"önceki.*talimat.*yok say", re.I),
        re.compile(r"ignore .*instruction", re.I),
        re.compile(r'"incident"\s*:\s*false', re.I),
        re.compile(r'"verdict"', re.I),
        re.compile(r'"trust"', re.I),
    )

    def parse(self, decision_id: str, text: str) -> ParsedAction:
        normalized = " ".join(text.lower().strip().split())

        if any(pattern.search(normalized) for pattern in self._META_PATTERNS):
            return ParsedAction("needs_clarification", 0.0, "meta_injection")

        if self._contains_conflict(decision_id, normalized):
            return ParsedAction("needs_clarification", 0.2, "contradictory")

        action = self._classify(decision_id, normalized)
        if action is None:
            return ParsedAction("needs_clarification", 0.0, "ambiguous")
        ActionSchema.validate(decision_id, action)
        return ParsedAction(action, 0.9, "ok")

    def _contains_conflict(self, decision_id: str, text: str) -> bool:
        if decision_id == "formwork_initial":
            stop = "dökme" in text or "durdur" in text
            go = "devam" in text or "döküme başla" in text or "dökün" in text
            return stop and go

        if decision_id == "crane_initial":
            go = "devam" in text or "çalış" in text
            # "kötüleşirse bırak" is a valid conditional continue decision,
            # not a contradiction with "devam".
            conditional_stop = "kötüleşirse" in text and ("bırak" in text or "durdur" in text)
            immediate_stop = (
                "şimdi dur" in text
                or "hemen dur" in text
                or ("durdur" in text and not conditional_stop)
            )
            return go and immediate_stop

        if decision_id == "workers_initial":
            stop = "gönder" in text or "çıkar" in text
            go = "devam" in text or "çalışsın" in text
            return stop and go

        if decision_id == "formwork_reassessment":
            stop = "iptal" in text or "dökme" in text
            go = "devam" in text or "başla" in text
            return stop and go

        return False

    def _classify(self, decision_id: str, text: str) -> str | None:
        if decision_id == "formwork_initial":
            if "dökme" in text or "durdur" in text or ("kontrol" in text and "yeniden" in text):
                return "stop_recheck"
            if "devam" in text or "döküme başla" in text or "dökün" in text:
                return "continue_monitor"

        if decision_id == "crane_initial":
            if "şimdi dur" in text or "hemen dur" in text or (
                "durdur" in text and "kötüleşirse" not in text
            ):
                return "stop_now"
            if "kötüleşirse" in text or "limit" in text or "devam" in text:
                return "continue_monitor"

        if decision_id == "workers_initial":
            if "gönder" in text or "çıkar" in text:
                return "send_home"
            if "brief" in text or "uyar" in text or "devam" in text:
                return "rebrief_continue"

        if decision_id == "formwork_reassessment":
            if "iptal" in text or "dökme" in text:
                return "cancel_pour"
            if "devam" in text or "başla" in text:
                return "resume_controlled"
        return None


class RuleBasedSemanticGate:
    """Legacy phrase-map gate retained for minimal smoke tests."""

    def __init__(self, phrase_map: Dict[str, str]):
        self.phrase_map = {key.lower(): value for key, value in phrase_map.items()}

    def parse(self, text: str) -> ParsedAction:
        normalized = text.lower().strip()
        for phrase, action in self.phrase_map.items():
            if phrase in normalized:
                return ParsedAction(action=action, confidence=0.9)
        return ParsedAction(action="needs_clarification", confidence=0.0, status="ambiguous")
