from __future__ import annotations

import hashlib
from typing import Any, Dict

from .models import SyntheticProfile


def _unit_interval(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big")
    return number / float((1 << 64) - 1)


def profile_from_mirofish(agent: Dict[str, Any], player_id: int) -> SyntheticProfile:
    """Convert a MiroFish/OASIS persona dict into playtest traits.

    MVP deliberately uses stable hashing rather than a second LLM call.
    Later we can add an explicit trait-extraction prompt with tests.
    """
    persona = str(agent.get("persona") or agent.get("bio") or agent.get("name") or player_id)
    name = str(agent.get("name") or agent.get("username") or f"agent-{player_id}")

    risk = _unit_interval(persona, "risk")
    rules = _unit_interval(persona, "rules")
    ops = _unit_interval(persona, "operations")
    curiosity = _unit_interval(persona, "curiosity")
    deference = _unit_interval(persona, "deference")
    honesty = _unit_interval(persona, "honesty")

    label = "mirofish"
    return SyntheticProfile(
        player_id=player_id,
        label=label,
        risk_tolerance=round(risk, 4),
        rule_orientation=round(rules, 4),
        operations_priority=round(ops, 4),
        curiosity=round(curiosity, 4),
        authority_deference=round(deference, 4),
        honesty=round(honesty, 4),
    )
