from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import SyntheticProfile


def _unit_interval(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big")
    return number / float((1 << 64) - 1)


def profile_from_mirofish(agent: Dict[str, Any], player_id: int) -> SyntheticProfile:
    """Convert a real MiroFish/OASIS persona dict into playtest traits.

    MiroFish writes Reddit-style profiles with fields such as:
    user_id, username, name, bio, persona, age, gender, mbti, country,
    profession and interested_topics.

    The MVP uses stable hashing rather than a second LLM call, which makes
    import deterministic and cheap. A learned/LLM trait extractor can be
    introduced later behind this boundary.
    """
    persona = str(agent.get("persona") or agent.get("bio") or agent.get("name") or player_id)
    name = str(agent.get("name") or agent.get("username") or f"agent-{player_id}")

    risk = _unit_interval(persona, "risk")
    rules = _unit_interval(persona, "rules")
    ops = _unit_interval(persona, "operations")
    curiosity = _unit_interval(persona, "curiosity")
    deference = _unit_interval(persona, "deference")
    honesty = _unit_interval(persona, "honesty")

    return SyntheticProfile(
        player_id=player_id,
        label=f"mirofish:{name}",
        risk_tolerance=round(risk, 4),
        rule_orientation=round(rules, 4),
        operations_priority=round(ops, 4),
        curiosity=round(curiosity, 4),
        authority_deference=round(deference, 4),
        honesty=round(honesty, 4),
        source="mirofish",
    )


def _extract_profile_list(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("profiles", "agents", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Unsupported MiroFish profile JSON format")


def load_reddit_profiles_json(
    path: str | Path,
    limit: Optional[int] = None,
) -> List[SyntheticProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_profiles = list(_extract_profile_list(payload))
    if limit is not None:
        raw_profiles = raw_profiles[:limit]
    return [
        profile_from_mirofish(agent, player_id=index + 1)
        for index, agent in enumerate(raw_profiles)
    ]
