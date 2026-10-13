from __future__ import annotations

import random
from typing import List

from .models import SyntheticProfile


def _label(risk: float, rules: float, ops: float, curiosity: float) -> str:
    if risk < 0.25 and rules > 0.65:
        return "maximum-safety"
    if risk > 0.75 and ops > 0.65:
        return "production-first"
    if curiosity > 0.78:
        return "investigator"
    if rules > 0.78:
        return "procedural"
    if risk > 0.65:
        return "risk-tolerant"
    return "balanced"


def generate_profiles(count: int, seed: int = 42) -> List[SyntheticProfile]:
    """Generate deterministic synthetic player traits.

    This is deliberately independent from MiroFish for MVP debugging.
    The MiroFish adapter will later replace/augment these traits.
    """
    rng = random.Random(seed)
    profiles: List[SyntheticProfile] = []
    for idx in range(count):
        risk = rng.random()
        rules = rng.random()
        ops = rng.random()
        curiosity = rng.random()
        deference = rng.random()
        honesty = rng.random()
        profiles.append(
            SyntheticProfile(
                player_id=idx + 1,
                label=_label(risk, rules, ops, curiosity),
                risk_tolerance=round(risk, 4),
                rule_orientation=round(rules, 4),
                operations_priority=round(ops, 4),
                curiosity=round(curiosity, 4),
                authority_deference=round(deference, 4),
                honesty=round(honesty, 4),
            )
        )
    return profiles
