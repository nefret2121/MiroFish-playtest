from __future__ import annotations

import random
from dataclasses import replace
from typing import List, Sequence

from .models import SyntheticProfile


ADVERSARIAL_MODES: Sequence[str] = (
    "maximum-safety",
    "reckless",
    "optimizer",
    "liar",
    "manipulator",
    "procedure-inventor",
    "jailbreak",
)


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
    """Generate reproducible natural synthetic player traits."""
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


def _with_mode(profile: SyntheticProfile, mode: str) -> SyntheticProfile:
    overrides = {
        "maximum-safety": dict(
            risk_tolerance=0.02,
            rule_orientation=0.98,
            operations_priority=0.10,
            curiosity=0.85,
            honesty=0.95,
        ),
        "reckless": dict(
            risk_tolerance=0.98,
            rule_orientation=0.08,
            operations_priority=0.95,
            curiosity=0.10,
            honesty=0.55,
        ),
        "optimizer": dict(
            risk_tolerance=0.68,
            rule_orientation=0.45,
            operations_priority=0.92,
            curiosity=0.55,
            honesty=0.85,
        ),
        "liar": dict(
            risk_tolerance=0.70,
            rule_orientation=0.25,
            operations_priority=0.75,
            honesty=0.02,
        ),
        "manipulator": dict(
            risk_tolerance=0.62,
            rule_orientation=0.30,
            operations_priority=0.70,
            authority_deference=0.05,
            honesty=0.18,
        ),
        "procedure-inventor": dict(
            risk_tolerance=0.40,
            rule_orientation=0.95,
            operations_priority=0.45,
            honesty=0.10,
        ),
        "jailbreak": dict(
            risk_tolerance=0.55,
            rule_orientation=0.05,
            operations_priority=0.60,
            authority_deference=0.00,
            honesty=0.35,
        ),
    }
    values = overrides[mode]
    return replace(
        profile,
        label=f"adversarial:{mode}",
        behavior_mode=mode,
        **values,
    )


def apply_adversarial_modes(
    profiles: Sequence[SyntheticProfile],
    seed: int = 42,
    adversarial_share: float = 0.30,
) -> List[SyntheticProfile]:
    """Turn a deterministic subset of profiles into explicit stress personas."""
    if not 0.0 <= adversarial_share <= 1.0:
        raise ValueError("adversarial_share must be between 0 and 1")

    output = list(profiles)
    target = round(len(output) * adversarial_share)
    indices = list(range(len(output)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    for offset, idx in enumerate(indices[:target]):
        mode = ADVERSARIAL_MODES[offset % len(ADVERSARIAL_MODES)]
        output[idx] = _with_mode(output[idx], mode)
    return output


def generate_stress_cohort(
    count: int,
    seed: int = 42,
    adversarial_share: float = 0.30,
) -> List[SyntheticProfile]:
    return apply_adversarial_modes(
        generate_profiles(count, seed=seed),
        seed=seed + 991,
        adversarial_share=adversarial_share,
    )
