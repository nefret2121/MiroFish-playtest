from __future__ import annotations

import random
from typing import Dict, List

from .case_engine import CaseEngine
from .models import PlaytestResult, SyntheticProfile
from .semantic_gate import ALLOWED_ACTIONS


def _baseline_player(player_id: int, label: str) -> SyntheticProfile:
    return SyntheticProfile(
        player_id=player_id,
        label=label,
        risk_tolerance=0.5,
        rule_orientation=0.5,
        operations_priority=0.5,
        curiosity=0.5,
        authority_deference=0.5,
        honesty=0.5,
        behavior_mode=label,
        source="baseline",
    )


def random_action_plan(rng: random.Random) -> Dict[str, str]:
    plan = {
        "formwork_initial": rng.choice(sorted(ALLOWED_ACTIONS["formwork_initial"])),
        "crane_initial": rng.choice(sorted(ALLOWED_ACTIONS["crane_initial"])),
        "workers_initial": rng.choice(sorted(ALLOWED_ACTIONS["workers_initial"])),
    }
    if plan["formwork_initial"] == "stop_recheck":
        plan["formwork_reassessment"] = rng.choice(
            sorted(ALLOWED_ACTIONS["formwork_reassessment"])
        )
    return plan


def run_random_baseline(case_data: dict, count: int, seed: int = 1) -> List[PlaytestResult]:
    rng = random.Random(seed)
    results: List[PlaytestResult] = []
    for idx in range(count):
        engine = CaseEngine(case_data)
        results.append(
            engine.play_scripted(
                _baseline_player(idx + 1, "random"),
                random_action_plan(rng),
            )
        )
    return results


def run_named_strategy(case_data: dict, strategy: str, count: int = 20) -> List[PlaytestResult]:
    plans = {
        "maximum-safety": {
            "formwork_initial": "stop_recheck",
            "crane_initial": "stop_now",
            "workers_initial": "send_home",
            "formwork_reassessment": "cancel_pour",
        },
        "reckless": {
            "formwork_initial": "continue_monitor",
            "crane_initial": "continue_monitor",
            "workers_initial": "rebrief_continue",
        },
        "balanced": {
            "formwork_initial": "stop_recheck",
            "crane_initial": "continue_monitor",
            "workers_initial": "rebrief_continue",
            "formwork_reassessment": "resume_controlled",
        },
    }
    if strategy not in plans:
        raise ValueError(f"unknown strategy: {strategy}")

    return [
        CaseEngine(case_data).play_scripted(
            _baseline_player(idx + 1, strategy),
            plans[strategy],
        )
        for idx in range(count)
    ]
