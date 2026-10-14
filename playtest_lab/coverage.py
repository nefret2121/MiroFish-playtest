from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Set

from .models import PlaytestResult


EXPECTED_DECISIONS = {
    "formwork_initial",
    "crane_initial",
    "workers_initial",
    "formwork_reassessment",
}
EXPECTED_ACCOUNTABILITY_LEVELS = {"0", "1", "2", "3", "4"}


def coverage_report(results: Iterable[PlaytestResult]) -> Dict[str, Any]:
    decision_seen = Counter()
    action_edges = defaultdict(set)
    levels: Set[str] = set()
    claim_statuses: Set[str] = set()

    for result in results:
        levels.add(str(result.accountability["level"]))
        for decision in result.decisions:
            decision_seen[decision.decision_id] += 1
            action_edges[decision.decision_id].add(decision.action)
        for claim in result.claims:
            claim_statuses.add(claim.status)

    seen = set(decision_seen)
    return {
        "decision_nodes_seen": sorted(seen),
        "decision_nodes_missing": sorted(EXPECTED_DECISIONS - seen),
        "decision_seen_count": dict(decision_seen),
        "action_edges_seen": {key: sorted(value) for key, value in action_edges.items()},
        "accountability_levels_seen": sorted(levels),
        "accountability_levels_missing": sorted(EXPECTED_ACCOUNTABILITY_LEVELS - levels),
        "claim_statuses_seen": sorted(claim_statuses),
    }
