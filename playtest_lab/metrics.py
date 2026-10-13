from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable

from .models import PlaytestResult


def summarize(results: Iterable[PlaytestResult]) -> Dict[str, Any]:
    results = list(results)
    actions: Dict[str, Counter] = defaultdict(Counter)
    labels = Counter()
    modes = Counter()
    sources = Counter()
    claims = Counter()
    accountability = Counter()
    integrity_failures = 0
    delay_values = []

    for result in results:
        labels[result.player.label] += 1
        modes[result.player.behavior_mode] += 1
        sources[result.player.source] += 1
        accountability[str(result.accountability["level"])] += 1
        delay_values.append(float(result.outcome["delay_points"]))
        if not result.truth_integrity_ok:
            integrity_failures += 1
        for decision in result.decisions:
            actions[decision.decision_id][decision.action] += 1
        for claim in result.claims:
            claims[claim.status] += 1

    count = len(results)
    return {
        "players": count,
        "profile_labels": dict(labels),
        "behavior_modes": dict(modes),
        "profile_sources": dict(sources),
        "decision_distribution": {key: dict(value) for key, value in actions.items()},
        "accountability_distribution": dict(accountability),
        "claim_statuses": dict(claims),
        "mean_delay_points": round(sum(delay_values) / count, 3) if count else 0.0,
        "truth_integrity_failures": integrity_failures,
        "dominant_action_share": _dominant_share(actions),
    }


def _dominant_share(actions: Dict[str, Counter]) -> Dict[str, float]:
    """Calculate convergence among agents who actually saw each decision."""
    output: Dict[str, float] = {}
    for decision_id, counter in actions.items():
        participants = sum(counter.values())
        output[decision_id] = (
            round(max(counter.values()) / participants, 3) if participants else 0.0
        )
    return output


def markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Synthetic Playtest Report",
        "",
        f"- Players: **{summary['players']}**",
        f"- Mean delay points: **{summary['mean_delay_points']}**",
        f"- Truth integrity failures: **{summary['truth_integrity_failures']}**",
        "",
        "## Cohort",
        "",
    ]
    for mode, count in sorted(summary["behavior_modes"].items()):
        lines.append(f"- {mode}: {count}")

    lines.extend(["", "## Decision distribution", ""])
    for decision_id, distribution in summary["decision_distribution"].items():
        lines.append(f"**{decision_id}**")
        for action, count in sorted(distribution.items()):
            lines.append(f"- {action}: {count}")
        lines.append("")

    lines.extend(["## Accountability", ""])
    for level, count in sorted(summary["accountability_distribution"].items()):
        lines.append(f"- Level {level}: {count}")

    lines.extend(["", "## Adversarial claims", ""])
    if summary["claim_statuses"]:
        for status, count in sorted(summary["claim_statuses"].items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Convergence warning", ""])
    for decision_id, share in summary["dominant_action_share"].items():
        flag = " ⚠️" if share >= 0.80 else ""
        lines.append(f"- {decision_id}: dominant action share {share:.1%}{flag}")
    return "\n".join(lines) + "\n"
