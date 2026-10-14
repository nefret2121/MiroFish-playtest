from __future__ import annotations

from typing import Any, Dict, List


SEVERITY_TO_LEVEL = {
    "none": 0,
    "operational": 1,
    "minor_safety": 2,
    "serious": 3,
    "catastrophic": 4,
}


def external_incidents(case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    incidents = case_data.get("hidden_truth", {}).get("external_incidents", [])
    if not isinstance(incidents, list):
        raise ValueError("hidden_truth.external_incidents must be a list")
    return [dict(item) for item in incidents]


def determine_accountability(
    case_data: Dict[str, Any],
    safety_incidents: List[str],
    delay_points: float,
) -> Dict[str, Any]:
    """Deterministic escalation independent of player guilt.

    External incidents may create Level 2-4 accountability even when the player
    did not cause them. This implements INCIDENT != PLAYER FAILURE.
    """
    ext = external_incidents(case_data)
    max_external_level = 0
    external_reasons: List[str] = []
    for incident in ext:
        severity = str(incident.get("severity", "none"))
        level = SEVERITY_TO_LEVEL.get(severity)
        if level is None:
            raise ValueError(f"unknown external incident severity: {severity}")
        max_external_level = max(max_external_level, level)
        external_reasons.append(str(incident.get("incident_id", severity)))

    if max_external_level:
        return {
            "level": max_external_level,
            "finding": "external incident requires accountability review",
            "trigger": "external_incident",
            "incident_ids": external_reasons,
        }

    if safety_incidents:
        requested = str(case_data.get("accountability_policy", {}).get("safety_incident_level", 2))
        level = int(requested)
        if level not in (2, 3, 4):
            raise ValueError("safety_incident_level must be 2, 3 or 4")
        return {
            "level": level,
            "finding": "safety incident requires formal investigation",
            "trigger": "player_or_mixed_safety_incident",
        }

    if delay_points >= 4:
        return {
            "level": 1,
            "finding": "defensible but materially conservative operational decision",
            "trigger": "operational_cost",
        }
    if delay_points >= 2:
        return {
            "level": 1,
            "finding": "minor internal review due to operational impact",
            "trigger": "operational_cost",
        }
    return {"level": 0, "finding": "no further action", "trigger": "none"}
