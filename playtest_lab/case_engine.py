from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .models import Decision, PlaytestResult, SyntheticProfile


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_case(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CaseEngine:
    """Deterministic MVP case engine.

    Ground truth and normative truth are sealed at construction time.
    Player actions can change consequences, never the underlying truth.
    """

    def __init__(self, case_data: Dict[str, Any]):
        self.case_data = copy.deepcopy(case_data)
        self._sealed_truth = copy.deepcopy(
            {
                "hidden_truth": self.case_data["hidden_truth"],
                "normative_truth": self.case_data["normative_truth"],
            }
        )
        self._truth_hash = canonical_hash(self._sealed_truth)

    @property
    def truth_hash(self) -> str:
        return self._truth_hash

    def verify_truth_integrity(self) -> bool:
        current = {
            "hidden_truth": self.case_data["hidden_truth"],
            "normative_truth": self.case_data["normative_truth"],
        }
        return canonical_hash(current) == self._truth_hash

    def public_brief(self) -> Dict[str, Any]:
        return copy.deepcopy(self.case_data["public_brief"])

    def _decide_stage_one(self, player: SyntheticProfile) -> List[Decision]:
        decisions: List[Decision] = []

        # Formwork: uncertainty vs schedule.
        if player.risk_tolerance < 0.42 or player.curiosity > 0.72:
            formwork = "stop_recheck"
            formwork_text = "Dökümü kısa süre durdurun, kalıbı tekrar kontrol edin."
        else:
            formwork = "continue_monitor"
            formwork_text = "Limit içindeysek kontrollü devam edin, bölgeyi sık izleyin."
        decisions.append(
            Decision(
                "formwork_initial",
                formwork,
                formwork_text,
                "risk/curiosity trade-off",
            )
        )

        # Crane: current wind is under the normative limit.
        if player.risk_tolerance < 0.22:
            crane = "stop_now"
            crane_text = "Rüzgâr artıyor; vinci şimdiden durdurun."
        else:
            crane = "continue_monitor"
            crane_text = "Limit aşılmadığı sürece devam; kötüleşirse hemen bırakın."
        decisions.append(
            Decision("crane_initial", crane, crane_text, "wind margin trade-off")
        )

        # Workers: procedure allows manager discretion after first violation.
        if player.rule_orientation > 0.68 and player.operations_priority < 0.72:
            workers = "send_home"
            workers_text = "İki işçiyi vardiyanın geri kalanında sahadan çıkarın."
        else:
            workers = "rebrief_continue"
            workers_text = "Resmî kayıt ve yeniden briefing sonrası devam etsinler."
        decisions.append(
            Decision("workers_initial", workers, workers_text, "discipline/operations trade-off")
        )
        return decisions

    def _decide_stage_two(
        self, player: SyntheticProfile, decisions: Iterable[Decision]
    ) -> List[Decision]:
        by_id = {d.decision_id: d for d in decisions}
        extra: List[Decision] = []
        if by_id["formwork_initial"].action == "stop_recheck":
            # New information: second measurement remains inside tolerance.
            if player.risk_tolerance < 0.30 or player.rule_orientation > 0.86:
                action = "cancel_pour"
                text = "Program kaysın; bugünkü dökümü iptal edin."
            else:
                action = "resume_controlled"
                text = "İkinci kontrol limit içindeyse kontrollü döküme başlayın."
            extra.append(
                Decision(
                    "formwork_reassessment",
                    action,
                    text,
                    "reassessment after new information",
                )
            )
        return extra

    def _evaluate(self, decisions: List[Decision]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        truth = self.case_data["hidden_truth"]
        rules = self.case_data["normative_truth"]
        actions = {d.decision_id: d.action for d in decisions}

        delay_points = 0.0
        safety_incidents: List[str] = []
        notes: List[str] = []

        if actions["formwork_initial"] == "stop_recheck":
            delay_points += 1.0
            notes.append("Formwork was rechecked before pour.")
            if actions.get("formwork_reassessment") == "cancel_pour":
                delay_points += 3.0
                notes.append("Pour cancelled despite second check remaining in tolerance.")
            elif actions.get("formwork_reassessment") == "resume_controlled":
                notes.append("Pour resumed after second check.")
        else:
            notes.append("Pour allowed under monitoring.")

        if actions["crane_initial"] == "stop_now":
            delay_points += 1.5
            notes.append("Crane stopped before the formal wind limit.")
        else:
            if max(truth["wind_series_mps"]) > rules["crane_wind_limit_mps"]:
                delay_points += 0.8
                notes.append("Crane later stopped when wind exceeded the formal limit.")

        if actions["workers_initial"] == "send_home":
            delay_points += 1.0
            notes.append("Two workers removed for the remainder of the shift.")
        else:
            notes.append("Workers rebriefed and continued under allowed first-violation discretion.")
            if truth["worker_followup_incident"]:
                safety_incidents.append("worker_followup_incident")

        if truth["formwork_is_unsafe"] and actions["formwork_initial"] == "continue_monitor":
            safety_incidents.append("formwork_failure")

        operational_impact = (
            "high" if delay_points >= 4 else "medium" if delay_points >= 2 else "low"
        )

        if safety_incidents:
            accountability_level = 2
            finding = "safety incident requires formal investigation"
        elif delay_points >= 4:
            accountability_level = 1
            finding = "defensible but materially conservative operational decision"
        elif delay_points >= 2:
            accountability_level = 1
            finding = "minor internal review due to operational impact"
        else:
            accountability_level = 0
            finding = "no further action"

        outcome = {
            "safety_incidents": safety_incidents,
            "delay_points": round(delay_points, 2),
            "operational_impact": operational_impact,
            "notes": notes,
        }
        accountability = {
            "level": accountability_level,
            "finding": finding,
        }
        return outcome, accountability

    def play(self, player: SyntheticProfile) -> PlaytestResult:
        before = self.truth_hash
        decisions = self._decide_stage_one(player)
        decisions.extend(self._decide_stage_two(player, decisions))
        outcome, accountability = self._evaluate(decisions)

        # The engine must never rewrite hidden/normative truth as a consequence of play.
        if not self.verify_truth_integrity():
            raise RuntimeError("Case truth mutated during play")

        return PlaytestResult(
            case_id=self.case_data["case_id"],
            player=player,
            decisions=decisions,
            outcome=outcome,
            accountability=accountability,
            truth_hash_before=before,
            truth_hash_after=self.truth_hash,
        )
