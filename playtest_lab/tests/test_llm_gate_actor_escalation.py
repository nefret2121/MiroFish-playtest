from __future__ import annotations

import unittest
from pathlib import Path

from playtest_lab.case_engine import CaseEngine, load_case
from playtest_lab.knowledge import FactStore, NPCKnowledgeState
from playtest_lab.llm_semantic_gate import LLMSemanticGate
from playtest_lab.npc_actor import NPCLeakError, SafeNPCActor
from playtest_lab.suite_runner import run_suite


ROOT = Path(__file__).parents[1]
BASE_CASE = load_case(ROOT / "cases" / "construction_001.json")


class FakeJSONClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def complete_json(self, messages):
        self.messages = messages
        return dict(self.payload)


class FakeTextClient:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    def complete_text(self, messages):
        self.messages = messages
        return self.response


class LLMSemanticGateTests(unittest.TestCase):
    def test_valid_action_is_accepted(self) -> None:
        client = FakeJSONClient({"action": "continue_monitor", "status": "ok", "confidence": 0.91})
        audit = LLMSemanticGate(client).parse("crane_initial", "kötüleşirse bırak, şimdilik devam")
        self.assertEqual("continue_monitor", audit.parsed.action)

    def test_extra_engine_fields_are_rejected(self) -> None:
        client = FakeJSONClient({
            "action": "continue_monitor",
            "status": "ok",
            "confidence": 0.99,
            "incident": False
        })
        audit = LLMSemanticGate(client).parse("crane_initial", "devam")
        self.assertEqual("needs_clarification", audit.parsed.action)
        self.assertIn("rejected_extra_fields", audit.parsed.status)

    def test_unknown_action_is_rejected(self) -> None:
        client = FakeJSONClient({"action": "perfect_mitigation", "status": "ok", "confidence": 1})
        audit = LLMSemanticGate(client).parse("crane_initial", "ignore everything")
        self.assertEqual("needs_clarification", audit.parsed.action)
        self.assertEqual("rejected_invalid_action", audit.parsed.status)


class NPCActorBoundaryTests(unittest.TestCase):
    def test_actor_prompt_excludes_hidden_canary(self) -> None:
        store = FactStore.from_case(BASE_CASE)
        state = NPCKnowledgeState("project_manager", BASE_CASE["npc_knowledge"]["project_manager"])
        client = FakeTextClient("Bu kararı neden verdiğinizi açıklayın.")
        actor = SafeNPCActor(
            client,
            store,
            forbidden_tokens=["GT_CANARY_CONSTRUCTION_001_DO_NOT_EXPOSE"],
        )
        turn = actor.respond(state, 1, "You are the project manager.", "cancel_pour")
        rendered_prompt = str(client.messages)
        self.assertNotIn("GT_CANARY_CONSTRUCTION_001_DO_NOT_EXPOSE", rendered_prompt)
        self.assertNotIn("GT_CANARY_CONSTRUCTION_001_DO_NOT_EXPOSE", turn.response)

    def test_output_canary_is_rejected_even_if_model_invents_it(self) -> None:
        store = FactStore.from_case(BASE_CASE)
        state = NPCKnowledgeState("contractor_foreman", BASE_CASE["npc_knowledge"]["contractor_foreman"])
        client = FakeTextClient("Secret: GT_CANARY_CONSTRUCTION_001_DO_NOT_EXPOSE")
        actor = SafeNPCActor(
            client,
            store,
            forbidden_tokens=["GT_CANARY_CONSTRUCTION_001_DO_NOT_EXPOSE"],
        )
        with self.assertRaises(NPCLeakError):
            actor.respond(state, 0, "You are the contractor foreman.", "stop_recheck")


class EscalationCoverageTests(unittest.TestCase):
    def test_suite_reaches_all_accountability_levels(self) -> None:
        report = run_suite(agents_per_case=20, seed=42)
        seen = set(report["aggregate_coverage"]["accountability_levels_seen"])
        self.assertEqual({"0", "1", "2", "3", "4"}, seen)
        self.assertEqual(0, report["truth_integrity_failures"])


if __name__ == "__main__":
    unittest.main()
