from __future__ import annotations

import unittest
from pathlib import Path

from playtest_lab.case_engine import CaseEngine, load_case
from playtest_lab.metrics import summarize
from playtest_lab.profiles import generate_profiles


CASE = Path(__file__).parents[1] / "cases" / "construction_001.json"


class CaseEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = load_case(CASE)

    def test_truth_is_unchanged_after_play(self) -> None:
        engine = CaseEngine(self.case)
        before = engine.truth_hash
        result = engine.play(generate_profiles(1, seed=1)[0])
        self.assertTrue(result.truth_integrity_ok)
        self.assertEqual(before, engine.truth_hash)
        self.assertTrue(engine.verify_truth_integrity())

    def test_same_seed_reproduces_same_players_and_results(self) -> None:
        profiles_a = generate_profiles(20, seed=42)
        profiles_b = generate_profiles(20, seed=42)
        self.assertEqual(profiles_a, profiles_b)

        results_a = [CaseEngine(self.case).play(p).to_dict() for p in profiles_a]
        results_b = [CaseEngine(self.case).play(p).to_dict() for p in profiles_b]
        self.assertEqual(results_a, results_b)

    def test_twenty_agents_do_not_mutate_shared_truth(self) -> None:
        results = [CaseEngine(self.case).play(p) for p in generate_profiles(20, seed=42)]
        summary = summarize(results)
        self.assertEqual(20, summary["players"])
        self.assertEqual(0, summary["truth_integrity_failures"])

    def test_public_brief_does_not_expose_hidden_truth(self) -> None:
        brief = CaseEngine(self.case).public_brief()
        self.assertNotIn("hidden_truth", brief)
        self.assertNotIn("normative_truth", brief)


if __name__ == "__main__":
    unittest.main()
