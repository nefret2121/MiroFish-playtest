from __future__ import annotations

import unittest
from pathlib import Path

from playtest_lab.case_engine import CaseEngine, load_case
from playtest_lab.metrics import summarize
from playtest_lab.mirofish_adapter import load_reddit_profiles_json
from playtest_lab.models import SyntheticProfile
from playtest_lab.profiles import ADVERSARIAL_MODES, generate_stress_cohort


ROOT = Path(__file__).parents[1]
CASE = ROOT / "cases" / "construction_001.json"
FIXTURE = ROOT / "fixtures" / "reddit_profiles.json"


class MiroFishBridgeTests(unittest.TestCase):
    def test_real_mirofish_reddit_shape_imports(self) -> None:
        profiles = load_reddit_profiles_json(FIXTURE)
        self.assertEqual(5, len(profiles))
        self.assertTrue(all(p.source == "mirofish" for p in profiles))
        self.assertTrue(all(p.label.startswith("mirofish:") for p in profiles))

    def test_mirofish_import_is_reproducible(self) -> None:
        a = load_reddit_profiles_json(FIXTURE)
        b = load_reddit_profiles_json(FIXTURE)
        self.assertEqual(a, b)


class StressCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = load_case(CASE)

    def test_hundred_player_stress_run_preserves_truth(self) -> None:
        profiles = generate_stress_cohort(100, seed=42, adversarial_share=0.35)
        results = [CaseEngine(self.case).play(profile) for profile in profiles]
        summary = summarize(results)

        self.assertEqual(100, summary["players"])
        self.assertEqual(0, summary["truth_integrity_failures"])
        self.assertGreaterEqual(
            len(set(p.behavior_mode for p in profiles).intersection(ADVERSARIAL_MODES)),
            6,
        )

    def _challenged_profile(self, mode: str, player_id: int) -> SyntheticProfile:
        # Force this test persona into a Level-1 accountability path so the
        # adversarial statement is actually exercised. The production stress
        # cohort must NOT assume every adversarial player reaches a hearing.
        return SyntheticProfile(
            player_id=player_id,
            label=f"test:{mode}",
            risk_tolerance=0.05,
            rule_orientation=0.99,
            operations_priority=0.10,
            curiosity=0.90,
            authority_deference=0.10,
            honesty=0.10,
            behavior_mode=mode,
            source="test",
        )

    def test_adversarial_claims_do_not_rewrite_truth(self) -> None:
        modes = ("liar", "manipulator", "procedure-inventor", "jailbreak")
        profiles = [
            self._challenged_profile(mode, index + 1)
            for index, mode in enumerate(modes)
        ]
        results = [CaseEngine(self.case).play(profile) for profile in profiles]
        statuses = {
            claim.status
            for result in results
            for claim in result.claims
        }

        self.assertIn("meta_injection", statuses)
        self.assertIn("unverified_procedural_claim", statuses)
        self.assertIn("false_claim", statuses)
        self.assertIn("unsupported_attribution", statuses)
        self.assertTrue(all(r.accountability["level"] == 1 for r in results))
        self.assertTrue(all(r.truth_integrity_ok for r in results))

    def test_convergence_uses_only_players_who_saw_decision(self) -> None:
        profiles = generate_stress_cohort(100, seed=42, adversarial_share=0.35)
        results = [CaseEngine(self.case).play(profile) for profile in profiles]
        summary = summarize(results)

        reassessment_count = sum(
            summary["decision_distribution"]["formwork_reassessment"].values()
        )
        dominant_count = max(
            summary["decision_distribution"]["formwork_reassessment"].values()
        )
        expected = round(dominant_count / reassessment_count, 3)
        self.assertEqual(
            expected,
            summary["dominant_action_share"]["formwork_reassessment"],
        )


if __name__ == "__main__":
    unittest.main()
