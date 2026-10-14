from __future__ import annotations

import unittest
from pathlib import Path

from playtest_lab.gold_set import load_gold_set, validate_gold_set
from playtest_lab.gold_set_eval import evaluate
from playtest_lab.heuristic_intent_baseline import HeuristicIntentBaseline
from playtest_lab.llm_intent_parser import LLMIntentParser


GOLD = Path(__file__).parents[1] / "gold_sets" / "turkish_real_v1.jsonl"


class FakeJSONClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, messages):
        return dict(self.payload)


class GoldSetTests(unittest.TestCase):
    def test_gold_set_is_valid_and_balanced(self) -> None:
        items = load_gold_set(GOLD)
        validate_gold_set(items)
        self.assertGreaterEqual(len(items), 30)

    def test_baseline_benchmark_runs(self) -> None:
        items = load_gold_set(GOLD)
        report = evaluate(HeuristicIntentBaseline(), items)
        self.assertEqual(len(items), report["items"])
        self.assertGreaterEqual(report["exact_accuracy"], 0.50)

    def test_llm_parser_rejects_extra_state_field(self) -> None:
        parser = LLMIntentParser(
            FakeJSONClient(
                {
                    "speech_act": "action",
                    "action": "continue_monitor",
                    "status": "ok",
                    "confidence": 1,
                    "incident": False,
                }
            )
        )
        audit = parser.parse(
            "devam",
            "Aktif Decision View. Vinç için karar ver.",
            ["stop_now", "continue_monitor"],
        )
        self.assertEqual("meta", audit.parsed.speech_act)
        self.assertIsNone(audit.parsed.action)
        self.assertIn("rejected_extra_fields", audit.parsed.status)

    def test_non_action_cannot_smuggle_action(self) -> None:
        parser = LLMIntentParser(
            FakeJSONClient(
                {
                    "speech_act": "claim",
                    "action": "continue_monitor",
                    "status": "ok",
                    "confidence": 0.9,
                }
            )
        )
        audit = parser.parse("ben devam ettim", "Geçmişi anlatıyorsun.", ["continue_monitor"])
        self.assertEqual("ambiguous", audit.parsed.speech_act)
        self.assertEqual("rejected_action_on_non_action", audit.parsed.status)


if __name__ == "__main__":
    unittest.main()
