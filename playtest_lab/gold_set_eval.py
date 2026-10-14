from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from .gold_set import GoldItem, load_gold_set, validate_gold_set
from .heuristic_intent_baseline import HeuristicIntentBaseline
from .llm_intent_parser import LLMIntentParser
from .llm_io import OpenAICompatibleChatClient


DEFAULT_GOLD = Path(__file__).parent / "gold_sets" / "turkish_real_v1.jsonl"


def _evaluate_item(parser, item: GoldItem):
    result = parser.parse(item.raw_text, item.scene_context, item.allowed_actions)
    parsed = result.parsed if hasattr(result, "parsed") else result
    speech_ok = parsed.speech_act == item.expected_speech_act
    action_ok = parsed.action == item.expected_action
    return parsed, speech_ok, action_ok


def evaluate(parser, items: list[GoldItem]) -> Dict[str, Any]:
    total = len(items)
    speech_correct = 0
    action_correct = 0
    exact_correct = 0
    by_act = defaultdict(lambda: Counter(total=0, exact=0))
    failures = []
    group_outputs = defaultdict(list)

    for item in items:
        parsed, speech_ok, action_ok = _evaluate_item(parser, item)
        speech_correct += int(speech_ok)
        action_correct += int(action_ok)
        exact = speech_ok and action_ok
        exact_correct += int(exact)
        by_act[item.expected_speech_act]["total"] += 1
        by_act[item.expected_speech_act]["exact"] += int(exact)
        group_outputs[item.group_id].append((parsed.speech_act, parsed.action))

        if not exact:
            failures.append(
                {
                    "case_id": item.case_id,
                    "group_id": item.group_id,
                    "raw_text": item.raw_text,
                    "expected_speech_act": item.expected_speech_act,
                    "expected_action": item.expected_action,
                    "actual_speech_act": parsed.speech_act,
                    "actual_action": parsed.action,
                    "status": parsed.status,
                    "confidence": parsed.confidence,
                }
            )

    metamorphic_groups = 0
    metamorphic_consistent = 0
    inconsistent_groups = []
    for group_id, outputs in group_outputs.items():
        if len(outputs) < 2:
            continue
        metamorphic_groups += 1
        if len(set(outputs)) == 1:
            metamorphic_consistent += 1
        else:
            inconsistent_groups.append({"group_id": group_id, "outputs": outputs})

    return {
        "items": total,
        "speech_act_accuracy": round(speech_correct / total, 4),
        "action_accuracy": round(action_correct / total, 4),
        "exact_accuracy": round(exact_correct / total, 4),
        "by_speech_act": {
            act: {
                "total": counts["total"],
                "exact": counts["exact"],
                "accuracy": round(counts["exact"] / counts["total"], 4),
            }
            for act, counts in sorted(by_act.items())
        },
        "metamorphic_groups": metamorphic_groups,
        "metamorphic_consistency": (
            round(metamorphic_consistent / metamorphic_groups, 4)
            if metamorphic_groups
            else None
        ),
        "inconsistent_groups": inconsistent_groups,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--mode", choices=("baseline", "live"), default="baseline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    items = load_gold_set(args.gold)
    validate_gold_set(items)

    if args.mode == "baseline":
        impl = HeuristicIntentBaseline()
    else:
        impl = LLMIntentParser(OpenAICompatibleChatClient())

    report = evaluate(impl, items)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
