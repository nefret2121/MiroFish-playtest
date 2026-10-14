from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from .gold_set import load_gold_set
from .gold_set_eval import evaluate
from .llm_intent_parser import LLMIntentParser
from .llm_io import OpenAICompatibleChatClient


DEFAULT_GOLD = Path(__file__).parent / "gold_sets" / "turkish_real_holdout_v1.jsonl"


def run_repeated(gold_path: Path, repeats: int) -> Dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be >= 2")

    items = load_gold_set(gold_path)
    parser = LLMIntentParser(OpenAICompatibleChatClient())
    run_reports = []
    outputs = defaultdict(list)

    for _ in range(repeats):
        report = evaluate(parser, items)
        run_reports.append(report)
        for item in items:
            audit = parser.parse(item.raw_text, item.scene_context, item.allowed_actions)
            outputs[item.case_id].append(
                {
                    "speech_act": audit.parsed.speech_act,
                    "action": audit.parsed.action,
                    "status": audit.parsed.status,
                }
            )

    stable_items = 0
    per_item = {}
    for item in items:
        observed = [(x["speech_act"], x["action"]) for x in outputs[item.case_id]]
        counter = Counter(observed)
        stable = len(counter) == 1
        stable_items += int(stable)
        per_item[item.case_id] = {
            "stable": stable,
            "modal_share": round(max(counter.values()) / repeats, 4),
            "outputs": outputs[item.case_id],
        }

    exact_scores = [r["exact_accuracy"] for r in run_reports]
    action_scores = [r["action_accuracy"] for r in run_reports]
    return {
        "items": len(items),
        "repeats": repeats,
        "exact_accuracy_runs": exact_scores,
        "exact_accuracy_mean": round(sum(exact_scores) / repeats, 4),
        "exact_accuracy_min": min(exact_scores),
        "exact_accuracy_max": max(exact_scores),
        "action_accuracy_runs": action_scores,
        "action_accuracy_mean": round(sum(action_scores) / repeats, 4),
        "fully_stable_items": stable_items,
        "item_stability_rate": round(stable_items / len(items), 4),
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_repeated(args.gold, args.repeats)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
