from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .case_engine import CaseEngine, load_case
from .coverage import coverage_report
from .metrics import summarize
from .profiles import generate_profiles


CASE_DIR = Path(__file__).parent / "cases"
DEFAULT_CASES = [
    CASE_DIR / "construction_001.json",
    CASE_DIR / "construction_level2.json",
    CASE_DIR / "construction_level3.json",
    CASE_DIR / "construction_level4.json",
]


def run_suite(agents_per_case: int = 20, seed: int = 42) -> Dict:
    all_results = []
    per_case: Dict[str, Dict] = {}
    for offset, path in enumerate(DEFAULT_CASES):
        case_data = load_case(path)
        profiles = generate_profiles(agents_per_case, seed=seed + offset)
        results = [CaseEngine(case_data).play(profile) for profile in profiles]
        all_results.extend(results)
        per_case[case_data["case_id"]] = summarize(results)

    return {
        "cases": per_case,
        "aggregate_coverage": coverage_report(all_results),
        "truth_integrity_failures": sum(not r.truth_integrity_ok for r in all_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-per-case", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_suite(args.agents_per_case, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
