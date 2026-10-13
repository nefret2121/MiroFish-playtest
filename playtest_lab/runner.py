from __future__ import annotations

import argparse
import json
from pathlib import Path

from .case_engine import CaseEngine, load_case
from .metrics import markdown_report, summarize
from .profiles import generate_profiles


DEFAULT_CASE = Path(__file__).parent / "cases" / "construction_001.json"


def run(case_path: Path, agents: int, seed: int, out_dir: Path) -> dict:
    case_data = load_case(case_path)
    profiles = generate_profiles(agents, seed=seed)

    # Every player receives the exact same persisted case truth.
    results = [CaseEngine(case_data).play(profile) for profile in profiles]
    summary = summarize(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic playtests")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("playtest_lab/out"))
    args = parser.parse_args()

    summary = run(args.case, args.agents, args.seed, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
