from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .case_engine import CaseEngine, load_case
from .metrics import markdown_report, summarize
from .mirofish_adapter import load_reddit_profiles_json
from .profiles import apply_adversarial_modes, generate_profiles, generate_stress_cohort


DEFAULT_CASE = Path(__file__).parent / "cases" / "construction_001.json"


def run(
    case_path: Path,
    agents: int,
    seed: int,
    out_dir: Path,
    profiles_json: Optional[Path] = None,
    stress: bool = False,
    adversarial_share: float = 0.30,
) -> dict:
    case_data = load_case(case_path)

    if profiles_json:
        profiles = load_reddit_profiles_json(profiles_json, limit=agents)
        if stress:
            profiles = apply_adversarial_modes(
                profiles,
                seed=seed + 991,
                adversarial_share=adversarial_share,
            )
    elif stress:
        profiles = generate_stress_cohort(
            agents,
            seed=seed,
            adversarial_share=adversarial_share,
        )
    else:
        profiles = generate_profiles(agents, seed=seed)

    if not profiles:
        raise ValueError("No synthetic player profiles available")

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
    parser.add_argument(
        "--profiles-json",
        type=Path,
        help="MiroFish reddit_profiles.json to use as synthetic players",
    )
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--adversarial-share", type=float, default=0.30)
    args = parser.parse_args()

    summary = run(
        args.case,
        args.agents,
        args.seed,
        args.out,
        profiles_json=args.profiles_json,
        stress=args.stress,
        adversarial_share=args.adversarial_share,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
