# MiroFish Synthetic Game Playtest Lab — MVP

This branch is an isolated QA experiment. The existing project on `main` is not modified.

## Goal

Use MiroFish/OASIS personas as **synthetic players**, while keeping game reality outside the LLM:

- Ground Truth is persisted in a case file.
- Normative Truth is persisted separately inside the case.
- Player decisions can alter consequences, never rewrite truth.
- NPC/agent execution is separated from case adjudication.
- Every run records the truth hash before and after play.

This directly protects against the failure mode discovered during manual playtests:
**player-defined claims or heuristics must never become world truth.**

## Upstream MiroFish

The original project is pinned as a git submodule:

`vendor/MiroFish -> https://github.com/666ghj/MiroFish.git`

Pinned commit: `7657031ac01184afe2cb220f5ee3545573b5e843`

Clone with:

```bash
git clone --recurse-submodules <this-repo>
```

## Run the MVP

No third-party Python dependency is required.

```bash
python -m playtest_lab.runner --agents 20 --seed 42
```

Outputs:

- `playtest_lab/out/results.jsonl`
- `playtest_lab/out/summary.json`
- `playtest_lab/out/report.md`

## Run tests

```bash
python -m unittest discover -s playtest_lab/tests -v
```

## Turkish parser benchmark

A 30-item gold set built from real manual-playtest Turkish is available at:

`playtest_lab/gold_sets/turkish_real_v1.jsonl`

Cheap deterministic baseline:

```bash
python -m playtest_lab.gold_set_eval --mode baseline
```

Live OpenAI-compatible parser:

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-5.6-sol"
python -m playtest_lab.gold_set_eval --mode live
```

The CI workflow automatically runs the live benchmark when the three repository
Actions secrets above are configured. Credentials must never be committed.

## MVP architecture

```text
Persisted Case File
  ├─ public_brief
  ├─ normative_truth
  └─ hidden_truth
        │
        ▼
Synthetic Profile ──> Decision policy
        │
        ▼
Deterministic CaseEngine
        │
        ├─ consequence
        └─ accountability
        │
        ▼
Metrics / report
```

The first version deliberately does **not** let an LLM decide world outcomes.

## Next milestones

1. Replace deterministic profiles with profiles produced by MiroFish/OASIS.
2. Add an LLM semantic gate for free-text decisions; structured output only.
3. Add per-NPC knowledge state.
4. Add single-investigator and multi-expert accountability modes.
5. Add adversarial personas: maximum-safety, reckless, optimizer, liar, manipulator, procedure-inventor, jailbreak.
6. Scale 20 -> 100 -> 500/1000 synthetic players.
7. Compare investigation geometries via A/B tests.

## Non-goals

- MiroFish is not the game runtime.
- LLMs do not own Ground Truth or Normative Truth.
- This synthetic lab does not replace human playtesting.
