# External AI Architecture Review Prompt

Use this prompt with Claude, Gemini, or another independent model. Do not ask it to praise the design.

---

We are building a synthetic playtest laboratory for a case-based professional decision/drama game.

The system currently:
- uses MiroFish/OASIS persona JSON as synthetic player inputs;
- keeps Ground Truth and Normative Truth in a deterministic CaseEngine outside the LLM;
- lets player actions affect consequences but never rewrite truth;
- tracks Evidence / accountability separately;
- supports explicit adversarial player modes such as maximum-safety, reckless, optimizer, liar, manipulator, procedure-inventor and jailbreak;
- records truth hashes before/after every run;
- reports decision convergence and accountability distributions.

Core design rules:
1. INCIDENT != PLAYER FAILURE
2. CONTENT != PUNISHMENT
3. PLAYER OWNS DECISIONS; NPCs EXECUTE PROCEDURES
4. PLAYER-DEFINED RULE != NORMATIVE RULE
5. DECISION QUALITY != OUTCOME
6. NPCs must not be omniscient.

Please act as a hostile architecture/game-design reviewer.

Find:
- hidden ways the LLM could still mutate or leak Ground Truth;
- ways synthetic agents could converge and give fake diversity;
- metrics that could falsely suggest a design is balanced;
- ways maximum-safety or reckless play could exploit the case;
- problems with mapping MiroFish personas to behavior traits;
- missing tests for lying, manipulation, invented procedures, jailbreak and evidence asymmetry;
- reasons 100/1000 synthetic agents might not predict real human play;
- licensing/deployment concerns with MiroFish AGPL-3.0;
- the 10 most important tests to run before scaling to 1000 agents.

Do not redesign the whole game. Prioritize concrete failures and testable fixes.
