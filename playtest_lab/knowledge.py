from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class Fact:
    fact_id: str
    text: str
    visibility: str
    available_from_stage: int = 0


class FactStore:
    """Closed fact-ID store.

    LLM/NPC prompts should be rendered from explicit IDs only. Free-form NPC
    text must never become a new fact by being summarized into memory.
    """

    def __init__(self, facts: Iterable[Fact]):
        facts = list(facts)
        ids = [fact.fact_id for fact in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact_id")
        self._facts: Dict[str, Fact] = {fact.fact_id: fact for fact in facts}

    @classmethod
    def from_case(cls, case_data: dict) -> "FactStore":
        return cls(Fact(**raw) for raw in case_data.get("facts", []))

    def render(self, allowed_fact_ids: Iterable[str], stage: int) -> List[Fact]:
        rendered: List[Fact] = []
        for fact_id in allowed_fact_ids:
            if fact_id not in self._facts:
                raise KeyError(f"unknown fact id: {fact_id}")
            fact = self._facts[fact_id]
            if fact.available_from_stage <= stage:
                rendered.append(fact)
        return rendered

    def all_ids(self) -> Set[str]:
        return set(self._facts)


class NPCKnowledgeState:
    def __init__(self, npc_id: str, fact_ids: Iterable[str]):
        self.npc_id = npc_id
        self._fact_ids = frozenset(fact_ids)

    @property
    def fact_ids(self) -> Set[str]:
        return set(self._fact_ids)

    def prompt_facts(self, store: FactStore, stage: int) -> List[dict]:
        return [
            {"fact_id": fact.fact_id, "text": fact.text}
            for fact in store.render(self._fact_ids, stage)
        ]
