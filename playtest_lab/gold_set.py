from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class GoldItem:
    case_id: str
    group_id: str
    scene_context: str
    raw_text: str
    allowed_actions: tuple[str, ...]
    expected_speech_act: str
    expected_action: Optional[str]

    @classmethod
    def from_dict(cls, raw: dict) -> "GoldItem":
        return cls(
            case_id=str(raw["case_id"]),
            group_id=str(raw.get("group_id", raw["case_id"])),
            scene_context=str(raw["scene_context"]),
            raw_text=str(raw["raw_text"]),
            allowed_actions=tuple(str(x) for x in raw.get("allowed_actions", [])),
            expected_speech_act=str(raw["expected_speech_act"]),
            expected_action=(
                None if raw.get("expected_action") is None else str(raw["expected_action"])
            ),
        )


def load_gold_set(path: str | Path) -> List[GoldItem]:
    items: List[GoldItem] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}") from exc
        items.append(GoldItem.from_dict(raw))
    return items


def validate_gold_set(items: Sequence[GoldItem]) -> None:
    if not items:
        raise ValueError("gold set is empty")
    ids = [item.case_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate gold-set case_id")
    categories = {item.expected_speech_act for item in items}
    required = {"action", "compound_action", "claim", "question", "hypothetical", "meta", "ambiguous"}
    missing = required - categories
    if missing:
        raise ValueError(f"missing speech-act categories: {sorted(missing)}")
