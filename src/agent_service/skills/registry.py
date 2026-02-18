from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


_SKILLS: Dict[str, SkillSpec] = {}


def register(skill: SkillSpec) -> None:
    if skill.name in _SKILLS:
        raise ValueError(f"duplicate skill: {skill.name}")
    _SKILLS[skill.name] = skill


def list_skill_specs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in sorted(_SKILLS.values(), key=lambda x: x.name):
        out.append(
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters_schema,
            }
        )
    return out


def invoke_skill(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if name not in _SKILLS:
        raise KeyError(f"unknown skill: {name}")
    a = dict(args or {})
    return _SKILLS[name].handler(a)
