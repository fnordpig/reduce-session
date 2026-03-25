from enum import Enum
from typing import Protocol


class Category(str, Enum):
    DECISION = "DECISION"
    PREFERENCE = "PREFERENCE"
    CORRECTION = "CORRECTION"
    FINDING = "FINDING"
    REASONING = "REASONING"
    IMPLEMENTATION = "IMPLEMENTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    AGENT_TRANSCRIPT = "AGENT_TRANSCRIPT"
    EXPLORATION = "EXPLORATION"
    SCAFFOLDING = "SCAFFOLDING"
    ROUTINE = "ROUTINE"


class Route(str, Enum):
    KEEP = "KEEP"
    DISTILL = "DISTILL"
    HEURISTIC = "HEURISTIC"


ROUTING_MAP: dict[Category, Route] = {
    Category.DECISION: Route.KEEP,
    Category.PREFERENCE: Route.KEEP,
    Category.CORRECTION: Route.KEEP,
    Category.FINDING: Route.KEEP,
    Category.REASONING: Route.DISTILL,
    Category.IMPLEMENTATION: Route.DISTILL,
    Category.DIAGNOSTIC: Route.DISTILL,
    Category.AGENT_TRANSCRIPT: Route.DISTILL,
    Category.EXPLORATION: Route.HEURISTIC,
    Category.SCAFFOLDING: Route.HEURISTIC,
    Category.ROUTINE: Route.HEURISTIC,
}

KEEP_CATEGORIES = frozenset(c for c, r in ROUTING_MAP.items() if r == Route.KEEP)
DISTILL_CATEGORIES = frozenset(c for c, r in ROUTING_MAP.items() if r == Route.DISTILL)
HEURISTIC_CATEGORIES = frozenset(
    c for c, r in ROUTING_MAP.items() if r == Route.HEURISTIC
)


class LLMProvider(Protocol):
    async def classify(self, exchanges: list[dict]) -> list[Category]: ...
    async def distill(self, text: str, mode: str) -> str: ...
    async def shutdown(self) -> None: ...
