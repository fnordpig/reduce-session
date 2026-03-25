from enum import Enum
from typing import Protocol


class Category(str, Enum):
    # KEEP — user intent, preserve fully
    INSTRUCTION = "INSTRUCTION"
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    INQUIRY = "INQUIRY"
    DECISION = "DECISION"
    FEEDBACK = "FEEDBACK"

    # DISTILL — compress with type-specific prompts
    EXPLANATION = "EXPLANATION"
    IMPLEMENTATION = "IMPLEMENTATION"
    REASONING = "REASONING"
    DEBUGGING = "DEBUGGING"
    METRICS = "METRICS"
    COMPILATION = "COMPILATION"
    PLANNING = "PLANNING"
    TESTING = "TESTING"
    GIT_OPERATION = "GIT_OPERATION"
    ANALYSIS = "ANALYSIS"

    # HEURISTIC — existing pipeline
    STATUS_UPDATE = "STATUS_UPDATE"
    NOTIFICATION = "NOTIFICATION"
    LOG_OUTPUT = "LOG_OUTPUT"
    SCAFFOLDING = "SCAFFOLDING"
    ERROR_OUTPUT = "ERROR_OUTPUT"


class Route(str, Enum):
    KEEP = "KEEP"
    DISTILL = "DISTILL"
    HEURISTIC = "HEURISTIC"


ROUTING_MAP: dict[Category, Route] = {
    # KEEP
    Category.INSTRUCTION: Route.KEEP,
    Category.CLARIFICATION: Route.KEEP,
    Category.CONFIRMATION: Route.KEEP,
    Category.INQUIRY: Route.KEEP,
    Category.DECISION: Route.KEEP,
    Category.FEEDBACK: Route.KEEP,
    # DISTILL
    Category.EXPLANATION: Route.DISTILL,
    Category.IMPLEMENTATION: Route.DISTILL,
    Category.REASONING: Route.DISTILL,
    Category.DEBUGGING: Route.DISTILL,
    Category.METRICS: Route.DISTILL,
    Category.COMPILATION: Route.DISTILL,
    Category.PLANNING: Route.DISTILL,
    Category.TESTING: Route.DISTILL,
    Category.GIT_OPERATION: Route.DISTILL,
    Category.ANALYSIS: Route.DISTILL,
    # HEURISTIC
    Category.STATUS_UPDATE: Route.HEURISTIC,
    Category.NOTIFICATION: Route.HEURISTIC,
    Category.LOG_OUTPUT: Route.HEURISTIC,
    Category.SCAFFOLDING: Route.HEURISTIC,
    Category.ERROR_OUTPUT: Route.HEURISTIC,
}

KEEP_CATEGORIES = frozenset(c for c, r in ROUTING_MAP.items() if r == Route.KEEP)
DISTILL_CATEGORIES = frozenset(c for c, r in ROUTING_MAP.items() if r == Route.DISTILL)
HEURISTIC_CATEGORIES = frozenset(
    c for c, r in ROUTING_MAP.items() if r == Route.HEURISTIC
)


class LLMProvider(Protocol):
    async def classify(self, exchanges: list[dict]) -> list[Category]: ...
    async def distill(
        self, text: str, mode: str, category: str | None = None
    ) -> str: ...
    async def shutdown(self) -> None: ...
