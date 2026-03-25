from reduce_session.llm import (
    Category,
    Route,
    ROUTING_MAP,
    KEEP_CATEGORIES,
    DISTILL_CATEGORIES,
    HEURISTIC_CATEGORIES,
)


def test_all_categories_have_routes():
    for cat in Category:
        assert cat in ROUTING_MAP


def test_keep_categories():
    assert Category.INSTRUCTION in KEEP_CATEGORIES
    assert Category.CLARIFICATION in KEEP_CATEGORIES
    assert Category.CONFIRMATION in KEEP_CATEGORIES
    assert Category.INQUIRY in KEEP_CATEGORIES
    assert Category.DECISION in KEEP_CATEGORIES
    assert Category.FEEDBACK in KEEP_CATEGORIES
    assert len(KEEP_CATEGORIES) == 6


def test_distill_categories():
    assert Category.EXPLANATION in DISTILL_CATEGORIES
    assert Category.IMPLEMENTATION in DISTILL_CATEGORIES
    assert Category.REASONING in DISTILL_CATEGORIES
    assert Category.DEBUGGING in DISTILL_CATEGORIES
    assert Category.METRICS in DISTILL_CATEGORIES
    assert Category.COMPILATION in DISTILL_CATEGORIES
    assert Category.PLANNING in DISTILL_CATEGORIES
    assert Category.TESTING in DISTILL_CATEGORIES
    assert Category.GIT_OPERATION in DISTILL_CATEGORIES
    assert Category.ANALYSIS in DISTILL_CATEGORIES
    assert len(DISTILL_CATEGORIES) == 10


def test_heuristic_categories():
    assert Category.STATUS_UPDATE in HEURISTIC_CATEGORIES
    assert Category.NOTIFICATION in HEURISTIC_CATEGORIES
    assert Category.LOG_OUTPUT in HEURISTIC_CATEGORIES
    assert Category.SCAFFOLDING in HEURISTIC_CATEGORIES
    assert Category.ERROR_OUTPUT in HEURISTIC_CATEGORIES
    assert len(HEURISTIC_CATEGORIES) == 5


def test_categories_partition():
    all_cats = KEEP_CATEGORIES | DISTILL_CATEGORIES | HEURISTIC_CATEGORIES
    assert all_cats == set(Category)
    assert len(all_cats) == 21


def test_route_enum():
    assert Route.KEEP.value == "KEEP"
    assert Route.DISTILL.value == "DISTILL"
    assert Route.HEURISTIC.value == "HEURISTIC"
