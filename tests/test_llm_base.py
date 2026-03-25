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
    assert Category.DECISION in KEEP_CATEGORIES
    assert Category.PREFERENCE in KEEP_CATEGORIES
    assert Category.CORRECTION in KEEP_CATEGORIES
    assert Category.FINDING in KEEP_CATEGORIES


def test_distill_categories():
    assert Category.REASONING in DISTILL_CATEGORIES
    assert Category.IMPLEMENTATION in DISTILL_CATEGORIES
    assert Category.DIAGNOSTIC in DISTILL_CATEGORIES
    assert Category.AGENT_TRANSCRIPT in DISTILL_CATEGORIES


def test_heuristic_categories():
    assert Category.EXPLORATION in HEURISTIC_CATEGORIES
    assert Category.SCAFFOLDING in HEURISTIC_CATEGORIES
    assert Category.ROUTINE in HEURISTIC_CATEGORIES


def test_categories_partition():
    all_cats = KEEP_CATEGORIES | DISTILL_CATEGORIES | HEURISTIC_CATEGORIES
    assert all_cats == set(Category)


def test_route_enum():
    assert Route.KEEP.value == "KEEP"
    assert Route.DISTILL.value == "DISTILL"
    assert Route.HEURISTIC.value == "HEURISTIC"
