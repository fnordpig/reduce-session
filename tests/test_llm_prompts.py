from reduce_session.llm.base import Category, DISTILL_CATEGORIES
from reduce_session.llm.prompts import (
    CLASSIFY_SYSTEM,
    DISTILL_SUMMARIZE_SYSTEM,
    DISTILL_STRIP_SYSTEM,
    DISTILL_TYPE_PROMPTS,
    format_classify_prompt,
    parse_classify_response,
    format_distill_prompt,
)


def test_classify_system_prompt_has_all_categories():
    for cat in Category:
        assert cat.value in CLASSIFY_SYSTEM


def test_format_classify_prompt_single():
    exchanges = [{"role": "user", "text": "Yes do it", "tool_name": None}]
    prompt = format_classify_prompt(exchanges)
    assert "Exchange 1:" in prompt
    assert "[user]" in prompt


def test_format_classify_prompt_batch():
    exchanges = [
        {"role": "user", "text": "Use Metal", "tool_name": None},
        {"role": "assistant", "text": "Let me check...", "tool_name": None},
        {"role": "tool", "text": "exit code 0", "tool_name": "Bash"},
    ]
    prompt = format_classify_prompt(exchanges)
    assert "Exchange 3:" in prompt
    assert "[Bash]" in prompt


def test_format_classify_prompt_truncates_long_text():
    exchanges = [{"role": "user", "text": "x" * 1000, "tool_name": None}]
    prompt = format_classify_prompt(exchanges)
    assert "..." in prompt
    assert len(prompt) < 800


def test_parse_classify_response_valid():
    result = parse_classify_response('["INSTRUCTION", "REASONING"]', 2)
    assert result == [Category.INSTRUCTION, Category.REASONING]


def test_parse_classify_response_markdown_fence():
    result = parse_classify_response('```json\n["INSTRUCTION"]\n```', 1)
    assert result == [Category.INSTRUCTION]


def test_parse_classify_response_wrong_count():
    result = parse_classify_response('["INSTRUCTION"]', 3)
    assert len(result) == 3


def test_parse_classify_response_invalid_json():
    result = parse_classify_response("not json", 2)
    assert len(result) == 2
    assert all(c == Category.SCAFFOLDING for c in result)


def test_parse_classify_response_unknown_category():
    result = parse_classify_response('["INSTRUCTION", "UNKNOWN"]', 2)
    assert result[0] == Category.INSTRUCTION
    assert result[1] == Category.SCAFFOLDING


def test_format_distill_prompt():
    text = "Let me check the file."
    for mode in ("summarize", "strip_scaffold"):
        prompt = format_distill_prompt(text, mode)
        assert text in prompt


def test_distill_type_prompts():
    """Every DISTILL category has a type-specific prompt."""
    for cat in DISTILL_CATEGORIES:
        assert cat.value in DISTILL_TYPE_PROMPTS, (
            f"Missing DISTILL_TYPE_PROMPTS entry for {cat.value}"
        )


def test_format_distill_with_category():
    """Type-specific prompt is used when category is provided."""
    text = "Some explanation text here."
    # With category in DISTILL_TYPE_PROMPTS
    prompt = format_distill_prompt(text, "summarize", category="EXPLANATION")
    assert "conclusion" in prompt.lower()
    assert text in prompt
    # Without category — generic instruction
    generic = format_distill_prompt(text, "summarize")
    assert "compress" in generic.lower() or "shorter" in generic.lower()
    # With category but strip_scaffold mode — should NOT use type-specific
    strip = format_distill_prompt(text, "strip_scaffold", category="EXPLANATION")
    assert "strip" in strip.lower() or "novel" in strip.lower()
