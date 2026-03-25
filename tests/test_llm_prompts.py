from reduce_session.llm.base import Category
from reduce_session.llm.prompts import (
    CLASSIFY_SYSTEM,
    DISTILL_SUMMARIZE_SYSTEM,
    DISTILL_STRIP_SYSTEM,
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
    result = parse_classify_response('["DECISION", "REASONING"]', 2)
    assert result == [Category.DECISION, Category.REASONING]


def test_parse_classify_response_markdown_fence():
    result = parse_classify_response('```json\n["DECISION"]\n```', 1)
    assert result == [Category.DECISION]


def test_parse_classify_response_wrong_count():
    result = parse_classify_response('["DECISION"]', 3)
    assert len(result) == 3


def test_parse_classify_response_invalid_json():
    result = parse_classify_response("not json", 2)
    assert len(result) == 2
    assert all(c == Category.ROUTINE for c in result)


def test_parse_classify_response_unknown_category():
    result = parse_classify_response('["DECISION", "UNKNOWN"]', 2)
    assert result[0] == Category.DECISION
    assert result[1] == Category.ROUTINE


def test_format_distill_prompt():
    text = "Let me check the file."
    for mode in ("summarize", "strip_scaffold"):
        prompt = format_distill_prompt(text, mode)
        assert text in prompt
