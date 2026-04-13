"""Position-aware text trimming: truncate, trim_string, trim_tool_result, trim_toolUseResult."""

import json

from .compression import (
    clean_bash_text,
    dedup_system_reminders,
    entropy_ratio,
    structural_compress,
)
from .helpers import blended_limit


def truncate(text, limit, label=""):
    if not isinstance(text, str) or len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n[...{label} truncated {len(text)}->{limit}...]\n"
        + text[-half:]
    )


def trim_string(obj, key, limit, label):
    val = obj.get(key)
    if isinstance(val, str) and len(val) > limit:
        obj[key] = truncate(val, limit, label)


def _entropy_modulated_limit(text: str, limit: int) -> int:
    """Adjust truncation limit based on text entropy (information density)."""
    if not text or len(text) < 200:
        return limit
    ratio = entropy_ratio(text)
    if ratio < 0.3:
        return int(limit * 1.5)  # high info: more generous
    elif ratio > 0.5:
        return int(limit * 0.5)  # repetitive: more aggressive
    return limit


def trim_tool_result(block, tool_name, aggr, agg_lim, gen_lim):
    inner = block.get("content")
    key = (
        tool_name
        if tool_name in gen_lim
        else ("mcp" if tool_name.startswith("mcp__") else "default")
    )
    limit = blended_limit(key, aggr, agg_lim, gen_lim)
    if isinstance(inner, str):
        if tool_name == "Bash":
            inner = clean_bash_text(inner)
        inner = dedup_system_reminders(inner)
        inner = structural_compress(inner, aggr)
        limit = _entropy_modulated_limit(inner, limit)
        block["content"] = truncate(inner, limit, tool_name)
    elif isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if tool_name == "Bash":
                    text = clean_bash_text(text)
                text = dedup_system_reminders(text)
                text = structural_compress(text, aggr)
                item_limit = _entropy_modulated_limit(text, limit)
                item["text"] = truncate(text, item_limit, tool_name)


def _trim_tur_deep(tur, key, aggr, limit):
    """Trim a TUR field that may be str or dict with nested string values."""
    val = tur.get(key)
    if isinstance(val, str):
        val = structural_compress(val, aggr)
        tur[key] = truncate(val, limit, f"tur.{key}") if len(val) > limit else val
    elif isinstance(val, dict):
        # Recursively compress/truncate all string values in the dict
        total = len(json.dumps(val))
        if total > limit:
            for k, v in val.items():
                if isinstance(v, str) and len(v) > 200:
                    v = structural_compress(v, aggr)
                    val[k] = truncate(v, max(limit // 4, 200), f"tur.{key}.{k}")
                elif isinstance(v, list):
                    # Truncate long lists (e.g., message arrays in task)
                    if len(json.dumps(v)) > limit // 2:
                        val[k] = (
                            v[:3] + [{"_truncated": len(v) - 3}] if len(v) > 3 else v
                        )


def trim_toolUseResult(tur, aggr, agg_lim, gen_lim):
    if not isinstance(tur, dict):
        return
    bl = lambda k: blended_limit(k, aggr, agg_lim, gen_lim)
    # structural_compress all string fields, then truncate
    if isinstance(tur.get("originalFile"), str):
        tur["originalFile"] = structural_compress(tur["originalFile"], aggr)
    trim_string(tur, "originalFile", bl("tur.originalFile"), "tur.originalFile")
    if isinstance(tur.get("stdout"), str):
        tur["stdout"] = clean_bash_text(tur["stdout"])
        tur["stdout"] = structural_compress(tur["stdout"], aggr)
        trim_string(tur, "stdout", bl("tur.stdout"), "tur.stdout")
    if isinstance(tur.get("content"), str):
        tur["content"] = structural_compress(tur["content"], aggr)
    trim_string(tur, "content", bl("tur.content"), "tur.content")
    if isinstance(tur.get("oldString"), str):
        tur["oldString"] = structural_compress(tur["oldString"], aggr)
    trim_string(tur, "oldString", bl("tur.oldString"), "tur.oldString")
    if isinstance(tur.get("newString"), str):
        tur["newString"] = structural_compress(tur["newString"], aggr)
    trim_string(tur, "newString", bl("tur.newString"), "tur.newString")
    sp = tur.get("structuredPatch")
    if isinstance(sp, list):
        max_lines = int(20 + (1 - aggr) * 40)
        for patch in sp:
            if isinstance(patch, dict):
                pl = patch.get("lines")
                if isinstance(pl, list) and len(pl) > max_lines:
                    half = max_lines // 2
                    patch["lines"] = pl[:half] + ["[...truncated...]"] + pl[-half:]
    # Agent task/prompt/result fields
    _trim_tur_deep(tur, "task", aggr, bl("tur.content"))
    if isinstance(tur.get("prompt"), str):
        tur["prompt"] = structural_compress(tur["prompt"], aggr)
    trim_string(tur, "prompt", bl("tur.content"), "tur.prompt")
    if isinstance(tur.get("result"), str):
        tur["result"] = structural_compress(tur["result"], aggr)
    trim_string(tur, "result", bl("tur.content"), "tur.result")
    file_val = tur.get("file")
    fl = bl("tur.file")
    if isinstance(file_val, dict):
        if isinstance(file_val.get("content"), str):
            file_val["content"] = structural_compress(file_val["content"], aggr)
        trim_string(file_val, "content", fl, "tur.file.content")
    elif isinstance(file_val, str):
        tur["file"] = structural_compress(file_val, aggr)
        if len(tur["file"]) > fl:
            tur["file"] = truncate(tur["file"], fl, "tur.file")
    if isinstance(tur.get("prompt"), str):
        tur["prompt"] = structural_compress(tur["prompt"], aggr)
    if isinstance(tur.get("content"), str) and "prompt" in tur:
        trim_string(tur, "content", bl("Agent"), "tur.agent.content")
        trim_string(tur, "prompt", bl("tool_input.Agent"), "tur.agent.prompt")
